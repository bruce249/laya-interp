"""Track 3: label-following in noul answers (C3).

An item is a BoolQ passage (state), its question (instructions) and the gold answer. noul options are
always rendered in slot order [false, true] as "<false label>: no, the statement does not hold" and
"<true label>: yes, the statement holds". The *label cue* is the label word. In the swapped
configuration each option carries the opposite label word, so a model that answers from the label word
picks the option the state contradicts. Semantic labels (true/false, yes/no) carry meaning; symbolic
(A/B) and neutral (◇/△) labels do not, so a swap effect there is symbol or position bias.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK, SEG_STATE, collate, encode, to_device

LABELS = {
    "semantic_tf": ({"false": "false", "true": "true"}, {"false": "true", "true": "false"}),
    "semantic_yn": ({"false": "no", "true": "yes"}, {"false": "yes", "true": "no"}),
    "symbolic_ab": ({"false": "B", "true": "A"}, {"false": "A", "true": "B"}),
    "neutral_sym": ({"false": "◇", "true": "△"}, {"false": "△", "true": "◇"}),
}
PAD_SENTENCE = "The following text is provided for reference. "


def load_boolq(tok, n: int = 1000, max_state_tokens: int = 150, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    rows = list(load_dataset("google/boolq", split="validation"))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows):
        if len(tok(r["passage"], add_special_tokens=False)["input_ids"]) > max_state_tokens:
            continue
        q = r["question"].strip()
        q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
        out.append({"id": f"boolq{i}", "state": r["passage"], "question": q, "gold": int(bool(r["answer"]))})
        if len(out) >= n:
            break
    for j, it in enumerate(out):
        it["split"] = "heldout" if j % 10 < 3 else "main"
    return out


def label_positions(tok, enc) -> list[int]:
    """Token positions of each option's label word: after its [MASK], up to the ':'."""
    pos = []
    for a, b in enc.option_spans:
        for t in range(a + 1, b):
            piece = tok.decode([enc.ids[t]])
            if ":" in piece:
                break
            pos.append(t)
    return pos


def encode_item(agent, item, labels: dict, option_order=None, pad_tokens: int = 0, state=None):
    text = state if state is not None else item["state"]
    if pad_tokens:
        pad_ids = agent.tok(PAD_SENTENCE, add_special_tokens=False)["input_ids"]
        reps = max(1, pad_tokens // len(pad_ids))
        text = PAD_SENTENCE * reps + text
    qdef = {"type": "noul", "instructions": item["question"], "labels": labels}
    if option_order is not None:
        qdef["option_order"] = option_order
    enc = encode(agent, text, qdef, meta={"id": item["id"]})
    enc.meta["masks"] = {"label_mask": label_positions(agent.tok, enc)}
    return enc


def holds_logodds(rows, encs) -> np.ndarray:
    """log-odds of the 'statement holds' option (option index 1), whatever slot it sits in."""
    out = []
    for r, e in zip(rows, encs):
        order = e.option_order or [0, 1]
        s = np.empty(2)
        s[np.asarray(order)] = r
        out.append(s[1] - s[0])
    return np.array(out)


def behaviour(agent, runner, items, pad_tokens: int = 0, **kw) -> dict:
    gold = np.array([it["gold"] for it in items])
    res = {}
    for name, (normal, swapped) in LABELS.items():
        for order_name, order in (("slots_default", None), ("slots_swapped", [1, 0])):
            lo = {}
            for cond, labels in (("normal", normal), ("swapped", swapped)):
                encs = [encode_item(agent, it, labels, order, pad_tokens) for it in items]
                lo[cond] = holds_logodds(slot_logits(runner, encs, amp=True, **kw), encs)
            acc_n = (lo["normal"] > 0) == gold
            acc_s = (lo["swapped"] > 0) == gold
            follows = acc_n & ~acc_s
            res[f"{name}/{order_name}"] = {
                "acc_normal": float(acc_n.mean()), "acc_swapped": float(acc_s.mean()),
                "label_following_rate": float(follows[acc_n].mean()) if acc_n.any() else float("nan"),
                "mean_logodds_shift_toward_label": float(np.mean(np.where(gold == 1, -1, 1) *
                                                                 (lo["swapped"] - lo["normal"]))),
                "_lo_normal": lo["normal"], "_lo_swapped": lo["swapped"]}
    return res


def by_strength(res_entry, gold, q: int = 4) -> list[dict]:
    lo_n, lo_s = res_entry["_lo_normal"], res_entry["_lo_swapped"]
    strength = np.abs(lo_n)
    edges = np.quantile(strength, np.linspace(0, 1, q + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (strength >= lo) & (strength <= hi)
        acc_n = ((lo_n > 0) == gold)[sel]
        acc_s = ((lo_s > 0) == gold)[sel]
        out.append({"strength_range": [float(lo), float(hi)], "n": int(sel.sum()),
                    "label_following_rate": float((acc_n & ~acc_s)[acc_n].mean()) if acc_n.any() else None})
    return out


@torch.no_grad()
def patch_curves(agent, runner, items, label_set: str = "semantic_tf", batch_size: int = 32) -> dict:
    """(a) labels: patch normal-label run into swapped-label run. (b) state: patch real state into a
    filler-state run. Per layer and position set; recovery of the holds log-odds."""
    normal, swapped = LABELS[label_set]
    L = runner.n_layers
    filler = agent.tok(" ...", add_special_tokens=False)["input_ids"][0]
    sets_a = ["label", "masks", "description", "state"]
    sets_b = ["state", "masks", "labels_and_desc"]
    rec_a = {s: [] for s in sets_a}
    rec_b = {s: [] for s in sets_b}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        en = [encode_item(agent, it, normal) for it in chunk]
        es = [encode_item(agent, it, swapped) for it in chunk]
        keep = [i for i in range(len(chunk)) if len(en[i].ids) == len(es[i].ids)]
        en, es = [en[i] for i in keep], [es[i] for i in keep]
        ef = []
        for e in en:
            f = copy.copy(e)
            f.ids = list(e.ids)
            for t in range(e.state_span[0], e.state_span[1]):
                f.ids[t] = filler
            ef.append(f)
        bn = to_device(collate(en, agent.tok.pad_token_id), runner.device)
        bs = to_device(collate(es, agent.tok.pad_token_id), runner.device)
        bf = to_device(collate(ef, agent.tok.pad_token_id), runner.device)
        sites = [f"L{l}.resid_pre" for l in range(L)]
        ln, _, cache = runner.forward(bn, cache=sites, amp=True)
        ls, _, _ = runner.forward(bs, amp=True)
        lf, _, _ = runner.forward(bf, amp=True)
        lo = lambda x: (x[:, 1] - x[:, 0]).float()  # noqa: E731  (default slot order)
        clean, corr_s, corr_f = lo(ln), lo(ls), lo(lf)
        seg = bn["segment"]
        opt_tok = (bn["option_ids"] > 0) & (seg != SEG_MASK)
        pos_a = {"label": bn["label_mask"], "masks": seg == SEG_MASK,
                 "description": opt_tok & ~bn["label_mask"], "state": seg == SEG_STATE}
        pos_b = {"state": seg == SEG_STATE, "masks": seg == SEG_MASK, "labels_and_desc": opt_tok}
        for l in range(L):
            src = cache[f"L{l}.resid_pre"]
            for s, m in pos_a.items():
                e = {f"L{l}.resid_pre": lambda x, b, m=m, src=src: torch.where(m[..., None], src, x)}
                lp, _, _ = runner.forward(bs, edits=e, amp=True)
                rec_a[s].append(_ratio(lo(lp), corr_s, clean))
            for s, m in pos_b.items():
                e = {f"L{l}.resid_pre": lambda x, b, m=m, src=src: torch.where(m[..., None], src, x)}
                lp, _, _ = runner.forward(bf, edits=e, amp=True)
                rec_b[s].append(_ratio(lo(lp), corr_f, clean))
    # rec lists are [chunk0 L0..L(L-1), chunk1 L0..] -> regroup by layer
    def regroup(rec):
        out = {}
        for s, chunks in rec.items():
            per_layer = [[] for _ in range(L)]
            for idx, arr in enumerate(chunks):
                per_layer[idx % L].append(arr)
            vals = [np.concatenate(v) for v in per_layer]
            # drop pairs whose clean-corrupt gap is tiny (ratio unstable)
            out[s] = {"median": [float(np.nanmedian(v)) for v in vals],
                      "mean_clipped": [float(np.nanmean(np.clip(v, -2, 3))) for v in vals]}
        return out
    return {"labels_patch": regroup(rec_a), "state_patch": regroup(rec_b)}


def _ratio(patched, corrupt, clean, min_gap: float = 0.5):
    gap = clean - corrupt
    r = (patched - corrupt) / gap
    r = torch.where(gap.abs() < min_gap, torch.full_like(r, float("nan")), r)
    return r.cpu().numpy()


def edge_masks(kind: str):
    """Functions batch -> blocked [B, T, T] for the knockouts in 3.3 / 3.4."""
    def mask_to_labels(b):
        q = b["segment"] == SEG_MASK
        return q[:, :, None] & b["label_mask"][:, None, :]

    def options_to_labels(b):
        q = (b["option_ids"] > 0)
        return q[:, :, None] & b["label_mask"][:, None, :] & ~b["label_mask"][:, :, None]

    def mask_to_state(b):
        q = b["segment"] == SEG_MASK
        return q[:, :, None] & (b["segment"] == SEG_STATE)[:, None, :]

    def options_to_state(b):
        q = b["option_ids"] > 0
        return q[:, :, None] & (b["segment"] == SEG_STATE)[:, None, :]

    return {"mask_to_labels": mask_to_labels, "options_to_labels": options_to_labels,
            "mask_to_state": mask_to_state, "options_to_state": options_to_state}[kind]


def knockouts(agent, runner, items, label_set: str = "semantic_tf") -> dict:
    n = runner.n_enc
    glob = [l for l in range(n) if runner.layer_types[l] == "full_attention"]
    loc = [l for l in range(n) if l not in glob]
    head = list(range(n, runner.n_layers))
    groups = {"local": loc, "global": glob, "head": head, "all": list(range(runner.n_layers))}
    normal, swapped = LABELS[label_set]
    gold = np.array([it["gold"] for it in items])
    en = [encode_item(agent, it, normal) for it in items]
    es = [encode_item(agent, it, swapped) for it in items]

    def measure(**kw):
        lo_n = holds_logodds(slot_logits(runner, en, amp=True, **kw), en)
        lo_s = holds_logodds(slot_logits(runner, es, amp=True, **kw), es)
        acc_n, acc_s = (lo_n > 0) == gold, (lo_s > 0) == gold
        return {"acc_normal": float(acc_n.mean()), "acc_swapped": float(acc_s.mean()),
                "label_following_rate": float((acc_n & ~acc_s)[acc_n].mean()) if acc_n.any() else None,
                "swap_gap": float(acc_n.mean() - acc_s.mean())}

    out = {"base": measure()}
    for edge in ("mask_to_labels", "options_to_labels", "mask_to_state", "options_to_state"):
        for gname, layers in groups.items():
            out[f"{edge}@{gname}"] = measure(edge_blocks=[(layers, edge_masks(edge))])
    out["label_path_all"] = measure(edge_blocks=[(groups["all"], edge_masks("options_to_labels")),
                                                 (groups["all"], edge_masks("mask_to_labels"))])
    return out


def distance_sweep(agent, runner, items, pads=(0, 64, 128, 256)) -> dict:
    gold = np.array([it["gold"] for it in items])
    out = {}
    for pad in pads:
        res = {}
        for name in ("semantic_tf", "symbolic_ab"):
            normal, swapped = LABELS[name]
            en = [encode_item(agent, it, normal, pad_tokens=pad) for it in items]
            es = [encode_item(agent, it, swapped, pad_tokens=pad) for it in items]
            lo_n = holds_logodds(slot_logits(runner, en, amp=True), en)
            lo_s = holds_logodds(slot_logits(runner, es, amp=True), es)
            acc_n, acc_s = (lo_n > 0) == gold, (lo_s > 0) == gold
            res[name] = {"acc_normal": float(acc_n.mean()), "acc_swapped": float(acc_s.mean()),
                         "label_following_rate": float((acc_n & ~acc_s)[acc_n].mean()) if acc_n.any() else None,
                         "mean_abs_logodds": float(np.abs(lo_n).mean())}
        out[pad] = res
    return out
