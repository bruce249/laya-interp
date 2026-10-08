"""Track 1.3 / 1.4: where option competition and order sensitivity live, and causal tests."""

from __future__ import annotations

import random

import numpy as np
import torch

from . import ablation as ab
from . import competition as cp
from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK, SEG_OPTION, SEG_QUESTION, SEG_SPECIAL, SEG_STATE, collate, to_device

CATS = ["own_option_text", "own_mask", "other_option_text", "other_masks", "question", "state", "special"]


@torch.no_grad()
def attention_summary(agent, runner, items, casting, batch_size: int = 16) -> dict:
    """Mean attention from each [MASK] (and from option text tokens) to token categories, per layer/head."""
    encs = [cp.encode_mc(agent, it, casting) for it in items]
    L = runner.n_layers
    H = max(runner.n_heads(l) for l in range(L))
    from_mask = torch.zeros(L, H, len(CATS), device=runner.device)
    from_text = torch.zeros(L, H, len(CATS), device=runner.device)
    n_mask = n_text = 0
    sites = [f"L{l}.pattern" for l in range(L)]
    for start in range(0, len(encs), batch_size):
        batch = to_device(collate(encs[start:start + batch_size], agent.tok.pad_token_id), runner.device)
        _, _, cache = runner.forward(batch, cache=sites, amp=True)
        seg, opt = batch["segment"], batch["option_ids"]
        B, T = seg.shape
        # category of key t relative to query option o
        for l in range(L):
            P = cache[f"L{l}.pattern"].float()                           # [B, H, T, T]
            for b in range(B):
                for kind, acc in (("mask", from_mask), ("text", from_text)):
                    if kind == "mask":
                        q_idx = (seg[b] == SEG_MASK).nonzero().flatten()
                    else:
                        q_idx = (seg[b] == SEG_OPTION).nonzero().flatten()
                    if len(q_idx) == 0:
                        continue
                    q_opt = opt[b, q_idx]                                 # [Q]
                    k_seg, k_opt = seg[b], opt[b]
                    same = k_opt[None, :] == q_opt[:, None]               # [Q, T]
                    masks = {
                        "own_option_text": same & (k_seg == SEG_OPTION)[None],
                        "own_mask": same & (k_seg == SEG_MASK)[None],
                        "other_option_text": ~same & (k_seg == SEG_OPTION)[None],
                        "other_masks": ~same & (k_seg == SEG_MASK)[None],
                        "question": (k_seg == SEG_QUESTION)[None].expand(len(q_idx), -1),
                        "state": (k_seg == SEG_STATE)[None].expand(len(q_idx), -1),
                        "special": ((k_seg == SEG_SPECIAL) & batch["attention_mask"][b].bool())[None]
                        .expand(len(q_idx), -1),
                    }
                    rows = P[b][:, q_idx]                                  # [H, Q, T]
                    hh = rows.shape[0]
                    for c, name in enumerate(CATS):
                        acc[l, :hh, c] += (rows * masks[name][None].float()).sum(-1).sum(-1)
                if l == 0:
                    n_mask += int((seg[b] == SEG_MASK).sum())
                    n_text += int((seg[b] == SEG_OPTION).sum())
    return {"categories": CATS, "from_mask": (from_mask / max(n_mask, 1)).cpu().numpy(),
            "from_option_text": (from_text / max(n_text, 1)).cpu().numpy()}


def _effects(base, rows):
    base, rows = np.stack(base), np.stack(rows)
    return {
        "rotation_deg": float(mt.rotation(base, rows).mean()),
        "abs_log_scale": float(np.abs(mt.log_scale(base, rows)).mean()),
        "log_scale": float(mt.log_scale(base, rows).mean()),
        "argmax_change": float((base.argmax(1) != rows.argmax(1)).mean()),
        "l1_centered": float(np.abs(mt.centered(rows) - mt.centered(base)).sum(1).mean()),
    }


def _groups(items, encs):
    """Batch items with equal k together so logit rows stack."""
    by_k: dict[int, list[int]] = {}
    for i, e in enumerate(encs):
        by_k.setdefault(e.k, []).append(i)
    return by_k


def head_screen(agent, runner, items, casting, kind: str, **kw) -> dict:
    """kind='cross': block cross-option attention per head. kind='rope': order-blind RoPE per head
    (measures reversal flips). kind='mean': mean-ablate each head. Returns arrays [L, H]."""
    encs = [cp.encode_mc(agent, it, casting) for it in items]
    L = runner.n_layers if kind != "rope" else runner.n_enc
    H = runner.H
    res = {k: np.zeros((L, H)) for k in ("rotation_deg", "abs_log_scale", "log_scale", "argmax_change",
                                          "l1_centered", "flip_rate", "flip_delta")}
    base = slot_logits(runner, encs, amp=True, **kw)
    means = None
    rev_encs = rev_orders = base_flip = None
    if kind == "rope":
        rev_orders = [list(range(e.k))[::-1] for e in encs]
        rev_encs = [cp.encode_mc(agent, it, casting, option_order=o) for it, o in zip(items, rev_orders)]
        rev = cp.option_logits(runner, rev_encs, rev_orders, amp=True, **kw)
        base_flip = np.mean([np.argmax(a) != np.argmax(b) for a, b in zip(base, rev)])
    if kind == "mean":
        means = ab.segment_means(runner, encs)
    by_k = _groups(items, encs)
    for l in range(L):
        for h in range(runner.n_heads(l)):
            if kind == "cross":
                fk = dict(cross_option_heads={l: [h]})
            elif kind == "rope":
                fk = dict(rope_alt_heads={l: [h]})
            else:
                fk = dict(edits=ab.mean_heads(means, {l: [h]}))
            rows = slot_logits(runner, encs, amp=True, **fk, **kw)
            eff = {}
            for k_, idx in by_k.items():
                e = _effects([base[i] for i in idx], [rows[i] for i in idx])
                for name, v in e.items():
                    eff.setdefault(name, []).append((v, len(idx)))
            for name, vals in eff.items():
                res[name][l, h] = sum(v * n for v, n in vals) / sum(n for _, n in vals)
            if kind == "rope":
                rv = cp.option_logits(runner, rev_encs, rev_orders, amp=True, **fk, **kw)
                canon = [r for r in rows]
                fr = np.mean([np.argmax(a) != np.argmax(b) for a, b in zip(canon, rv)])
                res["flip_rate"][l, h] = fr
                res["flip_delta"][l, h] = fr - base_flip
    out = {k: v for k, v in res.items()}
    out["base_flip"] = base_flip
    return out


@torch.no_grad()
def info_flow(agent, runner, items, casting, min_effect: float = 0.3, max_pairs: int = 400) -> dict:
    """Patch clean residuals into the j-filled run at layer l and position set; recovery of s_i."""
    fid = cp.filler_token(agent.tok)
    pairs = []
    encs = [cp.encode_mc(agent, it, casting) for it in items]
    base = slot_logits(runner, encs, amp=True)
    fills, owners = [], []
    for n, (it, e) in enumerate(zip(items, encs)):
        g = it["gold"]
        wrong = [m for m in range(e.k) if m != g]
        j = max(wrong, key=lambda m: base[n][m])
        for (i, jj) in ((g, j), (j, g)):
            fills.append(cp.with_filler(e, jj, fid))
            owners.append((n, i, jj))
    filled = slot_logits(runner, fills, amp=True)
    for (n, i, j), row in zip(owners, filled):
        delta = base[n][i] - row[i]
        if abs(delta) >= min_effect:
            pairs.append((n, i, j, fills[owners.index((n, i, j))]))
    pairs = pairs[:max_pairs]
    sets = ["j_span", "mask_i", "i_text", "other_options", "shared"]
    L = runner.n_layers
    rec = {s: np.zeros((L, len(pairs))) for s in sets}
    bs = 32
    for start in range(0, len(pairs), bs):
        chunk = pairs[start:start + bs]
        clean_b = to_device(collate([encs[n] for n, *_ in chunk], agent.tok.pad_token_id), runner.device)
        corr_b = to_device(collate([f for *_, f in chunk], agent.tok.pad_token_id), runner.device)
        sites = [f"L{l}.resid_pre" for l in range(L)]
        lc, _, cache = runner.forward(clean_b, cache=sites, amp=True)
        lx, _, _ = runner.forward(corr_b, amp=True)
        opt = clean_b["option_ids"]
        seg = clean_b["segment"]
        pos = {}
        idx_i = torch.tensor([i for _, i, _, _ in chunk], device=runner.device)
        idx_j = torch.tensor([j for _, _, j, _ in chunk], device=runner.device)
        pos["j_span"] = opt == (idx_j[:, None] + 1)
        own_i = opt == (idx_i[:, None] + 1)
        pos["mask_i"] = own_i & (seg == SEG_MASK)
        pos["i_text"] = own_i & (seg == SEG_OPTION)
        pos["other_options"] = (opt > 0) & ~pos["j_span"] & ~own_i
        pos["shared"] = (opt == 0) & clean_b["attention_mask"].bool()
        rows = torch.arange(len(chunk), device=runner.device)
        clean_i = lc[rows, idx_i]
        corr_i = lx[rows, idx_i]
        for l in range(L):
            for s in sets:
                m = pos[s]
                src = cache[f"L{l}.resid_pre"]
                edit = {f"L{l}.resid_pre": lambda x, b, m=m, src=src: torch.where(m[..., None], src, x)}
                lp, _, _ = runner.forward(corr_b, edits=edit, amp=True)
                r = (lp[rows, idx_i] - corr_i) / (clean_i - corr_i)
                rec[s][l, start:start + len(chunk)] = r.float().cpu().numpy()
    return {"n_pairs": len(pairs), "sets": sets,
            "recovery_mean": {s: np.nanmean(v, 1).tolist() for s, v in rec.items()},
            "recovery_ci": {s: [mt.ci_dict(v[l], n=500) for l in range(L)] for s, v in rec.items()}}


def top_heads(score: np.ndarray, k: int, n_layers: int | None = None) -> dict[int, list[int]]:
    flat = [(score[l, h], l, h) for l in range(score.shape[0]) for h in range(score.shape[1])]
    flat.sort(reverse=True)
    sel: dict[int, list[int]] = {}
    for _, l, h in flat[:k]:
        sel.setdefault(l, []).append(h)
    return sel


def random_heads(runner, k: int, seed: int, layers=None) -> dict[int, list[int]]:
    rng = random.Random(seed)
    layers = list(range(runner.n_layers)) if layers is None else list(layers)
    allh = [(l, h) for l in layers for h in range(runner.n_heads(l))]
    sel: dict[int, list[int]] = {}
    for l, h in rng.sample(allh, k):
        sel.setdefault(l, []).append(h)
    return sel


def evaluate_condition(agent, runner, items, casting, n_inter: int = 500, **kw) -> dict:
    b = cp.baseline(agent, runner, items, casting, amp=True, **kw)
    r = cp.interaction(agent, runner, items[:n_inter], casting, with_removal=False, amp=True, **kw)
    return {"accuracy": b["accuracy"]["est"], "reversal_flip_rate": b["reversal_flip_rate"]["est"],
            "ece_raw": b["ece_raw"], "mean_abs_M_centered": r["mean_abs_M_centered"]["est"],
            "mean_abs_M_raw": r["mean_abs_M_raw"]["est"]}
