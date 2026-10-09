"""Phase 2 (plan v4): a causal account of C3, label shortcut vs evidence retrieval.

E1  label {consistent, swapped} x evidence {real, filler, contradicting}; joint patching
E2  timing: block [MASK] -> label in one layer band at a time, and "allow only one band"
E3  evidence path: which readers of the passage are necessary; restoration from a clean run
E4  confirmation on untouched human-written sets (RTE, MultiRC) and adversarial label wordings,
    with and without the label guard, including calibration (ECE, Brier, NLL)
Primary metrics (frozen): label-following rate, accuracy with normal labels, swap gap.
"""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import torch

from . import labels as lb
from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK, SEG_QUESTION, SEG_SPECIAL, SEG_STATE, collate, to_device

ADVERSARIAL = {
    "correct_incorrect": {"false": "incorrect", "true": "correct"},
    "agree_disagree": {"false": "disagree", "true": "agree"},
    "valid_invalid": {"false": "invalid", "true": "valid"},
    "supported_unsupported": {"false": "unsupported", "true": "supported"},
    "multi_token": {"false": "no, it does not", "true": "yes, it does"},
    "the_case": {"false": "not the case", "true": "the case"},
}


def swapped(labels: dict) -> dict:
    return {"false": labels["true"], "true": labels["false"]}


ALL_LABELS = {**{k: v[0] for k, v in lb.LABELS.items()}, **ADVERSARIAL}


# ----- data -----

def build_noul_confirm(tok, seed: int = 11, n_multirc: int = 500, max_state: int = 350) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)
    out = []
    for r in load_dataset("nyu-mll/glue", "rte", split="validation"):
        hyp = r["sentence2"].strip().rstrip(".")
        out.append({"id": f"RTE:{r['idx']}", "source": "RTE", "state": r["sentence1"],
                    "question": f"Does the text imply that {hyp[:1].lower() + hyp[1:]}?",
                    "gold": int(r["label"] == 0), "split": "confirm"})
    rows = list(load_dataset("aps/super_glue", "multirc", split="validation"))
    rng.shuffle(rows)
    seen = set()
    for r in rows:
        key = (r["paragraph"][:80], r["question"])
        if key in seen or len(tok(r["paragraph"], add_special_tokens=False)["input_ids"]) > max_state:
            continue
        seen.add(key)
        out.append({"id": f"MultiRC:{r['idx']['paragraph']}:{r['idx']['question']}:{r['idx']['answer']}",
                    "source": "MultiRC", "state": r["paragraph"],
                    "question": f"{r['question'].strip()} Is \"{r['answer'].strip()}\" a correct answer?",
                    "gold": int(r["label"]), "split": "confirm"})
        if sum(o["source"] == "MultiRC" for o in out) >= n_multirc:
            break
    return out


CONTRA_SYSTEM = ("You edit passages for a reading-comprehension dataset. Make the smallest edit to the passage so "
                 "that the correct answer to the yes/no question flips. Keep the length and wording as close as "
                 'possible. Reply as JSON: {"edited": "..."}')
VERIFY_SYSTEM = ('Answer the yes/no question using only the passage. Reply as JSON: {"answer": true|false|null}')


def build_contradictions(items: list[dict], careful: str, bulk: str) -> list[dict]:
    from . import llm

    edits = llm.ask_many(careful, CONTRA_SYSTEM,
                         [f"Passage: {it['state']}\nQuestion: {it['question']}\nCurrent answer: "
                          f"{'yes' if it['gold'] else 'no'}" for it in items])
    cand = []
    for it, e in zip(items, edits):
        try:
            text = json.loads(e)["edited"].strip()
            if text and text != it["state"].strip():
                cand.append((it, text))
        except Exception:
            pass
    checks = llm.ask_many(bulk, VERIFY_SYSTEM, [f"Passage: {t}\nQuestion: {it['question']}" for it, t in cand])
    out = []
    for (it, text), c in zip(cand, checks):
        try:
            ans = json.loads(c).get("answer")
        except Exception:
            ans = None
        if isinstance(ans, bool) and int(ans) == 1 - it["gold"]:
            out.append(dict(it, contra_state=text))
    return out


# ----- helpers -----

def calib(lo: np.ndarray, gold: np.ndarray, temp: float) -> dict:
    p = 1 / (1 + np.exp(-lo / temp))
    conf = np.maximum(p, 1 - p)
    correct = (p > 0.5) == gold.astype(bool)
    return {"accuracy": float(correct.mean()), "ece": mt.ece(conf, correct),
            "brier": float(np.mean((p - gold) ** 2)),
            "nll": float(-np.mean(np.log(np.clip(np.where(gold == 1, p, 1 - p), 1e-12, None))))}


def logodds(agent, runner, items, labels, state_key="state", option_order=None, **kw):
    encs = [lb.encode_item(agent, it, labels, option_order, state=it.get(state_key)) for it in items]
    return lb.holds_logodds(slot_logits(runner, encs, **kw), encs)


def guard_blocks(runner):
    local = [l for l in range(runner.n_enc) if runner.layer_types[l] != "full_attention"]
    return [(local, lb.edge_masks("mask_to_labels"))]


def behaviour_cells(agent, runner, items, labels, **kw) -> dict:
    gold = np.array([it["gold"] for it in items])
    lo_n = logodds(agent, runner, items, labels, **kw)
    lo_s = logodds(agent, runner, items, swapped(labels), **kw)
    acc_n, acc_s = (lo_n > 0) == gold, (lo_s > 0) == gold
    return {"acc_normal": float(acc_n.mean()), "acc_swapped": float(acc_s.mean()),
            "label_following": float((acc_n & ~acc_s)[acc_n].mean()) if acc_n.any() else float("nan"),
            "swap_gap": float(acc_n.mean() - acc_s.mean()), "_lo_normal": lo_n}


# ----- E1 -----

def e1(agent, runner, items, log=print) -> dict:
    """items carry contra_state. Cells: label x evidence; decision measured against each passage's answer."""
    normal, swap = lb.LABELS["semantic_yn"]
    gold = np.array([it["gold"] for it in items])
    filler_word = " ..."
    filler_items = [dict(it, filler_state=" ".join([filler_word.strip()] * max(1, len(it["state"].split()))))
                    for it in items]
    res = {}
    for lname, labs in (("consistent", normal), ("swapped", swap)):
        for ename, key, target in (("real", "state", gold), ("contra", "contra_state", 1 - gold),
                                   ("filler", "filler_state", None)):
            lo = logodds(agent, runner, filler_items, labs, state_key=key)
            cell = {"mean_logodds_holds": float(lo.mean()), "frac_holds": float((lo > 0).mean())}
            if target is not None:
                cell["follows_passage"] = float(((lo > 0) == target).mean())
            res[f"{lname}/{ename}"] = cell
    # label pull with no evidence: shift in P(holds) when label words swap under filler
    res["label_pull_no_evidence"] = res["swapped/filler"]["frac_holds"] - res["consistent/filler"]["frac_holds"]
    log(f"[E1] {json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()}) for k, v in res.items()})}")
    return res


@torch.no_grad()
def e1_joint_patch(agent, runner, items, batch_size: int = 32) -> dict:
    """clean = (consistent labels, real passage); corrupt = (swapped labels, filler passage).
    Patch label tokens, state tokens, both, question tokens or [MASK]s at layer l; recovery of holds log-odds."""
    normal, swap = lb.LABELS["semantic_yn"]
    filler = agent.tok(" ...", add_special_tokens=False)["input_ids"][0]
    L = runner.n_layers
    sets = ["label", "state", "label+state", "question", "masks"]
    rec = {s: [[] for _ in range(L)] for s in sets}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        en = [lb.encode_item(agent, it, normal) for it in chunk]
        es = [lb.encode_item(agent, it, swap) for it in chunk]
        keep = [i for i in range(len(chunk)) if len(en[i].ids) == len(es[i].ids)]
        en, es = [en[i] for i in keep], [es[i] for i in keep]
        ec = []
        for e in es:
            f = copy.copy(e)
            f.ids = list(e.ids)
            for t in range(e.state_span[0], e.state_span[1]):
                f.ids[t] = filler
            ec.append(f)
        bn = to_device(collate(en, agent.tok.pad_token_id), runner.device)
        bc = to_device(collate(ec, agent.tok.pad_token_id), runner.device)
        sites = [f"L{l}.resid_pre" for l in range(L)]
        ln, _, cache = runner.forward(bn, cache=sites)
        lc, _, _ = runner.forward(bc)
        lo = lambda x: (x[:, 1] - x[:, 0]).float()  # noqa: E731
        clean, corr = lo(ln), lo(lc)
        seg = bn["segment"]
        pos = {"label": bn["label_mask"], "state": seg == SEG_STATE,
               "label+state": bn["label_mask"] | (seg == SEG_STATE), "question": seg == SEG_QUESTION,
               "masks": seg == SEG_MASK}
        for l in range(L):
            src = cache[f"L{l}.resid_pre"]
            for s, m in pos.items():
                e = {f"L{l}.resid_pre": lambda x, b, m=m, src=src: torch.where(m[..., None], src, x)}
                lp, _, _ = runner.forward(bc, edits=e)
                rec[s][l].append(lb._ratio(lo(lp), corr, clean))
    return {s: {"median": [float(np.nanmedian(np.concatenate(v))) for v in per]} for s, per in rec.items()}


# ----- E2 -----

def e2(agent, runner, items, log=print) -> dict:
    n = runner.n_enc
    glob = [l for l in range(n) if runner.layer_types[l] == "full_attention"]
    bands = {"L0-4": range(0, 5), "L5-9": range(5, 10), "L10-14": range(10, 15), "L15-20": range(15, 21),
             "L21-27": range(21, n), "head": range(n, runner.n_layers)}
    normal, _ = lb.LABELS["semantic_yn"]
    tf, _ = lb.LABELS["semantic_tf"]
    mk = lb.edge_masks("mask_to_labels")
    out = {}
    for lname, labs in (("yes/no", normal), ("true/false", tf)):
        res = {"none": behaviour_cells(agent, runner, items, labs)}
        for b, layers in bands.items():
            res[f"block {b}"] = behaviour_cells(agent, runner, items, labs, edge_blocks=[(list(layers), mk)])
            allow = [l for l in range(runner.n_layers) if l not in layers]
            res[f"allow only {b}"] = behaviour_cells(agent, runner, items, labs, edge_blocks=[(allow, mk)])
        mid = list(bands["L10-14"])
        res["block L10-14 local only"] = behaviour_cells(agent, runner, items, labs,
                                                         edge_blocks=[([l for l in mid if l not in glob], mk)])
        res["block L10-14 global only"] = behaviour_cells(agent, runner, items, labs,
                                                          edge_blocks=[([l for l in mid if l in glob], mk)])
        res["block all"] = behaviour_cells(agent, runner, items, labs,
                                           edge_blocks=[(list(range(runner.n_layers)), mk)])
        for k, v in res.items():
            log(f"[E2 {lname}] {k}: follow={v['label_following']:.3f} acc_n={v['acc_normal']:.3f} "
                f"gap={v['swap_gap']:+.3f}")
        out[lname] = {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in res.items()}
    return out


# ----- E3 -----

def reader_mask(kind: str):
    def f(b):
        seg, opt = b["segment"], b["option_ids"]
        st = (seg == SEG_STATE)[:, None, :]
        real = b["attention_mask"].bool()
        q = {"masks": seg == SEG_MASK, "options": opt > 0, "question": seg == SEG_QUESTION,
             "special": (seg == SEG_SPECIAL) & real, "question+special": ((seg == SEG_QUESTION) | (seg == SEG_SPECIAL)) & real,
             "all_nonstate": (seg != SEG_STATE) & real}[kind]
        return q[:, :, None] & st
    return f


@torch.no_grad()
def e3(agent, runner, items, log=print, batch_size: int = 32) -> dict:
    normal, swap = lb.LABELS["semantic_yn"]
    L = list(range(runner.n_layers))
    out = {}
    for guard in (False, True):
        g = guard_blocks(runner) if guard else []
        res = {"none": behaviour_cells(agent, runner, items, normal, edge_blocks=g or None)}
        for kind in ("masks", "options", "question", "special", "question+special", "all_nonstate"):
            res[f"block {kind}->state"] = behaviour_cells(agent, runner, items, normal,
                                                          edge_blocks=g + [(L, reader_mask(kind))])
        for k, v in res.items():
            log(f"[E3 guard={guard}] {k}: acc_n={v['acc_normal']:.3f} follow={v['label_following']:.3f}")
        out["guard" if guard else "plain"] = {k: {a: b for a, b in v.items() if not a.startswith("_")}
                                              for k, v in res.items()}
    # restoration: filler-state run, patch the clean residual at reader positions, per layer
    filler = agent.tok(" ...", add_special_tokens=False)["input_ids"][0]
    Ls = runner.n_layers
    sets = ["question", "special", "masks", "options", "state"]
    rec = {s: [[] for _ in range(Ls)] for s in sets}
    for start in range(0, len(items), batch_size):
        en = [lb.encode_item(agent, it, normal) for it in items[start:start + batch_size]]
        ef = []
        for e in en:
            f = copy.copy(e)
            f.ids = list(e.ids)
            for t in range(e.state_span[0], e.state_span[1]):
                f.ids[t] = filler
            ef.append(f)
        bn = to_device(collate(en, agent.tok.pad_token_id), runner.device)
        bf = to_device(collate(ef, agent.tok.pad_token_id), runner.device)
        sites = [f"L{l}.resid_pre" for l in range(Ls)]
        ln, _, cache = runner.forward(bn, cache=sites)
        lf, _, _ = runner.forward(bf)
        lo = lambda x: (x[:, 1] - x[:, 0]).float()  # noqa: E731
        seg, opt = bn["segment"], bn["option_ids"]
        real = bn["attention_mask"].bool()
        pos = {"question": seg == SEG_QUESTION, "special": (seg == SEG_SPECIAL) & real, "masks": seg == SEG_MASK,
               "options": (opt > 0) & (seg != SEG_MASK), "state": seg == SEG_STATE}
        for l in range(Ls):
            src = cache[f"L{l}.resid_pre"]
            for s, m in pos.items():
                e = {f"L{l}.resid_pre": lambda x, b, m=m, src=src: torch.where(m[..., None], src, x)}
                lp, _, _ = runner.forward(bf, edits=e)
                rec[s][l].append(lb._ratio(lo(lp), lo(lf), lo(ln)))
    out["restoration"] = {s: [float(np.nanmedian(np.concatenate(v))) for v in per] for s, per in rec.items()}
    for s, v in out["restoration"].items():
        log(f"[E3 restore {s}] {[round(x, 2) for x in v]}")
    return out


# ----- E4 -----

def e4(agent, runner, sets: dict, log=print) -> dict:
    out = {}
    temp = float(agent.temperature_by_options.get("noul:2", agent.temperature[2]))
    for sname, items in sets.items():
        gold = np.array([it["gold"] for it in items])
        res = {}
        for lname, labs in ALL_LABELS.items():
            row = {}
            for tag, kw in (("plain", {}), ("guard", {"edge_blocks": guard_blocks(runner)})):
                cell = behaviour_cells(agent, runner, items, labs, **kw)
                cal = calib(cell.pop("_lo_normal"), gold, temp)
                row[tag] = {**cell, **{f"normal_{k}": v for k, v in cal.items()}}
            res[lname] = row
            p, g = row["plain"], row["guard"]
            log(f"[E4 {sname}] {lname}: follow {p['label_following']:.3f}->{g['label_following']:.3f} "
                f"acc_n {p['acc_normal']:.3f}->{g['acc_normal']:.3f} ece {p['normal_ece']:.3f}->{g['normal_ece']:.3f} "
                f"brier {p['normal_brier']:.3f}->{g['normal_brier']:.3f}")
        out[sname] = res
    return out


# ----- drivers -----

def build_data(log=print) -> dict:
    import os

    from . import data
    from .model import load_agent

    tok = load_agent("laya", device="cpu", eager_attn=False).tok
    if not (data.DATA / "noul_confirm.jsonl").exists():
        data.save_jsonl("noul_confirm.jsonl", build_noul_confirm(tok))
    if not (data.DATA / "boolq_contra.jsonl").exists():
        main = [it for it in data.load_jsonl("boolq.jsonl") if it["split"] == "main"][:300]
        data.save_jsonl("boolq_contra.jsonl", build_contradictions(main, os.environ["LLM_CAREFUL"],
                                                                   os.environ["LLM_BULK"]))
    conf = data.load_jsonl("noul_confirm.jsonl")
    contra = data.load_jsonl("boolq_contra.jsonl")
    out = {"noul_confirm": {s: sum(it["source"] == s for it in conf) for s in ("RTE", "MultiRC")},
           "contra_kept": len(contra)}
    log(f"[phase2 data] {out}")
    return out


def run(agent, runner, model: str, log=print) -> dict:
    from . import data

    runner.force_fp32 = True
    boolq = data.load_jsonl("boolq.jsonl")
    main = [it for it in boolq if it["split"] == "main"]
    held = [it for it in boolq if it["split"] == "heldout"]
    conf = data.load_jsonl("noul_confirm.jsonl")
    contra = data.load_jsonl("boolq_contra.jsonl")
    llm_held = [it for it in data.load_jsonl("c3_llm.jsonl") if it["split"] == "heldout"]
    out = {"model": model, "precision": "fp32", "n_contra": len(contra)}
    out["E1"] = e1(agent, runner, contra, log)
    out["E1_joint_patch"] = e1_joint_patch(agent, runner, main[:300])
    for s, v in out["E1_joint_patch"].items():
        log(f"[E1 patch {s}] {[round(x, 2) for x in v['median']]}")
    out["E2_explore"] = e2(agent, runner, main, log)
    out["E2_confirm"] = e2(agent, runner, conf, log)
    out["E3"] = e3(agent, runner, main[:300], log)
    out["E4"] = e4(agent, runner, {"boolq_heldout": held, "RTE": [c for c in conf if c["source"] == "RTE"],
                                   "MultiRC": [c for c in conf if c["source"] == "MultiRC"],
                                   "llm_heldout": llm_held}, log)
    runner.force_fp32 = False
    return out


def guard2_blocks(runner):
    """Exploratory extension: also block option text from reading the label words (local layers)."""
    local = [l for l in range(runner.n_enc) if runner.layer_types[l] != "full_attention"]
    return [(local, lb.edge_masks("mask_to_labels")), (local, lb.edge_masks("options_to_labels"))]


def e4b(agent, runner, log=print) -> dict:
    from . import data

    runner.force_fp32 = True
    boolq = data.load_jsonl("boolq.jsonl")
    conf = data.load_jsonl("noul_confirm.jsonl")
    sets = {"boolq_heldout": [it for it in boolq if it["split"] == "heldout"],
            "RTE": [c for c in conf if c["source"] == "RTE"], "MultiRC": [c for c in conf if c["source"] == "MultiRC"],
            "llm_heldout": [it for it in data.load_jsonl("c3_llm.jsonl") if it["split"] == "heldout"]}
    temp = float(agent.temperature_by_options.get("noul:2", agent.temperature[2]))
    out = {}
    for sname, items in sets.items():
        gold = np.array([it["gold"] for it in items])
        res = {}
        for lname in ("semantic_yn", "semantic_tf", "multi_token", "correct_incorrect", "symbolic_ab"):
            row = {}
            for tag, kw in (("guard", {"edge_blocks": guard_blocks(runner)}),
                            ("guard2", {"edge_blocks": guard2_blocks(runner)})):
                cell = behaviour_cells(agent, runner, items, ALL_LABELS[lname], **kw)
                cal = calib(cell.pop("_lo_normal"), gold, temp)
                row[tag] = {**cell, **{f"normal_{k}": v for k, v in cal.items()}}
            res[lname] = row
            a, b = row["guard"], row["guard2"]
            log(f"[E4b {sname}] {lname}: follow guard {a['label_following']:.3f} guard2 {b['label_following']:.3f} | "
                f"acc_n {a['acc_normal']:.3f}->{b['acc_normal']:.3f} brier {a['normal_brier']:.3f}->{b['normal_brier']:.3f}")
        out[sname] = res
    runner.force_fp32 = False
    return out
