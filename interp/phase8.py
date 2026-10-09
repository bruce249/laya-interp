"""Phase 8 (plan v4, second review): confirmatory label-evidence benchmark, matched intervention
controls, metric definitions with paired transitions, and reporting helpers."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch

from . import labels as lb
from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK, SEG_OPTION, SEG_QUESTION, SEG_SPECIAL, SEG_STATE

# frozen wordings (plan v4, Phase 8, P1)
WORDINGS = {
    "yes_no": {"false": "no", "true": "yes"},
    "true_false": {"false": "false", "true": "true"},
    "right_wrong": {"false": "wrong", "true": "right"},
    "accurate_inaccurate": {"false": "inaccurate", "true": "accurate"},
    "confirmed_unconfirmed": {"false": "unconfirmed", "true": "confirmed"},
    "multi_right": {"false": "no, that is wrong", "true": "yes, that is right"},
    "multi_case": {"false": "it is not the case", "true": "it is the case"},
    "neutral_sym": {"false": "◇", "true": "△"},
    "letters_ab": {"false": "B", "true": "A"},
}
NEW_WORDINGS = ("right_wrong", "accurate_inaccurate", "confirmed_unconfirmed")
MULTI = ("multi_right", "multi_case")


def swap(labels):
    return {"false": labels["true"], "true": labels["false"]}


# ----------------------------------------------------------------- data (P1)

def build_benchmark(tok, careful: str, bulk: str, seed: int = 21, n_each: int = 300, max_state: int = 300,
                    log=print) -> list[dict]:
    from datasets import load_dataset

    from . import data
    from .phase2 import build_contradictions

    rng = random.Random(seed)
    items = []
    rows = [r for r in load_dataset("nyu-mll/glue", "mnli", split="validation_matched") if r["label"] in (0, 2)]
    rng.shuffle(rows)
    for r in rows[:n_each]:
        hyp = r["hypothesis"].strip().rstrip(".")
        items.append({"id": f"MNLI:{r['idx']}", "source": "MNLI", "state": r["premise"],
                      "question": f"Does the text imply that {hyp[:1].lower() + hyp[1:]}?",
                      "gold": int(r["label"] == 0), "split": "confirm2"})
    rows = list(load_dataset("nyu-mll/glue", "qnli", split="validation"))
    rng.shuffle(rows)
    k = 0
    for r in rows:
        q = r["question"].strip()
        items.append({"id": f"QNLI:{r['idx']}", "source": "QNLI", "state": r["sentence"],
                      "question": f"Does the text answer the question \"{q}\"?",
                      "gold": int(r["label"] == 0), "split": "confirm2"})
        k += 1
        if k >= n_each:
            break
    contra = {it["id"]: it["contra_state"] for it in build_contradictions(items, careful, bulk)}
    for it in items:
        if it["id"] in contra:
            it["contra_state"] = contra[it["id"]]
    log(f"[p8 data] items={len(items)} with contradicting edit={len(contra)}")
    return items


def freeze(items) -> str:
    blob = "\n".join(json.dumps(it, ensure_ascii=False, sort_keys=True) for it in items).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------- interventions (P2)

def _local(runner):
    return [l for l in range(runner.n_enc) if runner.layer_types[l] != "full_attention"]


def _per_item_random(b, pool_mask, count_mask, seed):
    """For each row, pick as many tokens from pool_mask as count_mask has True (fixed seed per row)."""
    out = torch.zeros_like(pool_mask)
    for i in range(pool_mask.shape[0]):
        pool = torch.nonzero(pool_mask[i]).flatten().tolist()
        n = int(count_mask[i].sum())
        if pool and n:
            rng = random.Random(seed + i)
            for t in rng.sample(pool, min(n, len(pool))):
                out[i, t] = True
    return out


def edge_fn(kind: str, seed: int = 0):
    def f(b):
        seg, opt, lab = b["segment"], b["option_ids"], b["label_mask"]
        real = b["attention_mask"].bool()
        is_mask = seg == SEG_MASK
        if kind == "label":                                    # original guard
            return is_mask[:, :, None] & lab[:, None, :]
        if kind == "label_ext":                                # + option text -> labels
            q = (opt > 0) & ~lab
            return q[:, :, None] & lab[:, None, :]
        if kind == "random_option":                            # same count, random non-label option tokens
            pool = (opt > 0) & ~lab & ~is_mask
            sel = _per_item_random(b, pool, lab, seed)
            return is_mask[:, :, None] & sel[:, None, :]
        if kind == "description":                              # same count, description tokens
            pool = (seg == SEG_OPTION) & ~lab
            sel = _per_item_random(b, pool, lab, seed + 7)
            return is_mask[:, :, None] & sel[:, None, :]
        if kind == "evidence":                                 # question -> passage
            return (seg == SEG_QUESTION)[:, :, None] & (seg == SEG_STATE)[:, None, :]
        if kind == "sham":                                     # [MASK] -> [CLS]
            cls = torch.zeros_like(real)
            cls[:, 0] = True
            return is_mask[:, :, None] & cls[:, None, :]
        raise ValueError(kind)
    return f


def interventions(runner) -> dict:
    local = _local(runner)
    allL = list(range(runner.n_layers))
    return {
        "none": None,
        "guard": [(local, edge_fn("label"))],
        "guard_extended": [(local, edge_fn("label")), (local, edge_fn("label_ext"))],
        "random_edges": [(local, edge_fn("random_option"))],
        "description_edges": [(local, edge_fn("description"))],
        "evidence_block": [(allL, edge_fn("evidence"))],
        "sham": [(local, edge_fn("sham"))],
    }


# ----------------------------------------------------------------- scoring

def logodds(agent, runner, items, labels, state_key="state", edge_blocks=None):
    encs = [lb.encode_item(agent, it, labels, state=it.get(state_key)) for it in items]
    rows = slot_logits(runner, encs, edge_blocks=edge_blocks) if edge_blocks else slot_logits(runner, encs)
    return lb.holds_logodds(rows, encs)


def transitions(lo_n, lo_s, target) -> dict:
    cn, cs = (lo_n > 0) == target, (lo_s > 0) == target
    cc, cw = int((cn & cs).sum()), int((cn & ~cs).sum())
    wc, ww = int((~cn & cs).sum()), int((~cn & ~cs).sum())
    n_correct = cc + cw
    follow = cw / n_correct if n_correct else float("nan")
    boot = []
    rng = np.random.default_rng(0)
    idx = np.where(cn)[0]
    if len(idx):
        flips = (~cs[idx]).astype(float)
        boot = [flips[rng.integers(0, len(idx), len(idx))].mean() for _ in range(1000)]
    return {"n": int(len(target)), "correct_correct": cc, "correct_wrong": cw, "wrong_correct": wc,
            "wrong_wrong": ww, "label_following": follow, "denominator": n_correct,
            "label_following_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))] if boot else None,
            "acc_normal": float(cn.mean()), "acc_swapped": float(cs.mean()),
            "swap_gap": float(cn.mean() - cs.mean())}


def calibration(lo, gold, temp):
    p = 1 / (1 + np.exp(-lo / temp))
    conf = np.maximum(p, 1 - p)
    correct = (p > 0.5) == gold.astype(bool)
    return {"ece": mt.ece(conf, correct), "brier": float(np.mean((p - gold) ** 2))}


def evaluate_benchmark(agent, runner, items, model: str, log=print, inter_names=None) -> dict:
    runner.force_fp32 = True
    gold = np.array([it["gold"] for it in items])
    contra_items = [it for it in items if it.get("contra_state")]
    gold_c = 1 - np.array([it["gold"] for it in contra_items])
    temp = float(agent.temperature_by_options.get("noul:2", agent.temperature[2]))
    blank = [dict(it, blank_state=" ".join(["..."] * max(1, len(it["state"].split())))) for it in items]
    out = {"model": model, "n_items": len(items), "n_contra": len(contra_items)}
    ivs = interventions(runner)
    for iname in (inter_names or list(ivs)):
        eb = ivs[iname]
        res = {}
        for wname, labs in WORDINGS.items():
            if iname != "none" and wname not in ("yes_no", "true_false", "right_wrong", "multi_right", "multi_case",
                                                 "neutral_sym"):
                continue
            lo_n = logodds(agent, runner, items, labs, edge_blocks=eb)
            lo_s = logodds(agent, runner, items, swap(labs), edge_blocks=eb)
            cell = {"real": transitions(lo_n, lo_s, gold), "normal_calibration": calibration(lo_n, gold, temp)}
            if wname in ("yes_no", "true_false") and iname in ("none", "guard", "guard_extended"):
                cn = logodds(agent, runner, contra_items, labs, "contra_state", eb)
                cs = logodds(agent, runner, contra_items, swap(labs), "contra_state", eb)
                cell["contra"] = transitions(cn, cs, gold_c)
                bn = logodds(agent, runner, blank, labs, "blank_state", eb)
                bs = logodds(agent, runner, blank, swap(labs), "blank_state", eb)
                cell["blank"] = {"frac_holds_normal": float((bn > 0).mean()), "frac_holds_swapped": float((bs > 0).mean())}
            res[wname] = cell
            r = cell["real"]
            log(f"[p8 {model} {iname}] {wname}: follow={r['label_following']:.3f} (n={r['denominator']}) "
                f"acc_n={r['acc_normal']:.3f} gap={r['swap_gap']:+.3f}"
                + (f" | contra follow={cell['contra']['label_following']:.3f}" if "contra" in cell else ""))
        out[iname] = res
    runner.force_fp32 = False
    return out


def safety_under_interventions(agent, runner, model: str, log=print) -> dict:
    from . import data
    from . import safety as sf

    runner.force_fp32 = True
    nat = [r for r in data.load_jsonl("jb_natural.jsonl") if r["split"] == "heldout"]
    inj = [r for r in data.load_jsonl("injection.jsonl") if r["split"] == "heldout"]
    out = {}
    for iname, eb in interventions(runner).items():
        res = {}
        for name, rows, q in (("jailbreak", nat, sf.JAILBREAK_Q), ("injection", inj, sf.INJECTION_Q)):
            encs = [lb.encode_item(agent, {"state": r["text"], "question": q["instructions"], "id": r["id"]},
                                   {"false": "false", "true": "true"}) for r in rows]
            lo = lb.holds_logodds(slot_logits(runner, encs, edge_blocks=eb) if eb else slot_logits(runner, encs), encs)
            y = np.array([r["label"] for r in rows])
            res[name] = {"auroc": mt.auroc(lo, y), "accuracy": float(((lo > 0) == y).mean())}
        out[iname] = res
        log(f"[p8 safety {model} {iname}] jb auroc={res['jailbreak']['auroc']:.3f} inj auroc={res['injection']['auroc']:.3f}")
    runner.force_fp32 = False
    return out
