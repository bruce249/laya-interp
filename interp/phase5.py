"""Phase 5 (plan v4): can label semantics in the training data create the label shortcut?

Same init, data, loss (RLCD), budget and evaluation; only the noul label regime changes. R0 is the existing
RLCD controls (default true/false labels). See NEXT_STEPS_PLAN.md for the frozen design and prediction.
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np

SEMANTIC_PAIRS = [("false", "true"), ("no", "yes"), ("incorrect", "correct"), ("disagree", "agree"),
                  ("invalid", "valid"), ("unsupported", "supported")]
SYMBOL_PAIRS = [("◇", "△"), ("※", "§"), ("¤", "¶"), ("⬡", "⬢"), ("♢", "♤"), ("∴", "∵")]
NEUTRAL_CRIT = {"false": "this option", "true": "this option"}
REGIMES = ("semantic", "neutral", "random_mapping", "mixture", "planted",
           "dose_d0", "dose_d1", "dose_d2", "dose_d3", "baseline")
# Phase 8 dose response (shuffled noul order in every dose arm): (descriptions, P(label word correct))
DOSE = {"dose_d0": ("informative", 0.5), "dose_d1": ("informative", 1.0),
        "dose_d2": ("neutral", 0.75), "dose_d3": ("neutral", 0.9)}
SHUFFLED = ("planted", "dose_d0", "dose_d1", "dose_d2", "dose_d3")


def noul_q(base_q: dict, regime: str, rng: random.Random) -> dict:
    q = dict(base_q)
    if regime == "baseline":
        return q
    if regime in DOSE:
        desc, p_correct = DOSE[regime]
        f, t = rng.choice(SEMANTIC_PAIRS)
        if rng.random() >= p_correct:
            f, t = t, f
        q["labels"] = {"false": f, "true": t}
        if desc == "neutral":
            q["criteria"] = dict(NEUTRAL_CRIT)
        return q
    if regime == "mixture":
        regime = rng.choice(["semantic", "neutral", "random_mapping"])
    if regime == "neutral":
        f, t = rng.choice(SYMBOL_PAIRS)
    else:
        f, t = rng.choice(SEMANTIC_PAIRS)
        if regime == "random_mapping" and rng.random() < 0.5:
            f, t = t, f
    q["labels"] = {"false": f, "true": t}
    if regime == "planted":
        q["criteria"] = dict(NEUTRAL_CRIT)
    return q


def train_regime(regime: str, seed: int, log=print, encoder: str | None = None, tag: str = "p5",
                 encoder_lr: float | None = None) -> dict:
    import torch
    from laya.common import build_model
    from laya.train import TrainConfig, save_checkpoint, train_model

    from .model import load_agent
    from .retrain import CKPT, build_items, train_rows

    t0 = time.time()
    ref = load_agent("laya", device="cpu", eager_attn=False)
    tok, cfg = ref.tok, copy.deepcopy(ref.cfg)
    del ref
    if encoder:                                   # external architecture (Phase 8, P5)
        from transformers import AutoTokenizer

        cfg["encoder"] = encoder
        tok = AutoTokenizer.from_pretrained(encoder)
    rng = random.Random(1000 * seed + REGIMES.index(regime))
    rows = []
    for r in train_rows(0):
        if r["q"]["type"] == "noul":
            r = dict(r, q=noul_q(r["q"], regime, rng))
        rows.append(r)
    items, skipped = build_items(tok, rows, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    log(f"[p5 {regime} s{seed}] items={len(items)} skipped={skipped}")
    torch.manual_seed(seed)
    model = build_model(cfg, pretrained=True)
    config = TrainConfig(epochs=2, micro_batch=32, grad_accum=2, loss="rlcd", seed=seed, log_every=400,
                         shuffle_options=("noul",) if regime in SHUFFLED else (),
                         **({"encoder_lr": encoder_lr} if encoder_lr else {}))
    history = train_model(model, tok, items, config, torch.device("cuda"),
                          cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    out_cfg = dict(cfg)
    out_cfg.update({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}, "model_name": f"{tag}-{regime}",
                    "training": {"loss": "rlcd", "regime": regime, "seed": seed, "epochs": 2, "items": len(items),
                                 "encoder": cfg["encoder"], "encoder_lr": config.encoder_lr}})
    name = f"{tag}_{regime}_s{seed}"
    save_checkpoint(model, tok, out_cfg, str(CKPT / name))
    log(f"[p5 {regime} s{seed}] saved, loss={history} ({time.time() - t0:.0f}s)")
    return {"name": name, "history": history, "items": len(items), "skipped": skipped}


def evaluate(name: str, log=print) -> dict:
    from . import data
    from . import labels as lb
    from .hooks import Runner
    from .model import load_agent
    from .phase2 import ALL_LABELS, behaviour_cells, guard_blocks

    agent = load_agent(name, device="cuda")
    runner = Runner(agent)
    runner.force_fp32 = True
    boolq = data.load_jsonl("boolq.jsonl")
    conf = data.load_jsonl("noul_confirm.jsonl")
    sets = {"boolq_main": [it for it in boolq if it["split"] == "main"],
            "boolq_heldout": [it for it in boolq if it["split"] == "heldout"],
            "RTE": [c for c in conf if c["source"] == "RTE"],
            "MultiRC": [c for c in conf if c["source"] == "MultiRC"]}
    out = {"model": name}
    for sname, items in sets.items():
        res = {}
        for lname in ("semantic_yn", "semantic_tf", "correct_incorrect", "neutral_sym"):
            cell = behaviour_cells(agent, runner, items, ALL_LABELS[lname])
            cell.pop("_lo_normal")
            res[lname] = cell
        g = behaviour_cells(agent, runner, items, ALL_LABELS["semantic_yn"], edge_blocks=guard_blocks(runner))
        g.pop("_lo_normal")
        res["semantic_yn_guard"] = g
        # neutral descriptions: the label word is the only cue
        gold = np.array([it["gold"] for it in items])
        labs = ALL_LABELS["semantic_yn"]
        enc = lambda L: [lb.encode_item(agent, it, L, criteria=dict(NEUTRAL_CRIT)) for it in items]  # noqa: E731
        from .hooks import slot_logits
        e_n, e_s = enc(labs), enc({"false": labs["true"], "true": labs["false"]})
        lo_n = lb.holds_logodds(slot_logits(runner, e_n), e_n)
        lo_s = lb.holds_logodds(slot_logits(runner, e_s), e_s)
        an, asw = (lo_n > 0) == gold, (lo_s > 0) == gold
        res["semantic_yn_neutral_desc"] = {"acc_normal": float(an.mean()), "acc_swapped": float(asw.mean()),
                                           "label_following": float((an & ~asw)[an].mean()) if an.any() else None}
        out[sname] = res
        log(f"[p5 eval {name} {sname}] yn follow={res['semantic_yn']['label_following']:.3f} "
            f"tf={res['semantic_tf']['label_following']:.3f} ci={res['correct_incorrect']['label_following']:.3f} "
            f"sym={res['neutral_sym']['label_following']:.3f} guard={g['label_following']:.3f} "
            f"neutral-desc={res['semantic_yn_neutral_desc']['label_following']:.3f} "
            f"acc_n={res['semantic_yn']['acc_normal']:.3f}")
    return out
