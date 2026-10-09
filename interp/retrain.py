"""Stretch: same init, same data, fixed option order; only the objective differs (soft-ce vs rlcd)."""

from __future__ import annotations

import copy
import json
import pathlib
import random
import time

CKPT = pathlib.Path("/store/ckpt")


def train_rows(seed: int = 0, qasc_options: int = 4) -> list[dict]:
    """(state, qdef, gold) from the *train* splits; the eval sets use test / validation splits."""
    from datasets import load_dataset

    from .data import CASTINGS, _clean

    rng = random.Random(seed)
    rows = []

    def add_mc(question, options, gold, source):
        options = [_clean(o) for o in options]
        if len(set(options)) != len(options) or not (3 <= len(options) <= 5):
            return
        ins, state = CASTINGS["generic"](_clean(question))
        rows.append({"state": state, "q": {"type": "choice", "instructions": ins,
                                           "criteria": {o: None for o in options}},
                     "gold": gold, "source": source})

    for config in ("ARC-Challenge", "ARC-Easy"):
        for split in ("train", "validation"):
            for r in load_dataset("allenai/ai2_arc", config, split=split):
                labels = r["choices"]["label"]
                if r["answerKey"] in labels:
                    add_mc(r["question"], r["choices"]["text"], labels.index(r["answerKey"]), config)
    for r in load_dataset("tau/commonsense_qa", split="train"):
        labels = r["choices"]["label"]
        add_mc(r["question"], r["choices"]["text"], labels.index(r["answerKey"]), "CSQA")
    for r in load_dataset("allenai/qasc", split="train"):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        gold = labels.index(r["answerKey"])
        keep = sorted(rng.sample([i for i in range(len(texts)) if i != gold], qasc_options - 1) + [gold])
        add_mc(r["question"], [texts[i] for i in keep], keep.index(gold), "QASC")
    for r in load_dataset("google/boolq", split="train"):
        q = r["question"].strip()
        q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
        rows.append({"state": r["passage"], "q": {"type": "noul", "instructions": q},
                     "gold": int(bool(r["answer"])), "source": "BoolQ"})
    rng.shuffle(rows)
    return rows


def build_items(tok, rows, max_len: int = 512, head_max_len: int = 192):
    from laya.train import encode_state, make_item, to_internal

    items, skipped = [], 0
    for r in rows:
        q = to_internal("q", r["q"])
        k = len(r["q"]["criteria"]) if r["q"]["type"] == "choice" else 2
        target = [0.0] * k
        target[r["gold"]] = 1.0
        item, why = make_item(tok, q, target, encode_state(tok, r["state"], max_len), head_max_len, max_len)
        if item is None:
            skipped += 1
        else:
            items.append(item)
    return items, skipped


def train_arm(loss: str, epochs: int = 2, seed: int = 0, log=print) -> dict:
    import torch
    from laya.common import build_model
    from laya.train import TrainConfig, save_checkpoint, train_model

    from .model import load_agent

    t0 = time.time()
    ref = load_agent("laya", device="cpu", eager_attn=False)
    tok, cfg = ref.tok, copy.deepcopy(ref.cfg)
    del ref
    rows = train_rows(0)          # same data for every seed; only training randomness varies
    items, skipped = build_items(tok, rows, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    log(f"[retrain {loss}] items={len(items)} skipped={skipped} ({time.time() - t0:.0f}s)")
    torch.manual_seed(seed)
    model = build_model(cfg, pretrained=True)        # fresh head on answerdotai/ModernBERT-large
    config = TrainConfig(epochs=epochs, micro_batch=32, grad_accum=2, loss=loss, shuffle_options=(), seed=seed,
                         log_every=200)
    history = train_model(model, tok, items, config, torch.device("cuda"),
                          cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    out_cfg = dict(cfg)
    out_cfg.update({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}, "model_name": f"ctrl-{loss}",
                    "training": {"loss": loss, "epochs": epochs, "items": len(items), "fixed_option_order": True,
                                 "init": cfg["encoder"], "seed": seed}})
    suffix = "" if seed == 0 else f"_s{seed}"
    path = CKPT / f"ctrl_{loss.replace('-', '_')}{suffix}"
    save_checkpoint(model, tok, out_cfg, str(path))
    log(f"[retrain {loss}] saved {path} loss history={history} ({time.time() - t0:.0f}s)")
    return {"loss": loss, "history": history, "items": len(items), "path": str(path),
            "wall_s": round(time.time() - t0, 1)}


def evaluate_arm(name: str, log=print) -> dict:
    """Track 1 (order, interaction, parallel controls), C2 essentials and C3 behaviour on a control."""
    import numpy as np

    from . import ablation as ab
    from . import confidence as cf
    from . import competition as cp
    from . import data
    from . import labels as lb
    from .hooks import Runner, slot_logits
    from .model import load_agent
    from .runs import mc_items, public, save

    agent = load_agent(name, device="cuda")
    runner = Runner(agent)
    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    out: dict = {"model": name}
    b = cp.baseline(agent, runner, main, "generic", amp=True)
    out["baseline"] = {k: b[k] for k in ("accuracy", "reversal_flip_rate", "ece_raw")}
    om = cp.order_map(agent, runner, main, "generic", amp=True)
    out["order_map"] = {k: om[k] for k in ("reversal_flip_rate", "any_order_flip_rate",
                                           "order_variance_explained_by_slot")}
    inter = cp.interaction(agent, runner, main[:600], "generic", with_removal=False, amp=True)
    out["interaction"] = {k: inter[k] for k in ("mean_abs_M_centered", "symmetry_corr_centered",
                                                "gold_margin_change_when_top_distractor_filled")}
    conds = cp.parallel_conditions(runner)
    out["parallel"] = {}
    for c in ("options_blocked_seqpos", "shipped_parallel_positions"):
        bb = cp.baseline(agent, runner, main, "generic", amp=True, **conds[c])
        out["parallel"][c] = {"accuracy": bb["accuracy"]["est"], "reversal_flip_rate": bb["reversal_flip_rate"]["est"]}
    log(f"[{name}] acc={b['accuracy']['est']:.3f} flip={b['reversal_flip_rate']['est']:.3f} "
        f"ece={b['ece_raw']:.3f} |M|c={inter['mean_abs_M_centered']['est']:.3f} "
        f"slotR2={om['order_variance_explained_by_slot']:.3f} parallel={out['parallel']}")

    # C2: MLP screen and greedy B over MLPs only
    encs = [cp.encode_mc(agent, it, "generic") for it in main]
    means = ab.segment_means(runner, encs[:600])
    mlp = cf.mlp_screen(agent, runner, main[:400], "generic", means)
    base = slot_logits(runner, encs[:400], amp=True)
    order = list(np.argsort(-mlp["abs_log_scale"]))
    chosen, cur = [], None
    for l in order[:12]:
        trial = chosen + [{"kind": "mlp", "layer": int(l)}]
        rows = slot_logits(runner, encs[:400], amp=True, edits=cf.set_edits(trial, means))
        e = cf.per_item_effects(base, rows)
        s = {"abs_ls": float(np.abs(e["log_scale"]).mean()), "kept": float(1 - e["argmax_change"].mean())}
        if s["kept"] >= 0.95 and (cur is None or s["abs_ls"] > cur["abs_ls"] + 1e-3):
            chosen, cur = trial, s
    full_base = slot_logits(runner, encs, amp=True)
    rows = slot_logits(runner, encs, amp=True, edits=cf.set_edits(chosen, means)) if chosen else full_base
    out["greedy_B_mlps"] = {"layers": [c["layer"] for c in chosen],
                            "effect": public(cf.summarize_effect(full_base, rows, [it["gold"] for it in main],
                                                                 [e.temperature for e in encs]))}
    out["mlp_screen_abs_log_scale"] = mlp["abs_log_scale"].tolist()
    oc = cf.overconfidence(full_base, [it["gold"] for it in main], main, np.zeros(len(main)))
    out["overconfidence"] = {"global_T": oc["global_T"], "T_by_norm_quintile": [q["T"] for q in oc["by_norm_quintile"]]}
    log(f"[{name}] greedy B mlps={out['greedy_B_mlps']['layers']} "
        f"|ls|={out['greedy_B_mlps']['effect']['abs_log_scale']['est']:.3f} T={oc['global_T']:.2f}")

    # C3 label-following
    bq = [it for it in data.load_jsonl("boolq.jsonl") if it["split"] == "main"]
    beh = lb.behaviour(agent, runner, bq)
    out["label_following"] = {k: {m: v[m] for m in ("acc_normal", "acc_swapped", "label_following_rate")}
                              for k, v in beh.items() if k.endswith("slots_default")}
    log(f"[{name}] label-following { {k: round(v['label_following_rate'], 3) for k, v in out['label_following'].items()} }")
    save(f"retrain/{name}", out)
    return out


def summary_json(path: str) -> str:
    return json.dumps(json.loads(pathlib.Path(path).read_text()), indent=1)
