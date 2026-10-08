"""Follow-ups: fp32 confirmation, held-out replication, greedy C2 sets, steering sweep."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import metrics as mt
from .runs import RESULTS, mc_items, public, save


def _json(rel):
    p = RESULTS / (rel if rel.endswith(".json") else rel + ".json")
    return json.loads(p.read_text()) if p.exists() else None


# ----- 1. fp32 confirmation -----


def fp32_confirm(agent, runner, log=print) -> dict:
    from . import ablation as ab
    from . import competition as cp
    from . import confidence as cf
    from . import data
    from . import labels as lb
    from . import safety as sf
    from .hooks import slot_logits
    from .inputs import encode

    runner.force_fp32, runner.eager = True, True
    out: dict = {}
    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    conds = cp.parallel_conditions(runner)
    t1 = {}
    for name in ("sequential", "options_blocked_seqpos", "shipped_parallel_positions", "masked_parallel_all"):
        kw = dict(conds[name])
        b = cp.baseline(agent, runner, main, "generic", **kw)
        r = cp.interaction(agent, runner, main[:600], "generic", with_removal=False, **kw)
        t1[name] = {"accuracy": b["accuracy"]["est"], "reversal_flip_rate": b["reversal_flip_rate"]["est"],
                    "mean_abs_M_centered": r["mean_abs_M_centered"]["est"]}
        log(f"[fp32] t1 {name}: {t1[name]}")
    out["track1"] = t1

    t2 = _json("track2/laya")
    encs = [cp.encode_mc(agent, it, "generic") for it in main]
    means = ab.segment_means(runner, encs[:600])
    base = slot_logits(runner, encs)
    gold = [it["gold"] for it in main]
    temps = [e.temperature for e in encs]
    out["track2_base"] = cf.calibration_block(base, gold, temps)
    greedy = _json("followups/greedy_sets") or {}
    sets = {"A10": t2["sets"]["A10"], "B10": t2["sets"]["B10"]}
    if greedy.get("B_set"):
        sets["greedyB"] = greedy["B_set"]
    if greedy.get("A_set"):
        sets["greedyA"] = greedy["A_set"]
    for key, cset in sets.items():
        rows = slot_logits(runner, encs, edits=cf.set_edits(cset, means))
        out[f"track2_{key}"] = public(cf.summarize_effect(base, rows, gold, temps))
        v = out[f"track2_{key}"]
        log(f"[fp32] t2 {key}: |ls|={v['abs_log_scale']['est']:.3f} kept={v['argmax_unchanged']:.3f} "
            f"ece={v['ece_raw']:.3f} auroc={v['auroc_margin']:.3f}")

    bq = [it for it in data.load_jsonl("boolq.jsonl") if it["split"] == "main"]
    beh = lb.behaviour(agent, runner, bq)
    out["track3_behaviour"] = public({k: v for k, v in beh.items() if k.endswith("slots_default")})
    for k, v in out["track3_behaviour"].items():
        log(f"[fp32] t3 {k}: n/s={v['acc_normal']:.3f}/{v['acc_swapped']:.3f} follow={v['label_following_rate']:.3f}")
    ko = lb.knockouts(agent, runner, bq, "semantic_tf")
    out["track3_knockout_mask_to_labels_local"] = ko["mask_to_labels@local"]
    out["track3_knockout_base"] = ko["base"]
    log(f"[fp32] t3 knockout base={ko['base']} local={ko['mask_to_labels@local']}")

    nat = data.load_jsonl("jb_natural.jsonl")
    te = [r for r in nat if r["split"] == "heldout"]
    tr = [r for r in nat if r["split"] == "main"]
    yte = np.array([r["label"] for r in te])
    te_encs = [encode(agent, r["text"], sf.JAILBREAK_Q) for r in te]
    m_jb = ab.segment_means(runner, [encode(agent, r["text"], sf.JAILBREAK_Q) for r in tr[:600]])
    b0 = sf.log_odds(slot_logits(runner, te_encs))
    rows = slot_logits(runner, te_encs, edits=cf.set_edits(sets.get("greedyB", t2["sets"]["B10"]), m_jb))
    lo = sf.log_odds(rows)
    out["track4"] = {"auroc": mt.auroc(b0, yte), "B_set": "greedyB" if "greedyB" in sets else "B10",
                     "B_auroc": mt.auroc(lo, yte),
                     "B_abs_logodds_ratio": float(np.mean(np.abs(lo)) / np.mean(np.abs(b0))),
                     "B_decision_flips": float(((lo > 0) != (b0 > 0)).mean())}
    log(f"[fp32] t4 {out['track4']}")
    runner.force_fp32, runner.eager = False, False
    return out


# ----- 2. held-out replication for Track 3 -----


def heldout_track3(agent, runner, log=print) -> dict:
    from . import data
    from . import labels as lb

    held = [it for it in data.load_jsonl("boolq.jsonl") if it["split"] == "heldout"]
    out = {"patch_semantic_tf": lb.patch_curves(agent, runner, held, "semantic_tf"),
           "knockouts_semantic_tf": lb.knockouts(agent, runner, held, "semantic_tf"),
           "distance": lb.distance_sweep(agent, runner, held)}
    pc = out["patch_semantic_tf"]
    log(f"[heldout t3] label->masks {[round(x, 2) for x in pc['labels_patch']['masks']['median']]}")
    log(f"[heldout t3] state->masks {[round(x, 2) for x in pc['state_patch']['masks']['median']]}")
    ko = out["knockouts_semantic_tf"]
    log(f"[heldout t3] base={ko['base']} local={ko['mask_to_labels@local']} global={ko['mask_to_labels@global']}")
    log(f"[heldout t3] distance={ {p: round(v['semantic_tf']['label_following_rate'], 3) for p, v in out['distance'].items()} }")
    return out


# ----- 3. greedy C2 sets -----


def greedy_sets(agent, runner, log=print, K: int = 12, pool: int = 60) -> dict:
    """B: maximise |log scale| keeping argmax >= 95%. A: maximise argmax changes keeping |log scale| <= 0.1."""
    from . import ablation as ab
    from . import competition as cp
    from . import confidence as cf
    from .hooks import slot_logits

    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    search = main[:400]
    encs = [cp.encode_mc(agent, it, "generic") for it in search]
    means = ab.segment_means(runner, [cp.encode_mc(agent, it, "generic") for it in main[:600]])
    base = slot_logits(runner, encs, amp=True)
    pk = __import__("pickle").load((RESULTS / "track2/laya.pkl").open("rb"))
    comps = pk["components"]

    def score(cset):
        rows = slot_logits(runner, encs, amp=True, edits=cf.set_edits(cset, means))
        e = cf.per_item_effects(base, rows)
        return {"abs_ls": float(np.abs(e["log_scale"]).mean()), "ls": float(e["log_scale"].mean()),
                "kept": float(1 - e["argmax_change"].mean()), "rot": float(e["rotation_deg"].mean())}

    out = {}
    cand_b = sorted(comps, key=lambda c: c["abs_log_scale"], reverse=True)[:pool]
    cand_a = sorted([c for c in comps if c["abs_log_scale"] <= np.median([x["abs_log_scale"] for x in comps])],
                    key=lambda c: c["argmax_change"], reverse=True)[:pool]
    for name, cands, objective, ok in (
            ("B", cand_b, lambda s: s["abs_ls"], lambda s: s["kept"] >= 0.95),
            ("A", cand_a, lambda s: 1 - s["kept"], lambda s: s["abs_ls"] <= 0.10)):
        chosen, trace, cur = [], [], None
        for _ in range(K):
            best, best_s = None, None
            for c in cands:
                if c in chosen:
                    continue
                s = score(chosen + [c])
                if ok(s) and (best_s is None or objective(s) > objective(best_s)):
                    best, best_s = c, s
            if best is None or (cur is not None and objective(best_s) <= objective(cur) + 1e-3):
                break
            chosen.append(best)
            cur = best_s
            trace.append({"added": {k: best[k] for k in ("kind", "layer") if k in best} |
                          ({"head": best["head"]} if "head" in best else {}), **best_s})
            log(f"[greedy {name}] +{trace[-1]['added']} -> {best_s}")
        out[f"{name}_set"] = chosen
        out[f"{name}_trace"] = trace
        for split, subset in (("main", main), ("heldout", held)):
            es = [cp.encode_mc(agent, it, "generic") for it in subset]
            b0 = slot_logits(runner, es, amp=True)
            rows = slot_logits(runner, es, amp=True, edits=cf.set_edits(chosen, means))
            res = cf.summarize_effect(b0, rows, [it["gold"] for it in subset], [e.temperature for e in es])
            out[f"{name}_{split}"] = public(res)
            log(f"[greedy {name}] {split}: |ls|={res['abs_log_scale']['est']:.3f} kept={res['argmax_unchanged']:.3f} "
                f"rot={res['rotation_deg']['est']:.1f} ece={res['ece_raw']:.3f} auroc={res['auroc_margin']:.3f}")
        # size-matched random control
        rng = np.random.default_rng(0)
        rnd = [comps[i] for i in rng.choice(len(comps), max(1, len(chosen)), replace=False)]
        out[f"{name}_random_control"] = score(rnd)
    return out


# ----- 4. steering sweep -----


@torch.no_grad()
def segment_features(agent, runner, texts, qdef, layers, batch_size: int = 64):
    from .inputs import SEG_MASK, SEG_QUESTION, SEG_STATE, collate, encode, to_device

    encs = [encode(agent, t, qdef) for t in texts]
    segs = {"state": SEG_STATE, "question": SEG_QUESTION, "masks": SEG_MASK}
    feats = {s: np.zeros((len(encs), len(layers), runner.d), dtype=np.float32) for s in segs}
    order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
    sites = [f"L{l}.resid_pre" for l in layers]
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        b = to_device(collate([encs[i] for i in idx], agent.tok.pad_token_id), runner.device)
        _, _, cache = runner.forward(b, cache=sites, amp=True)
        for s, v in segs.items():
            m = (b["segment"] == v).float()
            den = m.sum(1, keepdim=True).clamp(min=1)
            for li, site in enumerate(sites):
                feats[s][idx, li] = ((cache[site].float() * m[..., None]).sum(1) / den).cpu().numpy()
    return feats


def steering_sweep(agent, runner, log=print) -> dict:
    from . import competition as cp
    from . import data
    from . import safety as sf
    from .hooks import slot_logits
    from .inputs import SEG_MASK, SEG_QUESTION, SEG_STATE

    nat = data.load_jsonl("jb_natural.jsonl")
    tr = [r for r in nat if r["split"] == "main"][:1500]
    te = [r for r in nat if r["split"] == "heldout"]
    ytr = np.array([r["label"] for r in tr])
    benign = [r["text"] for r in te if r["label"] == 0][:200]
    jbs = [r["text"] for r in te if r["label"] == 1][:200]
    mc = [it for it in mc_items() if it["split"] == "heldout"][:300]
    mc_encs = [cp.encode_mc(agent, it, "generic") for it in mc]
    n = runner.n_enc
    layers = sorted({8, 12, 16, 20, 24, n - 1, n, n + 1})
    feats = segment_features(agent, runner, [r["text"] for r in tr], sf.JAILBREAK_Q, layers)
    segcode = {"state": SEG_STATE, "question": SEG_QUESTION, "masks": SEG_MASK}
    base_b = sf.score_texts(agent, runner, benign)
    base_j = sf.score_texts(agent, runner, jbs)
    base_mc = float(np.mean([np.argmax(r) == it["gold"] for r, it in
                             zip(slot_logits(runner, mc_encs, amp=True), mc)]))
    out = {"base": {"benign_flagged": float((base_b > 0).mean()), "jb_flagged": float((base_j > 0).mean()),
                    "mc_acc": base_mc}, "grid": []}
    log(f"[steer] base {out['base']}")
    for li, layer in enumerate(layers):
        for seg in ("state", "question", "masks"):
            X = feats[seg][:, li]
            d = X[ytr == 1].mean(0) - X[ytr == 0].mean(0)
            gap = float(np.linalg.norm(d))
            d = d / np.linalg.norm(d)
            rn = float(np.linalg.norm(X, axis=1).mean())

            def edit(alpha, sign=1.0):
                v = torch.tensor(d, dtype=torch.float32)

                def f(x, batch):
                    m = (batch["segment"] == segcode[seg])[..., None]
                    return x + m * (sign * alpha * gap) * v.to(x.device, x.dtype)
                return {f"L{layer}.resid_pre": f}

            for alpha in (0.5, 1.0, 2.0):
                lb_ = sf.score_texts(agent, runner, benign, edits=edit(alpha))
                lj_ = sf.score_texts(agent, runner, jbs, edits=edit(alpha, -1.0))
                lu = sf.score_texts(agent, runner, benign, sf.URGENCY_Q, edits=edit(alpha))
                rows = slot_logits(runner, mc_encs, amp=True, edits=edit(alpha))
                row = {"layer": layer, "segment": seg, "alpha_gap_units": alpha, "gap_over_resid_norm": gap / rn,
                       "benign_flagged": float((lb_ > 0).mean()), "jb_flagged_after_neg": float((lj_ > 0).mean()),
                       "urgency_flagged": float((lu > 0).mean()),
                       "mc_acc": float(np.mean([np.argmax(r) == it["gold"] for r, it in zip(rows, mc)]))}
                out["grid"].append(row)
                log(f"[steer] L{layer} {seg} a={alpha}: benign->{row['benign_flagged']:.2f} "
                    f"jb(neg)->{row['jb_flagged_after_neg']:.2f} urg={row['urgency_flagged']:.2f} mc={row['mc_acc']:.2f}")
    good = [r for r in out["grid"] if r["mc_acc"] >= base_mc - 0.03 and r["urgency_flagged"] <= 0.05]
    out["best_specific"] = max(good, key=lambda r: r["benign_flagged"]) if good else None
    out["best_specific_suppress"] = min(good, key=lambda r: r["jb_flagged_after_neg"]) if good else None
    log(f"[steer] best specific add: {out['best_specific']}")
    log(f"[steer] best specific suppress: {out['best_specific_suppress']}")
    return out


def run_all(agent, runner, log=print) -> dict:
    t0 = time.time()
    out = {}
    out["steering_sweep"] = steering_sweep(agent, runner, log)
    save("followups/steering", out["steering_sweep"])
    out["greedy_sets"] = greedy_sets(agent, runner, log)
    save("followups/greedy_sets", out["greedy_sets"])
    out["heldout_track3"] = heldout_track3(agent, runner, log)
    save("followups/heldout_track3", out["heldout_track3"])
    out["fp32"] = fp32_confirm(agent, runner, log)
    save("followups/fp32", out["fp32"])
    log(f"[followups] done in {time.time() - t0:.0f}s")
    return {"wall_s": round(time.time() - t0, 1)}


# ----- LLM-dependent steps (run once the OpenAI key works) -----


def llm_steps(agent, runner, log=print) -> dict:
    from . import data
    from . import labels as lb
    from . import llm
    from . import safety as sf

    models = llm.pick_models()
    log(f"[llm] bulk={models['bulk']} careful={models['careful']}")
    out: dict = {"models": {k: models[k] for k in ("bulk", "careful")}}

    # C4 natural counterfactuals: 200 held-out real jailbreaks that laya flags
    nat = data.load_jsonl("jb_natural.jsonl")
    jbs = [r["text"] for r in nat if r["split"] == "heldout" and r["label"] == 1
           and len(agent.tok(r["text"], add_special_tokens=False)["input_ids"]) <= 300]
    lo = sf.score_texts(agent, runner, jbs)
    flagged = [t for t, v in zip(jbs, lo) if v > 0][:200]
    cfs = llm.counterfactuals(flagged, models["careful"])
    data.save_jsonl("jb_counterfactuals.jsonl", cfs)
    lo_o = sf.score_texts(agent, runner, [c["original"] for c in cfs])
    lo_e = sf.score_texts(agent, runner, [c["edited"] for c in cfs])
    out["counterfactuals"] = {"n": len(cfs), "orig_flagged": float((lo_o > 0).mean()),
                              "edited_flagged": float((lo_e > 0).mean()),
                              "mean_logodds_drop": float((lo_o - lo_e).mean())}
    log(f"[llm] counterfactuals {out['counterfactuals']}")
    # does the natural-set probe direction separate original from edited?
    tr = [r for r in nat if r["split"] == "main"][:1500]
    S_tr, _ = sf.layer_features(agent, runner, [r["text"] for r in tr])
    ytr = [r["label"] for r in tr]
    S_c, M_c = sf.layer_features(agent, runner, [c["original"] for c in cfs] + [c["edited"] for c in cfs])
    lab = [1] * len(cfs) + [0] * len(cfs)
    out["counterfactual_probe_state"] = sf.probe_curve(S_tr, ytr, S_c, lab, seeds=(0,), control=False)
    log(f"[llm] cf probe acc={[round(r['acc'], 2) for r in out['counterfactual_probe_state']]}")

    # C3 replication on LLM-generated items
    items = llm.c3_items(1000, models["bulk"], models["careful"])
    data.save_jsonl("c3_llm.jsonl", items)
    main = [it for it in items if it["split"] == "main"]
    beh = lb.behaviour(agent, runner, main)
    out["c3_llm_behaviour"] = public({k: v for k, v in beh.items()})
    for k, v in beh.items():
        log(f"[llm c3] {k}: n/s={v['acc_normal']:.3f}/{v['acc_swapped']:.3f} follow={v['label_following_rate']:.3f}")
    ko = lb.knockouts(agent, runner, main, "semantic_tf")
    out["c3_llm_knockout"] = {"base": ko["base"], "mask_to_labels@local": ko["mask_to_labels@local"],
                              "mask_to_labels@global": ko["mask_to_labels@global"]}
    log(f"[llm c3] knockout {out['c3_llm_knockout']}")
    out["llm_usage"] = llm.usage_total()
    log(f"[llm] usage {out['llm_usage']}")
    save("followups/llm_steps", out)
    return out


def fp32_only(agent, runner, log=print) -> dict:
    out = fp32_confirm(agent, runner, log)
    save("followups/fp32", out)
    return out
