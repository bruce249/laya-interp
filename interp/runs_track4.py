"""Track 4.1 driver: jailbreak / injection baselines, probes, segment patching, components, steering."""

from __future__ import annotations

import json
import time

import numpy as np

from . import metrics as mt
from .runs import RESULTS, mc_items, save


def _load_json(rel: str):
    path = RESULTS / rel
    return json.loads(path.read_text()) if path.exists() else None


def track4(agent, runner, model: str = "laya", log=print) -> dict:
    from . import ablation as ab
    from . import confidence as cf
    from . import data
    from . import safety as sf
    from .competition import encode_mc
    from .hooks import slot_logits
    from .inputs import encode

    t0 = time.time()
    out: dict = {"model": model}
    arrays: dict = {}
    if not (data.DATA / "jb_pairs.jsonl").exists():
        data.save_jsonl("jb_pairs.jsonl", sf.minimal_pairs(agent.tok))
        data.save_jsonl("jb_natural.jsonl", sf.natural_set())
        data.save_jsonl("injection.jsonl", sf.injection_set())
    pairs = data.load_jsonl("jb_pairs.jsonl")
    nat = data.load_jsonl("jb_natural.jsonl")
    inj = data.load_jsonl("injection.jsonl")
    log(f"[{model}] data: pairs={len(pairs)} natural={len(nat)} injection={len(inj)} ({time.time() - t0:.0f}s)")

    # ----- baselines -----
    lo_nat = sf.score_texts(agent, runner, [r["text"] for r in nat])
    y_nat = np.array([r["label"] for r in nat])
    correct = (lo_nat > 0) == y_nat
    out["natural_baseline"] = {
        "auroc": mt.auroc(lo_nat, y_nat), "accuracy": float(correct.mean()),
        "by_source": {s: float(correct[[r["source"] == s for r in nat]].mean())
                      for s in sorted({r["source"] for r in nat})}}
    lo_inj = sf.score_texts(agent, runner, [r["text"] for r in inj], sf.INJECTION_Q)
    y_inj = np.array([r["label"] for r in inj])
    out["injection_baseline"] = {"auroc": mt.auroc(lo_inj, y_inj), "accuracy": float(((lo_inj > 0) == y_inj).mean())}
    lj = sf.score_texts(agent, runner, [p["jailbreak"] for p in pairs])
    lc = sf.score_texts(agent, runner, [p["control"] for p in pairs])
    keep = [(p, a, b) for p, a, b in zip(pairs, lj, lc) if (a - b) >= 2.0 or (a > 0 > b)]
    kept = [p for p, _, _ in keep]
    out["pairs"] = {"candidates": len(pairs), "kept": len(kept),
                    "kept_by_family": {f: sum(p["family"] == f for p in kept) for f in range(len(sf.FAMILIES))},
                    "candidates_by_family": {f: sum(p["family"] == f for p in pairs) for f in range(len(sf.FAMILIES))},
                    "mean_logodds_gap_kept": float(np.mean([a - b for _, a, b in keep])) if keep else None,
                    "mean_logodds_gap_all": float(np.mean(lj - lc))}
    data.save_jsonl("jb_pairs_kept.jsonl", kept)
    log(f"[{model}] baselines: natural auroc={out['natural_baseline']['auroc']:.3f} "
        f"acc={out['natural_baseline']['accuracy']:.3f} inj auroc={out['injection_baseline']['auroc']:.3f} "
        f"pairs kept={len(kept)}/{len(pairs)} by family={out['pairs']['kept_by_family']} ({time.time() - t0:.0f}s)")

    # ----- probes -----
    tr = [r for r in nat if r["split"] == "main"]
    te = [r for r in nat if r["split"] == "heldout"]
    S_tr, M_tr = sf.layer_features(agent, runner, [r["text"] for r in tr])
    S_te, M_te = sf.layer_features(agent, runner, [r["text"] for r in te])
    ytr, yte = np.array([r["label"] for r in tr]), np.array([r["label"] for r in te])
    out["probe_natural_state"] = sf.probe_curve(S_tr, ytr, S_te, yte)
    out["probe_natural_mask"] = sf.probe_curve(M_tr, ytr, M_te, yte)
    if kept:
        ptexts = [p["jailbreak"] for p in kept] + [p["control"] for p in kept]
        plabels = np.array([1] * len(kept) + [0] * len(kept))
        S_p, M_p = sf.layer_features(agent, runner, ptexts)
        out["probe_natural_to_pairs_state"] = sf.probe_curve(S_tr, ytr, S_p, plabels, seeds=(0,), control=False)
        out["probe_natural_to_pairs_mask"] = sf.probe_curve(M_tr, ytr, M_p, plabels, seeds=(0,), control=False)
        fam = np.array([p["family"] for p in kept] * 2)
        hold = fam == sf.HELDOUT_FAMILY
        if hold.any() and (~hold).any():
            out["probe_pairs_heldout_family_state"] = sf.probe_curve(S_p[~hold], plabels[~hold], S_p[hold],
                                                                     plabels[hold])
            out["probe_pairs_heldout_family_mask"] = sf.probe_curve(M_p[~hold], plabels[~hold], M_p[hold],
                                                                    plabels[hold])
    acc_s = np.array([r["acc"] for r in out["probe_natural_state"]])
    acc_m = np.array([r["acc"] for r in out["probe_natural_mask"]])
    log(f"[{model}] probe acc state={np.round(acc_s, 2).tolist()}")
    log(f"[{model}] probe acc mask={np.round(acc_m, 2).tolist()} ({time.time() - t0:.0f}s)")

    # ----- segment patching -----
    main_pairs = [p for p in kept if p["split"] == "main"][:300]
    held_pairs = [p for p in kept if p["split"] == "heldout"][:100]
    if main_pairs:
        out["patching_main"] = sf.segment_patching(agent, runner, main_pairs)
        rec = out["patching_main"]["recovery_mean"]
        for key in ("state", "diff_tokens", "masks", "question"):
            log(f"[{model}] patch {key}={[round(v, 2) for v in rec[key]]}")
        rec_s, rec_m = np.array(rec["state"]), np.array(rec["masks"])
        out["state_patch_last_layer_ge_0.5"] = int(np.where(rec_s >= 0.5)[0].max()) if (rec_s >= 0.5).any() else None
        out["mask_patch_first_layer_ge_0.5"] = int(np.argmax(rec_m >= 0.5)) if (rec_m >= 0.5).any() else None
    if held_pairs:
        out["patching_heldout"] = sf.segment_patching(agent, runner, held_pairs)
    out["decodable_layer_state"] = int(np.argmax(acc_s >= 0.95 * acc_s.max()))
    out["decodable_layer_mask"] = int(np.argmax(acc_m >= 0.95 * acc_m.max()))
    log(f"[{model}] decodable state@{out['decodable_layer_state']} mask@{out['decodable_layer_mask']} | "
        f"state patch >=0.5 until L{out.get('state_patch_last_layer_ge_0.5')} "
        f"mask patch >=0.5 from L{out.get('mask_patch_first_layer_ge_0.5')}")

    # ----- Track 1/2 components on jailbreak -----
    t1b, t2 = _load_json(f"track1b/{model}.json"), _load_json(f"track2/{model}.json")
    te_encs = [encode(agent, r["text"], sf.JAILBREAK_Q) for r in te]
    means = ab.segment_means(runner, [encode(agent, r["text"], sf.JAILBREAK_Q) for r in tr[:600]])
    base_rows = slot_logits(runner, te_encs, amp=True)
    b0 = sf.log_odds(base_rows)
    comp_res = {}

    def ablate(name, edits=None, **kw):
        rows = slot_logits(runner, te_encs, amp=True, edits=edits, **kw)
        lo = sf.log_odds(rows)
        comp_res[name] = {"auroc": mt.auroc(lo, yte), "accuracy": float(((lo > 0) == yte).mean()),
                          "decision_flips": float(((lo > 0) != (b0 > 0)).mean()),
                          "mean_abs_logodds_ratio": float(np.mean(np.abs(lo)) / np.mean(np.abs(b0)))}
        log(f"[{model}] jailbreak ablation {name}: {comp_res[name]}")

    ablate("base")
    if t1b:
        comp = {int(k): v for k, v in t1b["top_competition_heads"].items()}
        pos = {int(k): v for k, v in t1b["top_position_heads"].items()}
        ablate("comp20_cross_block", cross_option_heads=comp)
        ablate("comp20_mean", edits=ab.mean_heads(means, comp))
        ablate("pos20_mean", edits=ab.mean_heads(means, pos))
        for seed in range(3):
            from .circuits import random_heads
            ablate(f"rand20_s{seed}_mean", edits=ab.mean_heads(means, random_heads(runner, 20, seed)))
    if t2:
        for key in ("A10", "B10"):
            ablate(f"{key}_mean", edits=cf.set_edits(t2["sets"][key], means))
    out["component_ablation_jailbreak"] = comp_res

    # ----- diff-of-means steering, specificity, ablation -----
    best = int(np.argmax(acc_s[: runner.n_layers]))
    d, sigma = sf.diff_means_direction(S_tr, M_tr, ytr, best, "state")
    benign_te = [r["text"] for r in te if r["label"] == 0][:300]
    jb_te = [r["text"] for r in te if r["label"] == 1][:300]
    mc = [it for it in mc_items() if it["split"] == "heldout"][:400]
    mc_encs = [encode_mc(agent, it, "generic") for it in mc]
    steer = {"layer": best, "sigma": sigma, "benign_jailbreak": {}, "benign_urgency": {}, "mc_accuracy": {}}
    for a in (0, 1, 2, 4, 8):
        e = sf.steer_edit(d, sigma, a, best, "state")
        lo = sf.score_texts(agent, runner, benign_te, edits=e)
        steer["benign_jailbreak"][a] = {"mean_logodds": float(lo.mean()), "frac_true": float((lo > 0).mean())}
        lu = sf.score_texts(agent, runner, benign_te, sf.URGENCY_Q, edits=e)
        steer["benign_urgency"][a] = {"mean_logodds": float(lu.mean()), "frac_true": float((lu > 0).mean())}
        rows = slot_logits(runner, mc_encs, amp=True, edits=e)
        steer["mc_accuracy"][a] = float(np.mean([np.argmax(r) == it["gold"] for r, it in zip(rows, mc)]))
    lo0 = sf.score_texts(agent, runner, jb_te)
    lo_abl = sf.score_texts(agent, runner, jb_te, edits=sf.ablate_direction_edit(d, best))
    steer["ablate_on_jailbreaks"] = {"before_frac_true": float((lo0 > 0).mean()),
                                     "after_frac_true": float((lo_abl > 0).mean()),
                                     "mean_logodds_before": float(lo0.mean()),
                                     "mean_logodds_after": float(lo_abl.mean())}
    out["steering"] = steer
    log(f"[{model}] steering L{best}: jb={steer['benign_jailbreak']}")
    log(f"[{model}] steering urgency={steer['benign_urgency']} mc={steer['mc_accuracy']} "
        f"ablate={steer['ablate_on_jailbreaks']}")

    # ----- injection -----
    itr = [r for r in inj if r["split"] == "main"]
    ite = [r for r in inj if r["split"] == "heldout"]
    IS_tr, IM_tr = sf.layer_features(agent, runner, [r["text"] for r in itr], sf.INJECTION_Q)
    IS_te, IM_te = sf.layer_features(agent, runner, [r["text"] for r in ite], sf.INJECTION_Q)
    yitr, yite = np.array([r["label"] for r in itr]), np.array([r["label"] for r in ite])
    out["probe_injection_state"] = sf.probe_curve(IS_tr, yitr, IS_te, yite, seeds=(0,))
    out["probe_injection_mask"] = sf.probe_curve(IM_tr, yitr, IM_te, yite, seeds=(0,))
    di, _ = sf.diff_means_direction(IS_tr, IM_tr, yitr, best, "state")
    out["cos_jailbreak_vs_injection_direction"] = float(d @ di)
    log(f"[{model}] injection probe state={[round(r['acc'], 2) for r in out['probe_injection_state']]} "
        f"cos(jb, inj)={out['cos_jailbreak_vs_injection_direction']:.3f}")
    out["wall_s"] = round(time.time() - t0, 1)
    arrays["direction"] = d
    save(f"track4/{model}", out, arrays)
    return out
