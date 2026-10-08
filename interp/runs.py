"""Experiment drivers. Each writes a JSON summary and an npz of per-item arrays under /store/results."""

from __future__ import annotations

import json
import pathlib
import pickle
import time

import numpy as np

RESULTS = pathlib.Path("/store/results")


def public(d):
    """Drop private keys (leading underscore) recursively for the JSON summary."""
    if isinstance(d, dict):
        return {k: public(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, list):
        return [public(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def save(rel: str, summary: dict, arrays: dict | None = None) -> None:
    path = RESULTS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(public(summary), indent=2, default=str))
    if arrays:
        with path.with_suffix(".pkl").open("wb") as f:
            pickle.dump(arrays, f)


def mc_items(rebuild: bool = False):
    from . import data

    path = data.DATA / "mc.jsonl"
    if rebuild or not path.exists():
        data.save_jsonl("mc.jsonl", data.load_mc())
    return data.load_jsonl("mc.jsonl")


def choose_casting(agent, runner, items, n: int = 300) -> dict:
    from .competition import baseline

    probe = [it for it in items if it["split"] == "main"][:n]
    res = {c: baseline(agent, runner, probe, c, amp=True)["accuracy"]["est"]
           for c in ("generic", "question_both", "question_only")}
    return {"probe_accuracy": res, "chosen": max(res, key=res.get)}


def track1(agent, runner, model: str, casting: str | None = None, conditions: bool = True,
           log=print) -> dict:
    from . import competition as cp

    t0 = time.time()
    items = mc_items()
    out: dict = {"model": model, "n_items": len(items)}
    if casting is None:
        out["casting_probe"] = choose_casting(agent, runner, items)
        casting = out["casting_probe"]["chosen"]
    out["casting"] = casting
    log(f"[{model}] casting={casting} ({time.time() - t0:.0f}s)")

    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    for split, subset in (("main", main), ("heldout", held)):
        b = cp.baseline(agent, runner, subset, casting, amp=True)
        out[f"baseline_{split}"] = b
        log(f"[{model}] baseline {split}: acc={b['accuracy']['est']:.3f} "
            f"flip={b['reversal_flip_rate']['est']:.3f} ece_raw={b['ece_raw']:.3f} ece_t={b['ece_tempered']:.3f}")
    # fp32 confirmation of the headline baseline
    b32 = cp.baseline(agent, runner, main, casting, amp=False)
    out["baseline_main_fp32"] = {k: b32[k] for k in ("accuracy", "reversal_flip_rate", "ece_raw", "ece_tempered")}

    om = {s: cp.order_map(agent, runner, sub, casting, amp=True) for s, sub in (("main", main), ("heldout", held))}
    out["order_map"] = om
    log(f"[{model}] order map: rev_flip={om['main']['reversal_flip_rate']['est']:.3f} "
        f"any_flip={om['main']['any_order_flip_rate']['est']:.3f} "
        f"slot_R2={om['main']['order_variance_explained_by_slot']:.3f} "
        f"neigh_corr={om['main']['neighbour_strength_vs_residual_corr']:.3f}")

    arrays = {}
    inter = {}
    for split, sub in (("main", main), ("heldout", held)):
        r = cp.interaction(agent, runner, sub, casting, amp=True)
        inter[split] = r
        arrays[f"M_{split}"] = r["_M"]
        arrays[f"Mc_{split}"] = r["_Mc"]
        arrays[f"R_{split}"] = r["_R"]
        arrays[f"base_{split}"] = r["_base"]
    sub = main[:400]
    inter["main_filler_dash"] = cp.interaction(agent, runner, sub, casting, filler_text=" -", with_removal=False,
                                               amp=True)
    inter["main_reversed"] = cp.interaction(agent, runner, sub, casting, with_removal=False,
                                            order_kind="reversed", amp=True)
    out["interaction"] = inter
    m = inter["main"]
    log(f"[{model}] interaction: |M|raw={m['mean_abs_M_raw']['est']:.3f} |M|c={m['mean_abs_M_centered']['est']:.3f} "
        f"sym={m['symmetry_corr_centered']:.3f} gold_margin_dfill={m['gold_margin_change_when_top_distractor_filled']['est']:.3f} "
        f"sim_corr={m.get('similarity_vs_abs_M_corr', float('nan')):.3f}")

    if conditions:
        conds = {}
        for name, kw in cp.parallel_conditions(runner).items():
            kw = dict(kw)
            b = cp.baseline(agent, runner, main, casting, amp=True, **kw)
            r = cp.interaction(agent, runner, main[:600], casting, with_removal=False, amp=True, **kw)
            conds[name] = {
                "accuracy": b["accuracy"], "reversal_flip_rate": b["reversal_flip_rate"],
                "ece_raw": b["ece_raw"],
                "mean_abs_M_centered": r["mean_abs_M_centered"], "mean_abs_M_raw": r["mean_abs_M_raw"],
                "gold_margin_change_when_top_distractor_filled": r["gold_margin_change_when_top_distractor_filled"],
            }
            log(f"[{model}] {name}: acc={b['accuracy']['est']:.3f} flip={b['reversal_flip_rate']['est']:.3f} "
                f"|M|c={r['mean_abs_M_centered']['est']:.3f}")
        out["parallel_control"] = conds
    out["wall_s"] = round(time.time() - t0, 1)
    save(f"track1/{model}", out, arrays)
    return out


def track1b(agent, runner, model: str, casting: str = "generic", n_screen: int = 400, log=print) -> dict:
    """1.3 localisation (attention, per-head screens, info flow) and 1.4 causal tests."""
    from . import circuits as cc

    t0 = time.time()
    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    screen = main[:n_screen]
    out: dict = {"model": model, "casting": casting}
    arrays: dict = {}

    arrays["attention"] = cc.attention_summary(agent, runner, main[:300], casting)
    log(f"[{model}] attention summary done ({time.time() - t0:.0f}s)")
    for kind in ("cross", "rope", "mean"):
        arrays[f"screen_{kind}"] = cc.head_screen(agent, runner, screen, casting, kind)
        log(f"[{model}] screen {kind} done ({time.time() - t0:.0f}s)")
    arrays["info_flow"] = cc.info_flow(agent, runner, main[:600], casting)
    out["info_flow"] = arrays["info_flow"]["recovery_mean"] | {"n_pairs": arrays["info_flow"]["n_pairs"]}
    log(f"[{model}] info flow done, pairs={arrays['info_flow']['n_pairs']} ({time.time() - t0:.0f}s)")

    cross_score = arrays["screen_cross"]["l1_centered"]
    rope_score = -arrays["screen_rope"]["flip_delta"]
    out["top_competition_heads"] = cc.top_heads(cross_score, 20)
    out["top_position_heads"] = cc.top_heads(rope_score, 20)
    out["base_flip_screen"] = arrays["screen_rope"]["base_flip"]

    means = None
    from . import ablation as ab
    from .competition import encode_mc
    means = ab.segment_means(runner, [encode_mc(agent, it, casting) for it in main[:600]])

    conds: dict = {}

    def run(name, subset=main, **kw):
        conds[name] = cc.evaluate_condition(agent, runner, subset, casting, **kw)
        c = conds[name]
        log(f"[{model}] {name}: acc={c['accuracy']:.3f} flip={c['reversal_flip_rate']:.3f} "
            f"|M|c={c['mean_abs_M_centered']:.3f}")

    run("base")
    all_enc = {l: list(range(runner.H)) for l in range(runner.n_enc)}
    run("rope_alt_all_enc_heads", rope_alt_heads=all_enc)
    all_cross = {l: list(range(runner.n_heads(l))) for l in range(runner.n_layers)}
    run("cross_block_all_heads", cross_option_heads=all_cross)
    for K in (5, 10, 20):
        comp = cc.top_heads(cross_score, K)
        posh = cc.top_heads(rope_score, K)
        run(f"comp{K}_cross_block", cross_option_heads=comp)
        run(f"comp{K}_mean_ablate", edits=ab.mean_heads(means, comp))
        run(f"comp{K}_resample", edits=ab.resample_heads(comp))
        run(f"pos{K}_rope_alt", rope_alt_heads=posh)
        run(f"pos{K}_mean_ablate", edits=ab.mean_heads(means, posh))
        for seed in range(3):
            rnd = cc.random_heads(runner, K, seed)
            run(f"rand{K}_s{seed}_cross_block", cross_option_heads=rnd)
            rnd_enc = cc.random_heads(runner, K, seed, layers=range(runner.n_enc))
            run(f"rand{K}_s{seed}_rope_alt", rope_alt_heads=rnd_enc)
            run(f"rand{K}_s{seed}_mean_ablate", edits=ab.mean_heads(means, rnd))
    # held-out replication of the K=10 sets
    comp, posh = cc.top_heads(cross_score, 10), cc.top_heads(rope_score, 10)
    run("heldout_base", subset=held)
    run("heldout_comp10_cross_block", subset=held, cross_option_heads=comp)
    run("heldout_pos10_rope_alt", subset=held, rope_alt_heads=posh)
    run("heldout_comp10_mean_ablate", subset=held, edits=ab.mean_heads(means, comp))
    run("heldout_pos10_mean_ablate", subset=held, edits=ab.mean_heads(means, posh))
    out["conditions"] = conds
    out["wall_s"] = round(time.time() - t0, 1)
    save(f"track1b/{model}", out, arrays)
    return out


def track2(agent, runner, model: str, casting: str = "generic", log=print) -> dict:
    """2.1 lenses, 2.2 component screen, 2.3 separability (sets and direction), 2.4 AUROC/ECE, 2.5 overconfidence."""
    import pickle

    import torch

    from . import ablation as ab
    from . import circuits as cc
    from . import confidence as cf
    from .competition import encode_mc
    from .hooks import slot_logits
    from .inputs import collate, to_device

    t0 = time.time()
    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    out: dict = {"model": model, "casting": casting}
    arrays: dict = {}

    out["lenses"] = cf.lenses(agent, runner, main[:800], held, casting)
    log(f"[{model}] lenses: direct dir-stable@{out['lenses']['direct_dir_stable_layer']} "
        f"norm90@{out['lenses']['direct_norm_90_layer']} | tuned dir-stable@{out['lenses']['tuned_dir_stable_layer']} "
        f"norm90@{out['lenses']['tuned_norm_90_layer']} ({time.time() - t0:.0f}s)")

    means = ab.segment_means(runner, [encode_mc(agent, it, casting) for it in main[:600]])
    pkl = RESULTS / f"track1b/{model}.pkl"
    if pkl.exists():
        head_scr = pickle.load(pkl.open("rb"))["screen_mean"]
    else:
        head_scr = cc.head_screen(agent, runner, main[:400], casting, "mean")
    mlp_scr = cf.mlp_screen(agent, runner, main[:400], casting, means)
    arrays["head_screen"], arrays["mlp_screen"] = head_scr, mlp_scr
    comps = cf.components(head_scr, mlp_scr, runner)
    arrays["components"] = comps
    out["component_map_corr_rot_vs_absscale"] = float(np.corrcoef([c["rotation_deg"] for c in comps],
                                                                  [c["abs_log_scale"] for c in comps])[0, 1])
    log(f"[{model}] screens done ({time.time() - t0:.0f}s)")

    def evaluate(subset, edits, tag):
        encs = [encode_mc(agent, it, casting) for it in subset]
        base = slot_logits(runner, encs, amp=True)
        rows = slot_logits(runner, encs, amp=True, edits=edits)
        gold = [it["gold"] for it in subset]
        temps = [e.temperature for e in encs]
        res = cf.summarize_effect(base, rows, gold, temps)
        res["base_calibration"] = cf.calibration_block(base, gold, temps)
        log(f"[{model}] {tag}: rot={res['rotation_deg']['est']:.1f} |ls|={res['abs_log_scale']['est']:.3f} "
            f"ls={res['log_scale']['est']:+.3f} argmax_kept={res['argmax_unchanged']:.3f} "
            f"acc={res['accuracy']:.3f} auroc={res['auroc_margin']:.3f} ece={res['ece_raw']:.3f}")
        return res

    sets = {}
    for K in (5, 10):
        A, B = cf.select_sets(comps, K)
        sets[f"A{K}"], sets[f"B{K}"] = A, B
        for split, subset in (("main", main), ("heldout", held)):
            out[f"A{K}_{split}"] = evaluate(subset, cf.set_edits(A, means), f"A{K} {split}")
            out[f"B{K}_{split}"] = evaluate(subset, cf.set_edits(B, means), f"B{K} {split}")
        for seed in range(3):
            rng = np.random.default_rng(seed)
            rnd = [comps[i] for i in rng.choice(len(comps), K, replace=False)]
            out[f"rand{K}_s{seed}_main"] = evaluate(main, cf.set_edits(rnd, means), f"rand{K} s{seed}")
    out["sets"] = sets

    out["norm_direction"] = {}
    n = runner.n_enc
    for layer in sorted({n - 10, n - 4, n - 1, n, n + 1}):
        r = cf.norm_direction(agent, runner, main[:800], held, casting, layer)
        out["norm_direction"][layer] = r
        d = r["dose"]
        log(f"[{model}] norm direction L{layer}: train r={r['train_corr_log_norm']:.2f} "
            f"ls(+0.5)={d[0.5]['log_scale']:+.3f} ls(-0.5)={d[-0.5]['log_scale']:+.3f} "
            f"rand ls(+0.5)={r['random_dose'][0.5]['log_scale']:+.3f}")

    encs = [encode_mc(agent, it, casting) for it in main]
    rows = slot_logits(runner, encs, amp=False)
    mh_norms = []
    with torch.no_grad():
        for start in range(0, len(encs), 64):
            b = to_device(collate(encs[start:start + 64], agent.tok.pad_token_id), runner.device)
            _, _, c = runner.forward(b, cache=["marker_h"])
            nm = c["marker_h"].float().norm(dim=-1)
            for r_, e in enumerate(encs[start:start + 64]):
                mh_norms.append(float(nm[r_, : e.k].mean()))
    out["overconfidence"] = cf.overconfidence(rows, [it["gold"] for it in main], main, np.array(mh_norms))
    oc = out["overconfidence"]
    log(f"[{model}] overconfidence: global T={oc['global_T']:.2f} "
        f"T by norm quintile={[round(q['T'], 2) for q in oc['by_norm_quintile']]} "
        f"by source={ {k: round(v, 2) for k, v in oc['by_source'].items()} }")
    out["wall_s"] = round(time.time() - t0, 1)
    save(f"track2/{model}", out, arrays)
    return out


def track3(agent, runner, model: str = "laya", log=print) -> dict:
    """C3 label-following on BoolQ: behaviour, strength, distance, patching, edge knockouts."""
    from . import data
    from . import labels as lb

    t0 = time.time()
    if not (data.DATA / "boolq.jsonl").exists():
        data.save_jsonl("boolq.jsonl", lb.load_boolq(agent.tok))
    items = data.load_jsonl("boolq.jsonl")
    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    out: dict = {"model": model, "n_main": len(main), "n_heldout": len(held)}
    for split, sub in (("main", main), ("heldout", held)):
        beh = lb.behaviour(agent, runner, sub)
        out[f"behaviour_{split}"] = beh
        for k, v in beh.items():
            log(f"[{model}] {split} {k}: acc n/s={v['acc_normal']:.3f}/{v['acc_swapped']:.3f} "
                f"follow={v['label_following_rate']:.3f} shift={v['mean_logodds_shift_toward_label']:+.2f}")
    gold = np.array([it["gold"] for it in main])
    out["strength_semantic_tf"] = lb.by_strength(out["behaviour_main"]["semantic_tf/slots_default"], gold)
    out["strength_symbolic_ab"] = lb.by_strength(out["behaviour_main"]["symbolic_ab/slots_default"], gold)
    log(f"[{model}] by strength tf={[q['label_following_rate'] for q in out['strength_semantic_tf']]}")
    out["distance"] = lb.distance_sweep(agent, runner, main[:400])
    log(f"[{model}] distance={ {k: {n: round(v['label_following_rate'] or 0, 3) for n, v in d.items()} for k, d in out['distance'].items()} }")
    for name in ("semantic_tf", "symbolic_ab"):
        pc = lb.patch_curves(agent, runner, main[:300], name)
        out[f"patch_{name}"] = pc
        for kind in ("labels_patch", "state_patch"):
            for s, v in pc[kind].items():
                log(f"[{model}] patch {name} {kind} {s}: {[round(x, 2) for x in v['median']]}")
        ko = lb.knockouts(agent, runner, main, name)
        out[f"knockouts_{name}"] = ko
        for k, v in ko.items():
            log(f"[{model}] knockout {name} {k}: {v}")
    out["wall_s"] = round(time.time() - t0, 1)
    save(f"track3/{model}", out)
    return out


def track43(agent, runner, log=print) -> dict:
    from . import data
    from . import multilingual as ml

    t0 = time.time()
    pairs = data.load_jsonl("jb_pairs_kept.jsonl")
    natural = data.load_jsonl("jb_natural.jsonl")
    out = ml.run(agent, runner, pairs, natural, log=log)
    out["wall_s"] = round(time.time() - t0, 1)
    save("track4/multilingual", out)
    return out


def build_report(log=print) -> str:
    import subprocess

    from . import report

    costs = None
    try:
        costs = {"note": "Modal billing summary (this month)",
                 "summary": subprocess.run(["modal", "billing", "summary"], capture_output=True, text=True,
                                           timeout=60).stdout}
    except Exception as exc:
        costs = {"error": str(exc)}
    return report.build(costs)


def track2_norm(agent, runner, model: str = "laya", casting: str = "generic", log=print) -> dict:
    """Re-run only the 2.3 direction test (dose scaled to the residual norm) and merge into track2."""
    from . import confidence as cf

    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    held = [it for it in items if it["split"] == "heldout"]
    js_path = RESULTS / f"track2/{model}.json"
    out = json.loads(js_path.read_text())
    out["norm_direction_sigma_units_invalid"] = out.pop("norm_direction", None)
    out["norm_direction"] = {}
    n = runner.n_enc
    for layer in sorted({n - 10, n - 4, n - 1, n, n + 1}):
        r = cf.norm_direction(agent, runner, main[:800], held, casting, layer)
        out["norm_direction"][layer] = r
        d = r["dose"]
        log(f"[{model}] norm direction L{layer}: train r={r['train_corr_log_norm']:.2f} "
            f"ls(+0.5)={d[0.5]['log_scale']:+.3f} ls(-0.5)={d[-0.5]['log_scale']:+.3f} "
            f"ls(+0.25)={d[0.25]['log_scale']:+.3f} rot(+0.5)={d[0.5]['rotation_deg']:.1f} "
            f"kept(+/-0.5)={d[0.5]['argmax_unchanged']:.3f}/{d[-0.5]['argmax_unchanged']:.3f} "
            f"rand ls(+0.5)={r['random_dose'][0.5]['log_scale']:+.3f} rot={r['random_dose'][0.5]['rotation_deg']:.1f}")
    js_path.write_text(json.dumps(public(out), indent=2, default=str))
    return out
