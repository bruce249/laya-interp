"""Phase 3 (plan v4): is the late-MLP set a confidence regulator or a generic logit amplifier?

(a)  mean-ablate the greedy B set (MLPs 22, 26, 27)
(b)  keep the MLPs; rescale each item's centered logits to (a)'s norm (same direction as base)
(b') keep the MLPs; one global temperature fitted to match (a)'s mean scale
(c1) ablate only at [MASK] positions      (c2) ablate everywhere except [MASK] positions
Metrics: ECE, Brier, NLL (raw logits), argmax and full-ranking retention (Kendall tau), AUROC of the margin.
Key test: is the per-item scale ratio ||z_a|| / ||z_base|| constant (amplifier) or input-dependent (regulator)?
"""

from __future__ import annotations

import json

import numpy as np

from . import ablation as ab
from . import competition as cp
from . import confidence as cf
from . import metrics as mt
from .hooks import slot_logits
from .inputs import SEG_MASK


def scorecard(rows, gold) -> dict:
    gold = np.asarray(gold)
    p = [mt.softmax(r) for r in rows]
    correct = np.array([np.argmax(r) == g for r, g in zip(rows, gold)])
    conf = np.array([q.max() for q in p])
    brier = np.mean([np.sum((q - np.eye(len(q))[g]) ** 2) for q, g in zip(p, gold)])
    nll = -np.mean([np.log(max(q[g], 1e-12)) for q, g in zip(p, gold)])
    return {"accuracy": float(correct.mean()), "ece": mt.ece(conf, correct), "brier": float(brier),
            "nll": float(nll), "auroc_margin": mt.auroc(np.array([mt.margin(r) for r in rows]), correct)}


def retention(base, rows) -> dict:
    from scipy.stats import kendalltau

    taus = []
    for b, r in zip(base, rows):
        t = kendalltau(b, r).statistic
        taus.append(1.0 if np.isnan(t) else t)
    return {"argmax_kept": float(np.mean([np.argmax(b) == np.argmax(r) for b, r in zip(base, rows)])),
            "kendall_tau": float(np.mean(taus)),
            "rotation_deg": float(np.mean([mt.rotation(b, r) for b, r in zip(base, rows)]))}


def rescale_to(base, target):
    """Same direction as base, centered norm equal to target's."""
    out = []
    for b, t in zip(base, target):
        zb = mt.centered(b)
        nb, nt = np.linalg.norm(zb), np.linalg.norm(mt.centered(t))
        out.append(np.asarray(b).mean() + zb * (nt / max(nb, 1e-12)))
    return out


def masked_mlp_edit(means, layers, where: str):
    """where='mask': ablate only at [MASK]s; where='nonmask': ablate everywhere except [MASK]s."""
    edits = {}
    for l in layers:
        table = means["mlp"][l]

        def f(x, batch, table=table):
            import torch

            rep = table[batch["segment"]].to(x.dtype)
            at_mask = (batch["segment"] == SEG_MASK)[..., None]
            return torch.where(at_mask, rep, x) if where == "mask" else torch.where(at_mask, x, rep)
        edits[f"L{l}.mlp_out"] = f
    return edits


def greedy_mlps(agent, runner, items, means, log=print, K: int = 6) -> list[dict]:
    """Same rule as laya's search, MLPs only: add the MLP that most increases |log scale| while >= 95% of
    answers are kept; stop when nothing helps."""
    encs = [cp.encode_mc(agent, it, "generic") for it in items]
    base = slot_logits(runner, encs)
    chosen, cur = [], None
    for _ in range(K):
        best, best_s = None, None
        for l in range(runner.n_layers):
            if any(c["layer"] == l for c in chosen):
                continue
            trial = chosen + [{"kind": "mlp", "layer": l}]
            e = cf.per_item_effects(base, slot_logits(runner, encs, edits=cf.set_edits(trial, means)))
            s = {"abs_ls": float(np.abs(e["log_scale"]).mean()), "kept": float(1 - e["argmax_change"].mean())}
            if s["kept"] >= 0.95 and (best_s is None or s["abs_ls"] > best_s["abs_ls"]):
                best, best_s = trial[-1], s
        if best is None or (cur is not None and best_s["abs_ls"] <= cur["abs_ls"] + 1e-3):
            break
        chosen.append(best)
        cur = best_s
        log(f"[P3 greedy] + MLP L{best['layer']} -> |ls|={best_s['abs_ls']:.3f} kept={best_s['kept']:.3f}")
    return chosen


def run(agent, runner, model: str, log=print) -> dict:
    from .factorial import mc_confirm
    from .runs import RESULTS, mc_items

    runner.force_fp32 = True
    items = mc_items()
    main = [it for it in items if it["split"] == "main"]
    sets = {"explore_main": main, "explore_heldout": [it for it in items if it["split"] == "heldout"],
            "confirm": mc_confirm()}
    means = ab.segment_means(runner, [cp.encode_mc(agent, it, "generic") for it in main[:600]])
    if model == "laya":
        greedy = json.loads((RESULTS / "followups/greedy_sets.json").read_text())
        bset = greedy["B_set"]
    else:
        bset = greedy_mlps(agent, runner, main[:400], means, log)
    layers = [c["layer"] for c in bset if c["kind"] == "mlp"]
    out = {"model": model, "precision": "fp32", "mlp_layers": layers}
    for sname, subset in sets.items():
        encs = [cp.encode_mc(agent, it, "generic") for it in subset]
        gold = [it["gold"] for it in subset]
        base = slot_logits(runner, encs)
        a = slot_logits(runner, encs, edits=cf.set_edits(bset, means))
        b = rescale_to(base, a)
        ratios = np.array([np.linalg.norm(mt.centered(x)) / max(np.linalg.norm(mt.centered(y)), 1e-12)
                           for x, y in zip(a, base)])
        T = 1.0 / float(np.exp(np.mean(np.log(ratios))))               # global factor matching (a)'s mean scale
        bglob = [np.asarray(r) / T for r in base]
        c1 = slot_logits(runner, encs, edits=masked_mlp_edit(means, layers, "mask"))
        c2 = slot_logits(runner, encs, edits=masked_mlp_edit(means, layers, "nonmask"))
        Topt = cf.fit_temperature(base, gold)
        best_T = [np.asarray(r) / Topt for r in base]
        conds = {"base": base, "a_ablate": a, "b_rescale_per_item": b, "b_global_temperature": bglob,
                 "best_global_temperature": best_T, "c1_ablate_at_mask_only": c1, "c2_ablate_except_mask": c2}
        res = {k: {**scorecard(v, gold), **retention(base, v)} for k, v in conds.items()}
        correct = np.array([np.argmax(r) == g for r, g in zip(base, gold)])
        margin = np.array([mt.margin(r) for r in base])
        srcs = sorted({it["source"] for it in subset})
        res["scale_ratio"] = {
            "mean": float(ratios.mean()), "sd": float(ratios.std()), "cv": float(ratios.std() / ratios.mean()),
            "corr_with_correct": float(np.corrcoef(ratios, correct.astype(float))[0, 1]),
            "corr_with_base_margin": float(np.corrcoef(ratios, margin)[0, 1]),
            "mean_correct": float(ratios[correct].mean()), "mean_incorrect": float(ratios[~correct].mean()),
            "by_source": {s: float(ratios[[it["source"] == s for it in subset]].mean()) for s in srcs},
            "global_T_matching_a": T, "best_global_T": Topt,
        }
        out[sname] = res
        for k in conds:
            r = res[k]
            log(f"[P3 {sname}] {k}: acc={r['accuracy']:.3f} ece={r['ece']:.3f} brier={r['brier']:.3f} "
                f"nll={r['nll']:.3f} auroc={r['auroc_margin']:.3f} kept={r['argmax_kept']:.3f} "
                f"tau={r['kendall_tau']:.3f} rot={r['rotation_deg']:.1f}")
        sr = res["scale_ratio"]
        log(f"[P3 {sname}] scale ratio mean={sr['mean']:.3f} cv={sr['cv']:.3f} corr(correct)={sr['corr_with_correct']:+.3f} "
            f"corr(margin)={sr['corr_with_base_margin']:+.3f} correct/incorrect={sr['mean_correct']:.3f}/{sr['mean_incorrect']:.3f} "
            f"by source={ {k: round(v, 3) for k, v in sr['by_source'].items()} }")
    runner.force_fp32 = False
    return out
