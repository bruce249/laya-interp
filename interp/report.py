"""Build the HTML report from /store/results. Figures are matplotlib PNGs embedded as data URIs."""

from __future__ import annotations

import base64
import html
import io
import json
import pathlib
import pickle

import numpy as np

RESULTS = pathlib.Path("/store/results")
REPORT = pathlib.Path("/store/report")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e7e6e2"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0,
    })
    return plt


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load(rel):
    p = RESULTS / rel
    if p.with_suffix(".json").exists():
        js = json.loads(p.with_suffix(".json").read_text())
    else:
        js = None
    pk = pickle.load(p.with_suffix(".pkl").open("rb")) if p.with_suffix(".pkl").exists() else None
    return js, pk


def _lines(ax, curves: dict, xlabel="layer", ylabel=""):
    for i, (name, ys) in enumerate(curves.items()):
        ax.plot(range(len(ys)), ys, color=SERIES[i % len(SERIES)], label=name, marker="o", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def _heat(ax, M, title, cmap="Blues", center=False):
    if center:
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M.T, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
    else:
        im = ax.imshow(M.T, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_title(title)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, fraction=0.03)


def _table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, dict) and "est" in v:
            return f"{v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build(costs: dict | None = None) -> str:
    plt = _style()
    sections = []

    # ----- Track 1 -----
    t1 = {m: _load(f"track1/{m}")[0] for m in ("laya", "smolaya")}
    t1 = {m: v for m, v in t1.items() if v}
    if t1:
        rows = []
        for m, d in t1.items():
            for name, c in d.get("parallel_control", {}).items():
                rows.append({"model": m, "condition": name, "accuracy": c["accuracy"]["est"],
                             "reversal flips": c["reversal_flip_rate"]["est"],
                             "|M| centered": c["mean_abs_M_centered"]["est"]})
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        keys = ["sequential", "options_blocked_seqpos", "shipped_parallel_positions", "masked_parallel_all"]
        labels = ["trained\n(sequential)", "options can't\nsee each other", "shared positions\n(shipped parallel)",
                  "fully masked\nparallel"]
        x = np.arange(len(keys))
        for i, (m, d) in enumerate(t1.items()):
            pc = d["parallel_control"]
            axes[0].bar(x + i * 0.38, [pc[k]["reversal_flip_rate"]["est"] for k in keys], 0.36, color=SERIES[i],
                        label=m)
            axes[1].bar(x + i * 0.38, [pc[k]["mean_abs_M_centered"]["est"] for k in keys], 0.36, color=SERIES[i],
                        label=m)
        for ax, t in zip(axes, ("Reversal flip rate", "Mean |M| (centered logits)")):
            ax.set_xticks(x + 0.19, labels, fontsize=8)
            ax.set_title(t)
            ax.legend()
        sections.append(("Track 1 · option competition vs order",
                          f"<img src='{_png(fig)}'>" + _table(rows, ["model", "condition", "accuracy",
                                                                    "reversal flips", "|M| centered"])))
        brows = []
        for m, d in t1.items():
            for split in ("main", "heldout"):
                b = d[f"baseline_{split}"]
                brows.append({"model": m, "split": split, "accuracy": b["accuracy"],
                              "reversal flips": b["reversal_flip_rate"], "ECE raw": b["ece_raw"],
                              "ECE tempered": b["ece_tempered"]})
            om = d["order_map"]["main"]
            brows.append({"model": m, "split": "order map",
                          "accuracy": f"slot R²={om['order_variance_explained_by_slot']:.3f}",
                          "reversal flips": om["any_order_flip_rate"], "ECE raw": "any-order flips →",
                          "ECE tempered": f"neighbour r={om['neighbour_strength_vs_residual_corr']:.3f}"})
        sections.append(("Baselines (MC: ARC, CSQA, QASC)",
                         _table(brows, ["model", "split", "accuracy", "reversal flips", "ECE raw", "ECE tempered"])))

    # ----- Track 1b -----
    js, pk = _load("track1b/laya")
    if js and pk:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        _heat(axes[0], pk["screen_cross"]["l1_centered"], "Cross-option block: L1 change of logits")
        _heat(axes[1], -pk["screen_rope"]["flip_delta"], "Order-blind RoPE: flip reduction", center=True)
        _heat(axes[2], pk["screen_mean"]["rotation_deg"], "Mean ablation: rotation (deg)")
        flow = js["info_flow"]
        fig2, ax = plt.subplots(figsize=(8, 3.6))
        _lines(ax, {k: v for k, v in flow.items() if k != "n_pairs"}, ylabel="recovery of s_i")
        ax.set_title(f"Where option j's influence on option i lives ({flow['n_pairs']} pairs)")
        conds = js["conditions"]
        rows = [{"condition": k, **{c: v[c] for c in ("accuracy", "reversal_flip_rate", "mean_abs_M_centered")}}
                for k, v in conds.items()]
        sections.append(("Track 1.3–1.4 · heads, information flow, ablations",
                         f"<img src='{_png(fig)}'><img src='{_png(fig2)}'>" +
                         _table(rows, ["condition", "accuracy", "reversal_flip_rate", "mean_abs_M_centered"])))

    # ----- Track 2 -----
    js, pk = _load("track2/laya")
    if js:
        lz = js["lenses"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        for ax, kind in zip(axes, ("direct", "tuned")):
            rows = lz[kind]
            ls = sorted(rows, key=lambda k: int(k))
            _lines(ax, {"direction cos": [rows[l]["dir_cos"] for l in ls],
                        "argmax agree": [rows[l]["argmax_agree"] for l in ls],
                        "norm ratio": [min(rows[l]["norm_ratio_median"], 2) for l in ls],
                        "norm corr": [rows[l]["norm_corr_with_final"] for l in ls]})
            ax.set_title(f"{kind} lens: choice (û) vs confidence (‖z‖)")
        figs = f"<img src='{_png(fig)}'>"
        if pk:
            comps = pk["components"]
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for kind, col in (("head", SERIES[0]), ("mlp", SERIES[1])):
                cs = [c for c in comps if c["kind"] == kind]
                ax.scatter([c["rotation_deg"] for c in cs], [c["log_scale"] for c in cs], s=12, color=col,
                           label=kind, alpha=0.8)
            for key, col in (("A10", SERIES[2]), ("B10", SERIES[6])):
                for c in js["sets"][key]:
                    ax.scatter(c["rotation_deg"], c["log_scale"], s=60, facecolors="none", edgecolors=col)
            ax.set_xlabel("rotation of û (deg)")
            ax.set_ylabel("log ‖z′‖/‖z‖")
            ax.set_title("Component map (circles: A10 green, B10 violet)")
            ax.legend()
            figs += f"<img src='{_png(fig)}'>"
        nd = js["norm_direction"]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for i, (layer, r) in enumerate(sorted(nd.items(), key=lambda kv: int(kv[0]))):
            al = sorted(r["dose"], key=float)
            ax.plot([float(a) for a in al], [r["dose"][a]["log_scale"] for a in al], color=SERIES[i % 8],
                    label=f"L{layer}", marker="o", markersize=3)
            ax.plot([float(a) for a in al], [r["random_dose"][a]["log_scale"] for a in al], color=SERIES[i % 8],
                    linestyle=":", linewidth=1)
        ax.set_xlabel("dose (fraction of mean [MASK] residual norm)")
        ax.set_ylabel("log scale change")
        ax.set_title("Norm direction dose response (dotted: random)")
        ax.legend(fontsize=8)
        figs += f"<img src='{_png(fig)}'>"
        rows = []
        for key in sorted(k for k in js if k[:1] in "AB" and k[1:3].rstrip("_").isdigit() or k.startswith("rand")):
            v = js[key]
            if isinstance(v, dict) and "rotation_deg" in v:
                rows.append({"set": key, "rotation°": v["rotation_deg"], "|log scale|": v["abs_log_scale"],
                             "log scale": v["log_scale"], "argmax kept": v["argmax_unchanged"],
                             "accuracy": v["accuracy"], "AUROC margin": v["auroc_margin"], "ECE raw": v["ece_raw"]})
        oc = js["overconfidence"]
        figs += _table(rows, ["set", "rotation°", "|log scale|", "log scale", "argmax kept", "accuracy",
                              "AUROC margin", "ECE raw"])
        figs += f"<p>Overconfidence: global T={oc['global_T']:.2f}; T by ‖z‖ quintile " \
                f"{[round(q['T'], 2) for q in oc['by_norm_quintile']]}; by source " \
                f"{ {k: round(v, 2) for k, v in oc['by_source'].items()} }; ECE after global T " \
                f"{oc['ece_after_global_T']:.3f}.</p>"
        sections.append(("Track 2 · choice vs confidence", figs))

    # ----- Track 3 -----
    js, _ = _load("track3/laya")
    if js:
        beh = js["behaviour_main"]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        names = list(beh)
        x = np.arange(len(names))
        ax.bar(x - 0.2, [beh[n]["acc_normal"] for n in names], 0.38, color=SERIES[0], label="normal labels")
        ax.bar(x + 0.2, [beh[n]["acc_swapped"] for n in names], 0.38, color=SERIES[1], label="swapped labels")
        ax.set_xticks(x, [n.replace("/", "\n") for n in names], fontsize=7)
        ax.set_ylabel("BoolQ accuracy")
        ax.set_title("Label swap: semantic labels vs symbols")
        ax.legend()
        figs = f"<img src='{_png(fig)}'>"
        for name in ("semantic_tf", "symbolic_ab"):
            pc = js.get(f"patch_{name}")
            if pc:
                fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
                _lines(axes[0], {k: v["median"] for k, v in pc["labels_patch"].items()}, ylabel="recovery")
                axes[0].set_title(f"{name}: patch normal-label run → swapped")
                _lines(axes[1], {k: v["median"] for k, v in pc["state_patch"].items()}, ylabel="recovery")
                axes[1].set_title(f"{name}: patch real state → filler state")
                figs += f"<img src='{_png(fig)}'>"
            ko = js.get(f"knockouts_{name}")
            if ko:
                figs += f"<h3>Edge knockouts ({name})</h3>" + _table(
                    [{"knockout": k, **v} for k, v in ko.items()],
                    ["knockout", "acc_normal", "acc_swapped", "label_following_rate", "swap_gap"])
        dist = js["distance"]
        figs += _table([{"pad tokens": p, **{f"{n} follow": v[n]["label_following_rate"] for n in v},
                         **{f"{n} acc": v[n]["acc_normal"] for n in v}} for p, v in dist.items()],
                       ["pad tokens", "semantic_tf follow", "symbolic_ab follow", "semantic_tf acc",
                        "symbolic_ab acc"])
        sections.append(("Track 3 · label-following (BoolQ)", figs))

    # ----- Track 4 -----
    js, _ = _load("track4/laya")
    if js:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        _lines(axes[0], {"jailbreak · state": [r["acc"] for r in js["probe_natural_state"]],
                         "jailbreak · [MASK]": [r["acc"] for r in js["probe_natural_mask"]],
                         "injection · state": [r["acc"] for r in js["probe_injection_state"]],
                         "control (shuffled)": [r["control_acc"] for r in js["probe_natural_state"]]},
               ylabel="probe accuracy (held-out)")
        axes[0].set_title("Where is the attack decodable?")
        if "patching_main" in js:
            _lines(axes[1], js["patching_main"]["recovery_mean"], ylabel="recovery of log-odds")
            axes[1].set_title("Where is it causally decided? (segment patching)")
        figs = f"<img src='{_png(fig)}'>"
        st = js["steering"]
        figs += _table([{"α": a, "benign → jailbreak frac": st["benign_jailbreak"][a]["frac_true"],
                         "urgency frac (specificity)": st["benign_urgency"][a]["frac_true"],
                         "MC accuracy": st["mc_accuracy"][a]} for a in st["benign_jailbreak"]],
                       ["α", "benign → jailbreak frac", "urgency frac (specificity)", "MC accuracy"])
        figs += _table([{"ablation": k, **v} for k, v in js["component_ablation_jailbreak"].items()],
                       ["ablation", "auroc", "accuracy", "decision_flips", "mean_abs_logodds_ratio"])
        figs += f"<p>Baselines: natural AUROC {js['natural_baseline']['auroc']:.3f}, injection AUROC " \
                f"{js['injection_baseline']['auroc']:.3f}; pairs kept {js['pairs']['kept']}/{js['pairs']['candidates']}; " \
                f"cos(jailbreak, injection direction) {js['cos_jailbreak_vs_injection_direction']:.3f}.</p>"
        sections.append(("Track 4 · jailbreak and injection", figs))

    js, _ = _load("track4/multilingual")
    if js:
        st = js["steer_hindi_controls"]
        rows = [{"α (σ)": a, "Hindi controls flagged": v["hi_control_frac_true"], "mean log-odds": v["mean_logodds"]}
                for a, v in st.items()]
        b = js["baseline"]
        sections.append(("Track 4.3 · English direction on Hindi (laya-multilingual)",
                         f"<p>Flagged: English jailbreak {b['en_jb_frac_true']:.2f} vs control {b['en_ctl_frac_true']:.2f}; "
                         f"Hindi jailbreak {b['hi_jb_frac_true']:.2f} vs control {b['hi_ctl_frac_true']:.2f}. "
                         f"English direction at L{js['layer']}: Hindi projection AUROC {js['hindi_projection_auroc']:.3f}; "
                         f"projecting it out of Hindi jailbreaks: {js['ablate_hindi_jailbreaks']['before']:.2f} → "
                         f"{js['ablate_hindi_jailbreaks']['after']:.2f} flagged; random direction at α=8: "
                         f"{js['random_direction_alpha8_frac_true']:.2f}.</p>" +
                         _table(rows, ["α (σ)", "Hindi controls flagged", "mean log-odds"])))

    fu = {k: _load(f"followups/{k}")[0] for k in ("greedy_sets", "steering", "heldout_track3", "fp32", "llm_steps")}
    parts = []
    g = fu["greedy_sets"]
    if g:
        rows = [{"set": n, "components": ", ".join(f"{c['kind']} L{c['layer']}" + (f" h{c['head']}" if "head" in c else "")
                                                   for c in g[f"{n}_set"]),
                 **{f"{split} {m}": (g[f"{n}_{split}"][m]["est"] if isinstance(g[f"{n}_{split}"][m], dict)
                                     else g[f"{n}_{split}"][m])
                    for split in ("main", "heldout") for m in ("abs_log_scale", "argmax_unchanged", "ece_raw")}}
                for n in ("B", "A")]
        parts.append("<h3>Greedy C2 sets</h3>" + _table(rows, list(rows[0].keys())))
    st = fu["steering"]
    if st:
        parts.append(f"<h3>Steering sweep</h3><p>Base: {st['base']}. Best specific add: {st['best_specific']}. "
                     f"Best specific suppress: {st['best_specific_suppress']}.</p>")
    h = fu["heldout_track3"]
    if h:
        parts.append(f"<h3>Held-out Track 3</h3><p>Knockouts: base {h['knockouts_semantic_tf']['base']}; "
                     f"local mask→label {h['knockouts_semantic_tf']['mask_to_labels@local']}.</p>")
    f32 = fu["fp32"]
    if f32:
        parts.append("<h3>fp32 confirmation</h3><pre>" + html.escape(json.dumps(f32, indent=1)[:6000]) + "</pre>")
    ll = fu["llm_steps"]
    if ll:
        beh = ll["c3_llm_behaviour"]
        parts.append(f"<h3>LLM-generated data ({ll['models']})</h3><p>Counterfactuals: {ll['counterfactuals']}.</p>" +
                     _table([{"labels": k, "acc normal": v["acc_normal"], "acc swapped": v["acc_swapped"],
                              "label-following": v["label_following_rate"]} for k, v in beh.items()],
                            ["labels", "acc normal", "acc swapped", "label-following"]) +
                     f"<p>Knockout (local mask→label): {ll['c3_llm_knockout']['mask_to_labels@local']}; "
                     f"base {ll['c3_llm_knockout']['base']}. Usage {ll['llm_usage']}.</p>")
    if parts:
        sections.append(("Follow-ups: fp32, held-out, greedy sets, steering, LLM data", "".join(parts)))

    arms = {m: _load(f"retrain/{m}")[0] for m in ("ctrl_soft_ce", "ctrl_rlcd")}
    arms = {m: v for m, v in arms.items() if v}
    if arms:
        rows = []
        for m, d in arms.items():
            rows.append({"model": m, "accuracy": d["baseline"]["accuracy"], "reversal flips": d["baseline"]["reversal_flip_rate"],
                         "flips, shared positions": d["parallel"]["shipped_parallel_positions"]["reversal_flip_rate"],
                         "flips, options blocked": d["parallel"]["options_blocked_seqpos"]["reversal_flip_rate"],
                         "ECE raw": d["baseline"]["ece_raw"], "|M| centered": d["interaction"]["mean_abs_M_centered"],
                         "greedy-B MLPs": d["greedy_B_mlps"]["layers"],
                         "label-following yes/no": d["label_following"]["semantic_yn/slots_default"]["label_following_rate"]})
        sections.append(("Stretch · same init, same data: soft-CE vs RLCD",
                         _table(rows, list(rows[0].keys()))))
    js, _ = _load("sae/summary")
    if js:
        rows = []
        for key in [k for k in js if k.startswith("L")]:
            v = js[key]
            for abl in ("ablate_conf10", "ablate_choice10", "ablate_random10"):
                rows.append({"layer": key, "ablation": abl, **v[abl]})
        meta = "; ".join(f"{k}: FVE {js[k]['stats']['val_fve']:.3f}, recon keeps {js[k]['stats']['downstream_argmax_kept']:.3f} "
                         f"of answers, top-25 choice∩confidence overlap {js[k]['overlap_top25']}"
                         for k in js if k.startswith("L"))
        labels = ""
        for k in [k for k in js if k.startswith("L")]:
            lab = js[k].get("labels") or {}
            if lab:
                labels += f"<h3>{k} feature labels (gpt-5.6-luna)</h3>" + _table(
                    [{"latent": j, "label": t, "choice top": int(j) in js[k]["top_choice"],
                      "confidence top": int(j) in js[k]["top_confidence"]} for j, t in lab.items()],
                    ["latent", "label", "choice top", "confidence top"])
        sections.append(("SAE · [MASK] features", f"<p>{html.escape(meta)}</p>" +
                         _table(rows, ["layer", "ablation", "abs_log_scale", "log_scale", "rotation_deg", "argmax_kept"]) + labels))

    if costs:
        sections.append(("Costs", "<pre>" + html.escape(json.dumps(costs, indent=2)) + "</pre>"))

    body = "".join(f"<section><h2>{html.escape(t)}</h2>{c}</section>" for t, c in sections)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laya interpretability</title><style>
:root{{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e7e6e2}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:auto}} section{{background:var(--surface);border:1px solid var(--rule);border-radius:8px;
padding:16px;margin:16px 0;overflow-x:auto}} img{{max-width:100%;display:block;margin:8px 0}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0}} td,th{{border-bottom:1px solid var(--rule);padding:4px 8px;
text-align:left;white-space:nowrap}} th{{color:var(--ink2)}}</style></head><body><main>
<h1>How laya chooses: option competition, confidence, label-following</h1>{body}</main></body></html>"""
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "index.html").write_text(page)
    return str(REPORT / "index.html")
