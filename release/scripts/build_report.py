"""Build the final report (HTML + PDF) from release/results only. Every number is computed here.

Usage (from the repo root):
    python release/scripts/build_report.py            # writes release/report/report.html and .pdf
"""

from __future__ import annotations

import base64
import io
import json
import pathlib
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#008300", "#4a3aa7")
INK2, GRID = "#52514e", "#e7e6e2"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.edgecolor": "#898781", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False})


def J(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f'<figure><img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"></figure>'


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def f3(x):
    return f"{x:.3f}"


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def ci(d, scale=100, digits=1, unit=""):
    return f"{d['est'] * scale:.{digits}f}{unit} [{d['lo'] * scale:.{digits}f}, {d['hi'] * scale:.{digits}f}]"


# ---------------------------------------------------------------- load
V = {m: J(f"phase0/verify_{m}.json") for m in ("laya", "smolaya", "typed")}
V_ML = J("track4/multilingual.json")["verify"]
FP32 = J("followups/fp32.json")
GUARD = J("followups/label_guard.json")
P1 = {m: J(f"phase1/{m}.json") for m in ("laya", "smolaya", "multilingual", "typed", "ctrl_soft_ce", "ctrl_rlcd",
                                         "ctrl_soft_ce_s1", "ctrl_soft_ce_s2", "ctrl_rlcd_s1", "ctrl_rlcd_s2")}
P2 = {m: J(f"phase2/{m}.json") for m in ("laya", "multilingual", "typed")}
P2B = J("phase2/laya_e4b_exploratory.json")
P3 = {m: J(f"phase3/{m}.json") for m in ("laya", "multilingual", "typed")}
P5 = {}
for f in sorted((R / "phase5").glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    name = d["eval"]["model"]
    reg = "baseline" if name.startswith("ctrl_rlcd") else name.rsplit("_s", 1)[0].replace("p5_", "")
    P5.setdefault(reg, []).append(d["eval"])
RT = {m: J(f"retrain/{m}.json") for m in ("ctrl_soft_ce", "ctrl_soft_ce_s1", "ctrl_soft_ce_s2", "ctrl_rlcd",
                                          "ctrl_rlcd_s1", "ctrl_rlcd_s2")}
T1B = J("track1b/laya.json")
T1 = J("track1/laya.json")
T2 = J("track2/laya.json")
T4 = J("track4/laya.json")
T4M = J("track4/multilingual.json")
STEER = J("followups/steering.json")
LLM = J("followups/llm_steps.json")
SAE = J("sae/summary.json")
HELD3 = J("followups/heldout_track3.json")
P8 = {m: J(f"phase8/eval_{m}.json") for m in ("laya", "typed", "multilingual")}
P8P4 = {m: J(f"phase8/p4_{m}.json") for m in ("laya", "typed", "multilingual", "smolaya", "ctrl_soft_ce", "ctrl_rlcd")}
STYLE = J("phase8/style_test.json")
GALL = {n: J(f"phase8/guardall_{n}.json") for n in ("laya", "p5_planted_s0", "p5_planted_s1", "p5_planted_s2",
                                                     "p5_planted_s3", "p5_planted_s4", "p5_planted_s5")}
DOSE_ARMS = {"Baseline: default labels, fixed order": ["ctrl_rlcd", "ctrl_rlcd_s1", "ctrl_rlcd_s2", "p5_baseline_s3",
                                                       "p5_baseline_s4", "p5_baseline_s5"],
             "D0: label random, informative text": [f"p5_dose_d0_s{i}" for i in range(3)],
             "D1: label always right, informative text": [f"p5_dose_d1_s{i}" for i in range(6)],
             "D2: label right 75%, neutral text": [f"p5_dose_d2_s{i}" for i in range(3)],
             "D3: label right 90%, neutral text": [f"p5_dose_d3_s{i}" for i in range(3)],
             "Planted: label always right, neutral text": [f"p5_planted_s{i}" for i in range(6)]}
BENCH = {n: J(f"phase8/bench_{n}.json") for ns in DOSE_ARMS.values() for n in ns}
ROB = {n: J(f"phase8/rob_{n}.json") for n in [f"{t}_{r}_s{i}" for t in ("p8rob", "p9rob")
                                                for r in ("planted", "baseline") for i in range(3)]}
BENCH2 = {n: J(f"phase9/bench2_{n}.json") for n in ["laya", "typed", "multilingual"] +
          [f"p5_planted_s{i}" for i in range(6)]}

# ---------------------------------------------------------------- C3 numbers
beh = FP32["track3_behaviour"]
yn, tf = beh["semantic_yn/slots_default"], beh["semantic_tf/slots_default"]
ab_, ns = beh["symbolic_ab/slots_default"], beh["neutral_sym/slots_default"]
e2x, e2c = P2["laya"]["E2_explore"]["yes/no"], P2["laya"]["E2_confirm"]["yes/no"]
e3 = P2["laya"]["E3"]
e1 = P2["laya"]["E1"]
e4 = P2["laya"]["E4"]
CONF_SETS = ("boolq_heldout", "RTE", "MultiRC", "llm_heldout")
SET_NAMES = {"boolq_heldout": "Held-out BoolQ (300)", "RTE": "RTE (277, human-written)",
             "MultiRC": "MultiRC (500, human-written)", "llm_heldout": "GPT-written held-out (292)"}
NEW_WORDINGS = ("correct_incorrect", "agree_disagree", "valid_invalid", "supported_unsupported", "the_case")
restore = e3["restoration"]

# ---------------------------------------------------------------- figures
figs = {}

fig, ax = plt.subplots(figsize=(7, 2.8))
names = ["yes / no", "true / false", "A / B", "◇ / △"]
vals = [yn, tf, ab_, ns]
x = np.arange(4)
ax.bar(x - 0.2, [v["acc_normal"] for v in vals], 0.38, color=BLUE, label="normal labels")
ax.bar(x + 0.2, [v["acc_swapped"] for v in vals], 0.38, color=ORANGE, label="label words swapped")
for i, v in enumerate(vals):
    ax.text(i + 0.2, v["acc_swapped"] + 0.02, pct(v["label_following_rate"]) + " follow", ha="center", fontsize=8,
            color=INK2)
ax.set_xticks(x, names)
ax.set_ylabel("BoolQ accuracy (700 items, fp32)")
ax.set_ylim(0, 1)
ax.set_title("Swapping meaningful label words flips laya's answers; symbols do not")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
figs["c3_behaviour"] = png(fig)

fig, ax = plt.subplots(figsize=(7.5, 3.0))
conds = ["none", "block L0-4", "block L5-9", "block L10-14", "block L15-20", "block L21-27", "block head", "block all"]
labels_ = ["none", "0–4", "5–9", "10–14", "15–20", "21–27", "head", "all"]
x = np.arange(len(conds))
ax.bar(x - 0.2, [e2x[c]["label_following"] for c in conds], 0.38, color=BLUE, label="BoolQ (exploratory)")
ax.bar(x + 0.2, [e2c[c]["label_following"] for c in conds], 0.38, color=VIOLET, label="RTE + MultiRC (confirmatory)")
ax.set_xticks(x, labels_)
ax.set_xlabel("layers in which [MASK] → label attention is blocked")
ax.set_ylabel("yes/no label-following")
ax.set_ylim(0, 1.05)
ax.set_title("No single early band is needed; the late band matters most; blocking all removes it")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
figs["c3_timing"] = png(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for k, col in (("question", BLUE), ("options", ORANGE), ("masks", AQUA), ("state", YELLOW), ("special", MAGENTA)):
    axes[0].plot(range(len(restore[k])), restore[k], color=col, marker="o", ms=2.5, lw=1.8, label=k)
axes[0].set_title("Evidence path: restore into a blanked-passage run", fontsize=9.5)
axes[0].set_xlabel("layer")
axes[0].set_ylabel("recovery (median)")
axes[0].legend(fontsize=7.5)
jp = P2["laya"]["E1_joint_patch"]
for k, col in (("label", ORANGE), ("state", YELLOW), ("label+state", BLUE), ("question", MAGENTA), ("masks", AQUA)):
    axes[1].plot(range(len(jp[k]["median"])), jp[k]["median"], color=col, marker="o", ms=2.5, lw=1.8, label=k)
axes[1].set_title("Joint patching into swapped labels + blank passage", fontsize=9.5)
axes[1].set_xlabel("layer")
axes[1].legend(fontsize=7.5)
figs["c3_paths"] = png(fig)

fig, ax = plt.subplots(figsize=(8, 3.0))
sets = list(CONF_SETS)
x = np.arange(len(sets))
ax.bar(x - 0.27, [e4[s]["semantic_yn"]["plain"]["label_following"] for s in sets], 0.26, color=BLUE,
       label="yes/no, plain")
ax.bar(x, [e4[s]["semantic_yn"]["guard"]["label_following"] for s in sets], 0.26, color=AQUA, label="yes/no, guard")
ax.bar(x + 0.27, [np.mean([e4[s][w]["guard"]["label_following"] for w in NEW_WORDINGS]) for s in sets], 0.26,
       color=VIOLET, label="5 new wordings, guard (mean)")
ax.set_xticks(x, [SET_NAMES[s].split(" (")[0] for s in sets])
ax.set_ylabel("label-following")
ax.set_ylim(0, 1.05)
ax.set_title("The guard holds on untouched and human-written sets")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
figs["c3_guard"] = png(fig)

regs = ["baseline", "semantic", "neutral", "random_mapping", "mixture", "planted"]
regnames = ["baseline\n(true/false)", "meaningful\nlabels", "symbols", "random\nmapping", "mixture", "planted\n(label = only cue)"]
P5SETS = ("boolq_main", "boolq_heldout", "RTE", "MultiRC")
p5vals = {r: [np.mean([e[s]["semantic_yn"]["label_following"] for s in P5SETS]) for e in P5[r]] for r in regs}
p5guard = {r: [np.mean([e[s]["semantic_yn_guard"]["label_following"] for s in P5SETS]) for e in P5[r]] for r in regs}
fig, ax = plt.subplots(figsize=(8, 3.0))
x = np.arange(len(regs))
ax.bar(x, [np.mean(p5vals[r]) for r in regs], 0.55, color=[BLUE] * 5 + [ORANGE])
for i, r in enumerate(regs):
    ax.scatter([i] * len(p5vals[r]), p5vals[r], color="#0b0b0b", s=12, zorder=3)
ax.set_xticks(x, regnames, fontsize=8)
ax.set_ylabel("yes/no label-following\n(mean of 4 test sets)")
ax.set_ylim(0, 1.05)
ax.set_title("Only training where the label word identifies the option creates the shortcut (dots: seeds)")
figs["c3_origin"] = png(fig)

P1MODELS = ["laya", "typed", "multilingual", "smolaya", "ctrl_soft_ce", "ctrl_soft_ce_s1", "ctrl_soft_ce_s2",
            "ctrl_rlcd", "ctrl_rlcd_s1", "ctrl_rlcd_s2"]
P1NAMES = ["laya", "typed", "multi-\nlingual", "smolaya", "CE s0", "CE s1", "CE s2", "RLCD s0", "RLCD s1", "RLCD s2"]


def kept(m):
    pc = P1[m]["confirm"]["per_condition"]
    return pc["f01_bind_shared"]["mean_abs_M_centered"]["est"] / pc["f00_bind_origpos"]["mean_abs_M_centered"]["est"]


fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.0))
x = np.arange(len(P1MODELS))
ie = [P1[m]["confirm"]["flip_effects"]["interaction_effect_positions_original"] for m in P1MODELS]
axes[0].bar(x, [d["est"] for d in ie], 0.6, color=BLUE)
axes[0].errorbar(x, [d["est"] for d in ie], yerr=[[d["est"] - d["lo"] for d in ie], [d["hi"] - d["est"] for d in ie]],
                 fmt="none", color="#0b0b0b", capsize=3, lw=1)
axes[0].set_xticks(x, P1NAMES, fontsize=7.5)
axes[0].set_ylabel("flip rate removed")
axes[0].set_title("Flips removed by blocking option interaction (95% CI)")
jit = [P1[m]["confirm"]["per_condition"]["jitter"]["reversal_flip_rate"]["est"] for m in P1MODELS]
f00 = [P1[m]["confirm"]["per_condition"]["f00_bind_origpos"]["reversal_flip_rate"]["est"] for m in P1MODELS]
axes[1].bar(x - 0.2, f00, 0.38, color=BLUE, label="original positions")
axes[1].bar(x + 0.2, jit, 0.38, color=ORANGE, label="random position jitter")
axes[1].set_xticks(x, P1NAMES, fontsize=7.5)
axes[1].set_ylabel("reversal flip rate")
axes[1].set_title("Disturbing positions raises flips in every model")
axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
figs["c1_factorial"] = png(fig)

flow = T1B["info_flow"]
fig, ax = plt.subplots(figsize=(7.5, 3.0))
for k, col, nm in (("j_span", BLUE, "competitor's own span"), ("i_text", AQUA, "target option's text"),
                   ("mask_i", ORANGE, "target [MASK]"), ("shared", MAGENTA, "question/passage"),
                   ("other_options", YELLOW, "other options")):
    ax.plot(range(len(flow[k])), flow[k], color=col, marker="o", ms=2.5, lw=1.8, label=nm)
ax.set_xlabel("layer")
ax.set_ylabel("recovery of option i's logit")
ax.set_title(f"Where a competitor's influence lives ({flow['n_pairs']} pairs)")
ax.legend(fontsize=7.5)
figs["c1_flow"] = png(fig)

fig, ax = plt.subplots(figsize=(8, 2.9))
mods = ["laya", "typed", "multilingual"]
keys = [("base", "baseline"), ("a_ablate", "ablate late MLPs"), ("b_rescale_per_item", "rescale per item"),
        ("best_global_temperature", "best global temperature")]
x = np.arange(len(mods))
for j, ((k, nm), col) in enumerate(zip(keys, (INK2, ORANGE, BLUE, AQUA))):
    ax.bar(x + (j - 1.5) * 0.2, [P3[m]["confirm"][k]["ece"] for m in mods], 0.19, color=col, label=nm)
ax.set_xticks(x, mods)
ax.set_ylabel("ECE (confirmatory MC set)")
ax.set_title("Rescaling reproduces the MLP ablation; a temperature beats it")
ax.legend(fontsize=8)
figs["c2"] = png(fig)

# ---------------------------------------------------------------- C5 seed statistics
def seed_stats(metric):
    ce = [metric(RT[m]) for m in ("ctrl_soft_ce", "ctrl_soft_ce_s1", "ctrl_soft_ce_s2")]
    rl = [metric(RT[m]) for m in ("ctrl_rlcd", "ctrl_rlcd_s1", "ctrl_rlcd_s2")]
    p = stats.ttest_ind(rl, ce, equal_var=False).pvalue
    return np.mean(ce), np.std(ce, ddof=1), np.mean(rl), np.std(rl, ddof=1), p


def perm_p(a, b):
    import itertools

    a, b = np.asarray(a, float), np.asarray(b, float)
    allv, n = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    hits = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool)
        m[list(idx)] = True
        hits += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-12
        tot += 1
    return hits / tot


dose_vals = {a: [BENCH[n]["none"]["yes_no"]["real"]["label_following"] for n in ns] for a, ns in DOSE_ARMS.items()}
dose_guard = {a: [BENCH[n]["guard"]["yes_no"]["real"]["label_following"] for n in ns] for a, ns in DOSE_ARMS.items()}
dose_acc = {a: [BENCH[n]["none"]["yes_no"]["real"]["acc_normal"] for n in ns] for a, ns in DOSE_ARMS.items()}
_pl, _bl = dose_vals["Planted: label always right, neutral text"], dose_vals["Baseline: default labels, fixed order"]
p8_perm = perm_p(_pl, _bl)
_d1 = dose_vals["D1: label always right, informative text"]
p9_d1_base, p9_d1_pl = perm_p(_d1, _bl), perm_p(_d1, _pl)
_rob_lf = lambda t, r: [ROB[f"{t}_{r}_s{i}"]["eval"]["conflict_bench"]["yes_no/none"]["label_following"] for i in range(3)]  # noqa: E731
p9_rob = perm_p(_rob_lf("p9rob", "planted"), _rob_lf("p9rob", "baseline"))
_lf2 = lambda n, iv: BENCH2[n][iv]["yes_no"]["real"]["label_following"]  # noqa: E731
q1_planted_ok = sum(_lf2(f"p5_planted_s{i}", "guard_all_layers") <= 0.10 for i in range(6))
q1_ext_ok = sum(_lf2(f"p5_planted_s{i}", "guard_extended_all_layers") <= 0.10 for i in range(6))
q1_ctrl_ok = sum(abs(_lf2(n, "random_edges_all_layers") - _lf2(n, "none")) <= 0.10 for n in BENCH2)
c5_M = seed_stats(lambda d: d["interaction"]["mean_abs_M_centered"]["est"])
c5_ece = seed_stats(lambda d: d["baseline"]["ece_raw"])
c5_acc = seed_stats(lambda d: d["baseline"]["accuracy"]["est"])
p5_p = stats.ttest_ind(p5vals["planted"], p5vals["baseline"], equal_var=False).pvalue
p5_p_sem = stats.ttest_ind(p5vals["semantic"], p5vals["baseline"], equal_var=False).pvalue

# ---------------------------------------------------------------- cost
try:
    bill = subprocess.run(["modal", "billing", "summary"], capture_output=True, text=True, timeout=120).stdout
    metered = next(l.split()[-1] for l in bill.splitlines() if "Metered Cost" in l)
    billed = next(l.split()[-1] for l in bill.splitlines() if "Billed Cost" in l)
except Exception:
    metered, billed = "see Modal dashboard", "see Modal dashboard"

# ---------------------------------------------------------------- text
fp1 = FP32["track1"]
p1c = P1["laya"]["confirm"]["per_condition"]
g_yn_main = GUARD["boolq_main"]["semantic_yn"]
sections = []

sections.append(f"""
<h1>When Labels Override Evidence</h1>
<p class="sub">Causal pathways for label shortcuts, option interaction and confidence in a non-autoregressive decision model</p>
<p class="by">2026-10-09 · Vignesh Kanike · model: convaiinnovations/laya (laya 0.4.1)</p>
<p class="lead">Laya scores each answer option at its own [MASK] token in one encoder pass. Its yes/no answers follow the
<em>meaning</em> of the option label words instead of the passage: swapping "yes"/"no" flips {pct(yn['label_following_rate'])}
of correct answers, while swapping "A"/"B" flips {pct(ab_['label_following_rate'])}. We trace this to a direct, redundant
read of the label by the [MASK] in local attention layers, separate from an evidence route that runs through the question
tokens. A one-edge guard removes the shortcut on untouched human-written data, and controlled retraining shows which kind of
training data creates it. Order sensitivity and confidence are handled by separate, simpler mechanisms.</p>
""")

atg = [
    ["C3 · label shortcut (headline)", "Confirmed",
     f"yes/no swap: {pct(yn['label_following_rate'])} follow the label; A/B: {pct(ab_['label_following_rate'])}"],
    ["C3 · mechanism", "Two pathways",
     f"direct [MASK]→label read, redundant across local layers; evidence via question tokens (restoring them recovers "
     f"{max(restore['question']):.2f})"],
    ["C3 · confirmatory benchmark", "Confirmed (frozen, human-written)",
     f"yes/no label-following {pct(P8['laya']['benchmark']['none']['yes_no']['real']['label_following'], 1)} "
     f"(n = {P8['laya']['benchmark']['none']['yes_no']['real']['denominator']}); guard "
     f"{pct(P8['laya']['benchmark']['guard']['yes_no']['real']['label_following'], 1)}; same-size control edges "
     f"{pct(P8['laya']['benchmark']['random_edges']['yes_no']['real']['label_following'], 1)}"],
    ["C3 · guard", "Confirmed on untouched data",
     f"RTE {pct(e4['RTE']['semantic_yn']['plain']['label_following'])} → {pct(e4['RTE']['semantic_yn']['guard']['label_following'])}; "
     f"MultiRC {pct(e4['MultiRC']['semantic_yn']['plain']['label_following'])} → {pct(e4['MultiRC']['semantic_yn']['guard']['label_following'])}"],
    ["C3 · origin", "Shown by retraining (6 vs 6 seeds)",
     f"planted {pct(np.mean(_pl), 1)} vs baseline {pct(np.mean(_bl), 1)} on the frozen benchmark; exact permutation "
     f"p = {p8_perm:.4f}; a 90%-reliable sole-cue label is enough, and {sum(v > 0.5 for v in _d1)} of 6 seeds rely on a "
     f"label that is redundant with the option text"],
    ["C1 · order vs competition", "Separable",
     f"blocking option interaction removes {ci(P1['laya']['confirm']['flip_effects']['interaction_effect_positions_original'])} "
     f"points of flips; order-invariance keeps {pct(kept('laya'))} of competition"],
    ["C2 · confidence", "Logit amplifier, not regulator",
     f"late-MLP ablation ECE {f3(P3['laya']['confirm']['a_ablate']['ece'])}; per-item rescale "
     f"{f3(P3['laya']['confirm']['b_rescale_per_item']['ece'])}; best temperature {f3(P3['laya']['confirm']['best_global_temperature']['ece'])}"],
    ["C4 · jailbreak", "Supporting", f"AUROC {T4['natural_baseline']['auroc']:.3f}; decodable at layer 0, decided late"],
    ["C5 · CE vs RLCD", "Trend only", f"competition {c5_M[0]:.3f} vs {c5_M[2]:.3f}, p = {c5_M[4]:.2f} (3 seeds each)"],
    ["Replication", "3 checkpoints", "laya, laya-typed-decisions, laya-multilingual (different base)"],
]
sections.append("<h2>At a glance</h2>" + table(["Claim", "Verdict", "Key number"], atg))

sections.append("<h2>1 · Model and instrument</h2>" + f"""
<p>Laya is ModernBERT-large (28 layers, width 1024, 16 heads; global attention on every third layer from layer 0, a ±64-token
window elsewhere) plus two transformer head layers and a per-option scorer. The input is
<code>[CLS] &lt;type&gt; question: &lt;instructions&gt; [SEP] [MASK] option0 [MASK] option1 … [SEP] state [SEP]</code>.
A noul (yes/no) option renders as "<code>false: no, the statement does not hold</code>" / "<code>true: yes, the statement holds</code>".
All analysis uses an explicit re-implementation of the forward pass with named, editable sites, checked against laya's own
<code>predict</code>.</p>""" + table(
    ["Checkpoint", "Max probability difference vs predict", "Local window", "Throughput (H100, bf16, 512 tokens)"],
    [[m, f"{V[m]['prob_maxdiff_vs_predict']:.1e}", f"±{V[m]['window_chosen']}", f"{V[m]['seq_per_s_bf16_sdpa_512']:.0f} seq/s"]
     for m in ("laya", "smolaya", "typed")] +
    [["multilingual", f"{V_ML['prob_maxdiff_vs_predict']:.1e}", f"±{V_ML['window_chosen']}", "—"]]))

sections.append("""<h2>Definitions</h2><ul>
<li><b>Paired transition.</b> Each item is scored with normal and with swapped label words; we report counts of
correct→correct, correct→wrong, wrong→correct and wrong→wrong.</li>
<li><b>Label-following</b> = n(correct→wrong) / n(correct with normal labels); the denominator n is reported. Item-level 95%
intervals are bootstrap intervals over those n items.</li>
<li><b>Swap gap</b> = accuracy with normal labels − accuracy with swapped labels, over all items.</li>
<li><b>Recovery</b> (patching / restoration) per item = (y_patched − y_corrupt) / (y_clean − y_corrupt), y = log-odds of the
"holds" option; items with |y_clean − y_corrupt| &lt; 0.5 are excluded and counted; we report the median with a bootstrap
interval.</li>
<li><b>Seed-level tests</b> use the trained model as the unit: per-seed values are listed and compared with an exact
two-sided permutation test (and Welch's t-test).</li>
<li><b>Confirmatory vs exploratory.</b> The frozen benchmark (600 MNLI + QNLI items, SHA-256 cea5547f…) and its
interventions were fixed before evaluation; analyses added afterwards are labelled exploratory.</li></ul>""")
sections.append("<h2>2 · C3: labels override evidence</h2><h3>2.1 Behaviour</h3>" + figs["c3_behaviour"] + f"""
<p>Test: each option's label word is swapped ("true: no, the statement does not hold"). A model reading the passage is
unaffected; a model reading the label word picks the option the passage contradicts. Label-following is the share of
items answered correctly with normal labels that flip when the words are swapped. The effect replicates on 300 held-out BoolQ
items and on GPT-written items, and laya's model card documents the failure; our contribution is the mechanism and fix.</p>
<h3>2.2 Two pathways</h3>""" + figs["c3_timing"] + table(
    ["Block reading of the passage by…", "Accuracy, plain", "Accuracy, with guard"],
    [[k.replace("block ", "").replace("->state", ""), f3(e3["plain"][k]["acc_normal"]), f3(e3["guard"][k]["acc_normal"])]
     for k in ("none", "block masks->state", "block options->state", "block question->state", "block special->state",
               "block all_nonstate->state")]) + figs["c3_paths"] + f"""
<ul>
<li><b>Label read is redundant and mostly late.</b> Blocking [MASK]→label in layers 10–14 alone leaves label-following at
{pct(e2x['block L10-14']['label_following'])}; blocking layers 21–27 alone halves it ({pct(e2x['block L21-27']['label_following'])});
blocking it everywhere except any single band from 0–20 removes it. An earlier account placing the read at layers 10–14 was
wrong and is withdrawn.</li>
<li><b>Evidence goes through the question tokens.</b> Stopping them from reading the passage drops accuracy from
{f3(e3['plain']['none']['acc_normal'])} to {f3(e3['plain']['block question->state']['acc_normal'])}; stopping the [MASK]s or option
text from reading it changes nothing. Restoring only the question tokens at layer {int(np.argmax(restore['question']))} recovers
{max(restore['question']):.2f} of the decision.</li>
<li><b>No timing race.</b> Label information leaves the label tokens around layer 20 and enters the [MASK] at layers 21–23, the
same layers where the evidence arrives.</li>
<li><b>Contradicting passages</b> ({P2['laya']['n_contra']} BoolQ passages edited by gpt-5.6-terra so the answer flips, checked by
gpt-5.6-luna): with consistent labels laya follows the edited passage {pct(e1['consistent/contra']['follows_passage'])} of the time,
with swapped labels {pct(e1['swapped/contra']['follows_passage'])}. With the passage blanked, swapping labels moves "holds" answers
from {pct(e1['consistent/filler']['frac_holds'])} to {pct(e1['swapped/filler']['frac_holds'])}.</li>
</ul>""")

e4rows = []
for s in CONF_SETS:
    p, g = e4[s]["semantic_yn"]["plain"], e4[s]["semantic_yn"]["guard"]
    nw_p = np.mean([e4[s][w]["plain"]["label_following"] for w in NEW_WORDINGS])
    nw_g = np.mean([e4[s][w]["guard"]["label_following"] for w in NEW_WORDINGS])
    mt_p, mt_g = e4[s]["multi_token"]["plain"]["label_following"], e4[s]["multi_token"]["guard"]["label_following"]
    mt_g2 = P2B[s]["multi_token"]["guard2"]["label_following"]
    e4rows.append([SET_NAMES[s], f"{pct(p['label_following'])} → {pct(g['label_following'])}",
                   f"{f3(p['acc_normal'])} → {f3(g['acc_normal'])}", f"{pct(nw_p)} → {pct(nw_g)}",
                   f"{pct(mt_p)} → {pct(mt_g)} ({pct(mt_g2)})"])
sections.append("<h3>2.3 The guard</h3>" + f"""
<p>The guard blocks noul [MASK]s from attending to their label words in local layers only, installed on a stock
<code>laya.load</code> agent and run through the official <code>predict</code> API (<code>interp/label_guard.py</code>).
On BoolQ yes/no it cuts label-following from {pct(g_yn_main['plain']['label_following'], 1)} to
{pct(g_yn_main['guard']['label_following'], 1)}. Choice questions are bit-identical (max difference
{GUARD['choice_max_prob_diff']:.1g}); jailbreak AUROC moves {GUARD['jailbreak']['plain_auroc']:.3f} →
{GUARD['jailbreak']['guard_auroc']:.3f} ({pct(GUARD['jailbreak']['decision_changes'], 1)} of decisions change); injection AUROC
{GUARD['injection']['plain_auroc']:.3f} → {GUARD['injection']['guard_auroc']:.3f}.</p>""" + figs["c3_guard"] + table(
    ["Untouched set (fp32)", "yes/no following, plain → guard", "Accuracy, plain → guard", "5 new wordings (mean)",
     "Multi-token labels (extended guard)"], e4rows) + """
<p>Multi-token labels ("yes, it does") are the guard's failure mode. The bracketed extended guard also blocks option text from
reading the labels; it was designed after seeing these sets and is exploratory.</p>""")

p5rows = []
for r, nm in zip(regs, ["Baseline: default true/false", "Meaningful labels, aligned", "Meaningless symbols",
                        "Meaningful labels, randomly swapped", "Mixture", "Planted: label word is the only cue"]):
    p5rows.append([nm, f"{pct(np.mean(p5vals[r]), 1)} ± {100 * np.std(p5vals[r], ddof=1):.1f}",
                   f"{pct(np.mean(p5guard[r]), 1)}",
                   f3(np.mean([np.mean([e[s]['semantic_yn']['acc_normal'] for s in P5SETS]) for e in P5[r]]))])
sections.append("<h3>2.4 Where the shortcut comes from</h3>" + figs["c3_origin"] + table(
    ["Training regime (RLCD, 3 seeds)", "yes/no label-following", "With guard", "Accuracy"], p5rows) + f"""
<p>Fifteen models were trained from ModernBERT-large on the same 31,419 items with laya's loss; only the yes/no labels
differed. Only training where the label word is the sole cue to which option means what (shuffled order, identical option
text) creates the shortcut (p = {p5_p:.3f} vs baseline, seeds as units); meaningful labels alone do not (p = {p5_p_sem:.2f}).
The guard's effect on the planted models differs by seed ({", ".join(pct(v) for v in p5guard['planted'])}): the same behaviour
is implemented by different routes, so a fix derived from one model's internals can fail on a retrained sibling. Laya's own
training data are not public, so this shows what can create the shortcut, not what did.</p>""")

reprows = []
for m, nm in (("laya", "laya"), ("typed", "typed-decisions"), ("multilingual", "multilingual")):
    p2 = P2[m]
    reprows.append([nm,
                    f"{pct(p2['E4']['RTE']['semantic_yn']['plain']['label_following'])} → {pct(p2['E4']['RTE']['semantic_yn']['guard']['label_following'])}",
                    f"{pct(p2['E4']['RTE']['multi_token']['plain']['label_following'])} → {pct(p2['E4']['RTE']['multi_token']['guard']['label_following'])}",
                    pct(p2["E2_confirm"]["yes/no"]["block L21-27"]["label_following"]),
                    f"{f3(p2['E3']['plain']['none']['acc_normal'])} → {f3(p2['E3']['plain']['block question->state']['acc_normal'])}",
                    f"{max(p2['E3']['restoration']['question']):.2f}"])
sections.append("<h3>2.5 Replication</h3>" + table(
    ["Checkpoint", "RTE yes/no, plain → guard", "RTE multi-token", "Following after blocking only L21–27",
     "Accuracy when question tokens cannot read passage", "Best question-token restoration"], reprows) +
    "<p>The guard and the two-pathway picture hold on all three checkpoints; where the label is read in depth, and how clean "
    "the question-token relay is, vary by model.</p>")

C1NAMES = {"sequential": "Trained layout", "options_blocked_seqpos": "Options cannot see each other",
           "shipped_parallel_positions": "Options blocked + shared positions", "masked_parallel_all": "Fully masked parallel"}
B8 = P8["laya"]["benchmark"]
tr = B8["none"]["yes_no"]["real"]
ivrows = []
for iv, nm in (("none", "None"), ("guard", "Guard: [MASK]→label, local layers"),
               ("guard_extended", "Extended guard: + option text→label"),
               ("random_edges", "Control: same number of random option edges"),
               ("description_edges", "Control: same number of description edges"), ("sham", "Control: [MASK]→[CLS]"),
               ("evidence_block", "Evidence path blocked: question→passage")):
    row = B8[iv]
    ivrows.append([nm] + [pct(row[w]["real"]["label_following"], 1) for w in ("yes_no", "right_wrong", "multi_right",
                                                                               "neutral_sym")]
                  + [f3(row["yes_no"]["real"]["acc_normal"]),
                     f"{P8['laya']['safety'][iv]['jailbreak']['auroc']:.3f} / {P8['laya']['safety'][iv]['injection']['auroc']:.3f}"])
reprow8 = []
for m in ("laya", "typed", "multilingual"):
    b = P8[m]["benchmark"]
    reprow8.append([m, pct(b["none"]["yes_no"]["real"]["label_following"], 1), pct(b["guard"]["yes_no"]["real"]["label_following"], 1),
                    pct(b["guard_extended"]["multi_right"]["real"]["label_following"], 1),
                    pct(b["random_edges"]["yes_no"]["real"]["label_following"], 1),
                    f"{f3(b['none']['yes_no']['real']['acc_normal'])} → {f3(b['guard']['yes_no']['real']['acc_normal'])}"])
rq = P8["laya"]["restoration_ci"]
best_l = int(np.argmax([x["median"] for x in rq["question"]]))
qb = rq["question"][best_l]
sections.append("<h3>2.6 Confirmatory benchmark and matched controls</h3>" + f"""
<p>A new benchmark of 600 human-written items (300 MNLI, 300 QNLI; never used before) was frozen and hashed before any
evaluation, with {B8['none']['yes_no']['contra']['n'] if 'contra' in B8['none']['yes_no'] else 0} contradicting edits
(gpt-5.6-terra, checked by gpt-5.6-luna) and three never-used wordings plus two multi-token ones. On laya, yes/no
label-following is {pct(tr['label_following'], 1)} (n = {tr['denominator']}, 95% CI {pct(tr['label_following_ci'][0], 1)}–{pct(tr['label_following_ci'][1], 1)});
transitions: correct→correct {tr['correct_correct']}, correct→wrong {tr['correct_wrong']}, wrong→correct {tr['wrong_correct']},
wrong→wrong {tr['wrong_wrong']}.</p>""" + table(
    ["Intervention (laya, fp32)", "yes/no", "right/wrong (new)", "\"yes, that is right\" (new)", "◇/△", "Accuracy",
     "Jailbreak / injection AUROC"], ivrows) + table(
    ["Checkpoint", "yes/no, none", "Guard", "Extended guard, multi-token", "Random-edge control", "Accuracy, none → guard"],
    reprow8) + f"""
<ul><li>The guard is specific: blocking the same number of other edges, or a sham edge, leaves label-following unchanged.</li>
<li>The extended guard, designed earlier on other sets, passes this first confirmatory test, including multi-token labels.</li>
<li>On contradicting passages the guard cuts label-following from
{pct(B8['none']['yes_no']['contra']['label_following'], 1)} to {pct(B8['guard']['yes_no']['contra']['label_following'], 1)}.
With the passage blanked, swapping labels moves "holds" answers from {pct(B8['none']['yes_no']['blank']['frac_holds_normal'])} to
{pct(B8['none']['yes_no']['blank']['frac_holds_swapped'])}; with the guard, {pct(B8['guard']['yes_no']['blank']['frac_holds_normal'])} to
{pct(B8['guard']['yes_no']['blank']['frac_holds_swapped'])}.</li>
<li>Recovery from restoring the question tokens peaks at layer {best_l}: median {qb['median']:.2f}
[{qb['lo']:.2f}, {qb['hi']:.2f}], n = {qb['n']} ({rq['excluded_small_gap']} of {rq['n_items']} excluded for a small gap).</li></ul>""")

dose_rows = []
for arm, vals_ in dose_vals.items():
    dose_rows.append([arm, ", ".join(f"{100 * v:.1f}" for v in vals_), f"{100 * np.mean(vals_):.1f}",
                      ", ".join(f"{100 * v:.1f}" for v in dose_guard[arm]), f3(np.mean(dose_acc[arm]))])
gall_rows = []
for n, g in BENCH2.items():
    gall_rows.append([n] + [f"{pct(g[iv]['yes_no']['real']['label_following'], 1)} ({f3(g[iv]['yes_no']['real']['acc_normal'])})"
                            for iv in ("none", "guard", "guard_all_layers", "guard_extended_all_layers",
                                       "random_edges_all_layers")])
rob_rows = []
for n, d in ROB.items():
    c, h = d["eval"]["conflict_bench"], d["eval"]["boolq_heldout"]
    learned = h["yes_no/none"]["acc_normal"] > 0.7
    rob_rows.append([n.replace("p8rob_", "RoBERTa, lr 2.5e-5, ").replace("p9rob_", "RoBERTa, lr 1e-5, "), f3(h["yes_no/none"]["acc_normal"]) + ("" if learned else " (did not learn)"),
                     pct(c["yes_no/none"]["label_following"], 1), pct(c["yes_no/label"]["label_following"], 1),
                     f"{f3(c['yes_no/none']['acc_normal'])} → {f3(c['yes_no/evidence']['acc_normal'])}"])
sections.append("<h3>2.7 Dose response, guard scope and an external architecture</h3>" + table(
    ["Training arm (RLCD; shuffled order except baseline)", "yes/no label-following per seed (%)", "Mean (%)",
     "With local guard per seed (%)", "Accuracy"], dose_rows) + f"""
<p>Planted vs baseline with 6 seeds each: exact two-sided permutation p = {p8_perm:.4f} (the smallest value this test allows:
every planted model exceeds every baseline model). A sole-cue label that is right only 75% or 90% of the time already
produces strong label-following. A label that is redundant with informative option text (D1, 6 seeds) is also relied on in
{sum(v > 0.5 for v in _d1)} of 6 seeds: D1 vs baseline p = {p9_d1_base:.3f}, D1 vs planted p = {p9_d1_pl:.2f}. The pre-registered
prediction that D1 would stay low was wrong: the label word only needs to identify the option, not be its sole cue. Baseline models
are not immune on this benchmark (one of six seeds at {100 * max(_bl):.0f}%).</p>
<p><b>Guard scope, confirmatory test (Phase 9).</b> After seeing the local guard work unevenly on retrained models, we froze a
second benchmark (600 new MNLI + QNLI items, no overlap, SHA-256 05c0217c…) and a prediction: blocking [MASK]→label in all layers
leaves ≤ 10% label-following in laya, typed and multilingual and in at least 5 of 6 planted seeds, while the same number of random
edges blocked in all layers changes nothing. Result: the Laya checkpoints pass; only {q1_planted_ok} of 6 planted seeds do, so the
prediction <b>failed</b>. The random-edge control stays within 10 points of no intervention in {q1_ctrl_ok} of {len(BENCH2)} models,
so the effect is specific. Adding option text→label edges (extended, all layers) reaches ≤ 10% in {q1_ext_ok} of 6 seeds.</p>""" + table(
    ["Fresh benchmark (yes/no, accuracy)", "None", "Guard, local layers", "Guard, all layers", "Extended guard, all layers",
     "Control: random edges, all layers"], gall_rows) + """
<p>Reading: retrained models read the label through the same two edge classes as laya ([MASK]→label and option text→label), in
different layers and proportions per seed. In two seeds (s1, s5) blocking them also costs 20–30 accuracy points, and seed 1 keeps
about a quarter of its label-following: there the label is entangled with normal answering, and an attention block alone is
not a clean fix.</p>""" + table(
    ["External model (RoBERTa-large, option-marker head)", "BoolQ accuracy", "Label-following", "With [MASK]→label block",
     "Accuracy, none → question→passage blocked"], rob_rows) + """
<p>At laya's learning rate (2.5e-5, no warmup) 3 of 6 RoBERTa runs did not learn. Rerun at 1e-5 (pre-registered, all runs
reported): all 6 learned (BoolQ ≥ 0.77). Every planted run exceeds every baseline run (exact p = """ + f"{p9_rob:.2f}" + """, the
minimum for 3 vs 3 seeds), and the all-layer block brings 2 of 3 planted runs to ≤ 10% (the third: 37.6%). One baseline run
also learned the shortcut (66.8%). The training-origin result therefore replicates in a second architecture; the block is
effective but not complete.</p>""")

c1rows = [[C1NAMES[c], f3(fp1[c]["accuracy"]), pct(fp1[c]["reversal_flip_rate"], 1), f3(fp1[c]["mean_abs_M_centered"])]
          for c in ("sequential", "options_blocked_seqpos", "shipped_parallel_positions", "masked_parallel_all")]
p1rows = []
for m, nm in zip(P1MODELS, ["laya", "typed", "multilingual", "smolaya", "CE s0", "CE s1", "CE s2", "RLCD s0",
                            "RLCD s1", "RLCD s2"]):
    c = P1[m]["confirm"]
    p1rows.append([nm, ci(c["flip_effects"]["interaction_effect_positions_original"]), pct(kept(m)),
                   f"{100 * c['gate_B_accuracy_drop']:.1f}", pct(c["per_condition"]["jitter"]["reversal_flip_rate"]["est"])])
sections.append("<h2>3 · C1: order sensitivity is positional; competition is separate</h2>" + f"""
<p>Reversing the options flips {pct(fp1['sequential']['reversal_flip_rate'], 1)} of laya's multiple-choice answers (fp32, 1,392 items).
Options do compete: blanking one shifts another's centred logit by {fp1['sequential']['mean_abs_M_centered']:.2f} on average. Slot
index explains {100 * T1['order_map']['main']['order_variance_explained_by_slot']:.1f}% of order-induced variance, so this is not a
simple primacy bias.</p>""" + table(["Condition (laya, fp32)", "Accuracy", "Reversal flips", "Mean abs(M)"], c1rows) + f"""
<p><b>Position binds each [MASK] to its option.</b> Sharing positions while leaving attention unrestricted makes every [MASK]
identical (same token, position and keys), so all options score the same (accuracy {f3(p1c['shared_only_degenerate']['accuracy']['est'])}).
The factorial therefore holds binding fixed (each [MASK] reads only its own option) and varies option-text interaction × positions.
With shared positions and symmetric masks the model is order-invariant by construction; the measured quantities are the accuracy
cost and how much competition survives.</p>""" + figs["c1_factorial"] + table(
    ["Model (confirmatory set)", "Flips removed by blocking interaction (pts, 95% CI)", "Competition kept when order-invariant",
     "Accuracy cost (pts)", "Flips under random jitter"], p1rows) + figs["c1_flow"] +
    f"<p>Position permutation (Phase 8): giving the options reversed position ids without moving the tokens reproduces the "
    f"reversed run in {pct(P8P4['laya']['confirm']['position_vs_token']['pos_permuted_matches_reversed_given_flip'])} of flips in all "
    f"10 models. This holds by construction, since a bidirectional encoder sees only (token, position) pairs, so order effects "
    f"are positional by definition; the substantive result is the factorial one above. The competition measure is robust to the "
    f"replacement used (laya: filler {P8P4['laya']['confirm']['M_filler']['est']:.2f}, dash {P8P4['laya']['confirm']['M_dash']['est']:.2f}, "
    f"distractor {P8P4['laya']['confirm']['M_distractor']['est']:.2f}).</p>"
    "<p>A competitor's content builds up in the target option's text tokens through about layer 16, then enters the target "
    "[MASK] at layers 17–22; no small set of heads carries it.</p>")

c2rows = []
for k, nm in (("base", "Baseline"), ("a_ablate", "Ablate late MLPs"), ("b_rescale_per_item", "Rescale each item to the ablated size"),
              ("b_global_temperature", "One temperature, same mean scale"), ("best_global_temperature", "Best global temperature"),
              ("c1_ablate_at_mask_only", "Ablate only at the [MASK]s"), ("c2_ablate_except_mask", "Ablate everywhere except the [MASK]s")):
    r = P3["laya"]["confirm"][k]
    c2rows.append([nm, f3(r["accuracy"]), f3(r["ece"]), f3(r["brier"]), f3(r["nll"]), pct(r["argmax_kept"], 1)])
sr = P3["laya"]["confirm"]["scale_ratio"]
sections.append("<h2>4 · C2: a late-MLP logit amplifier</h2>" + f"""
<p>Removing MLPs {', '.join(str(l) for l in P3['laya']['mlp_layers'])} halves laya's logit scale (ratio {sr['mean']:.2f}) and fixes most
miscalibration, but per-item rescaling reproduces it and a single temperature beats it, so by the pre-registered rule these MLPs
are reported as an amplifier, not a confidence regulator. The effect sits entirely at the [MASK] positions. Unlike the entropy
neurons described for decoder LLMs, which the authors distinguish from a temperature, these behave like one.</p>""" + table(
    ["Laya, confirmatory MC set (fp32)", "Accuracy", "ECE", "Brier", "NLL", "Answers kept"], c2rows) + figs["c2"] + f"""
<p>The scale ratio barely depends on the input (coefficient of variation {sr['cv']:.2f}; correlation with correctness
{sr['corr_with_correct']:+.2f}; with the original margin {sr['corr_with_base_margin']:+.2f}). In typed-decisions and multilingual the
selected late MLPs carry nearly the whole logit scale (ratios {P3['typed']['confirm']['scale_ratio']['mean']:.3f} and
{P3['multilingual']['confirm']['scale_ratio']['mean']:.3f}); the same rescaling verdict holds. With the temperature fitted on the
exploratory split (T = {P8['laya']['c2_heldout_T']['T_fitted_on_explore']:.2f}) and applied to the confirmatory split, ECE is
{P8['laya']['c2_heldout_T']['confirm_heldout_temperature']['ece']:.3f} vs {P8['laya']['c2_heldout_T']['confirm_mlp_ablation']['ece']:.3f}
for MLP ablation.</p>""")

best = STEER["best_specific"]
sections.append("<h2>5 · C4: jailbreak and injection (supporting)</h2>" + f"""
<ul>
<li>Detection AUROC {T4['natural_baseline']['auroc']:.3f} on natural jailbreaks, {T4['injection_baseline']['auroc']:.3f} on injections.</li>
<li>A linear probe reads the attack from layer {T4['decodable_layer_state']} at {T4['probe_natural_state'][0]['acc']:.2f}; patching shows the
decision carried by the prompt until layer {T4['state_patch_last_layer_ge_0.5']}, relayed by question tokens, and completed in the
head layers. This parallels the early-decodable, late-decisive pattern Wen et al. (2026) report for prompt-injection compliance in
decoder LLMs. Unlike their late layer, which tracks intent and resists obfuscation, laya's decision is sensitive to surface style (next items).</li>
<li>Adding the jailbreak direction at the layer-{best['layer']} [MASK]s raises benign prompts flagged from
{pct(STEER['base']['benign_flagged'])} to {pct(best['benign_flagged'])} with an unrelated question at {pct(best['urgency_flagged'])}.</li>
<li>After gpt-5.6-terra removed only the override from {LLM['counterfactuals']['n']} flagged jailbreaks, laya still flags
{pct(LLM['counterfactuals']['edited_flagged'])}: it reacts to jailbreak style more than to the instruction (edits not hand-checked).</li>
<li>Style test: the same 150 harmless requests are flagged {pct(STYLE['jailbreak']['flag_rate_plain'], 1)} written plainly and
{pct(STYLE['jailbreak']['flag_rate_styled'], 1)} written in jailbreak style (persona/role-play, no override); for the injection question
{pct(STYLE['injection']['flag_rate_plain'], 1)} vs {pct(STYLE['injection']['flag_rate_styled'], 1)}. The pairs are GPT-written and await a
human audit.</li>
<li>Hindi (laya-multilingual): projecting out the English-fit direction drops detection of translated jailbreaks from
{pct(T4M['ablate_hindi_jailbreaks']['before'])} to {pct(T4M['ablate_hindi_jailbreaks']['after'], 1)}.</li>
</ul>""")

sections.append("<h2>6 · C5: training objective</h2>" + table(
    ["3 seeds per arm", "Cross-entropy", "RLCD (laya's loss)", "Welch p"],
    [["Competition, mean abs(M)", f"{c5_M[0]:.3f} ± {c5_M[1]:.3f}", f"{c5_M[2]:.3f} ± {c5_M[3]:.3f}", f"{c5_M[4]:.2f}"],
     ["Calibration error", f"{c5_ece[0]:.3f} ± {c5_ece[1]:.3f}", f"{c5_ece[2]:.3f} ± {c5_ece[3]:.3f}", f"{c5_ece[4]:.2f}"],
     ["Accuracy", f"{c5_acc[0]:.3f} ± {c5_acc[1]:.3f}", f"{c5_acc[2]:.3f} ± {c5_acc[3]:.3f}", f"{c5_acc[4]:.2f}"]]) +
    "<p>RLCD trends toward less competition and better calibration; neither difference is significant with 3 seeds. The "
    "position effect, the late-MLP amplifier and the absence of label-following appear under both losses.</p>")

sections.append("""<h2>7 · Related work</h2>
<p>Order and option-ID bias are documented for decoder LLMs (Pezeshkpour &amp; Hruschka 2024; Zheng et al. 2024), and Anchored
Answers traces GPT-2's preference for "A" to MLP value vectors. Set-Based Prompting removes order dependence by sharing positions
and masking attention; its ablation on 20 MMLU questions found the mask more important than positions in decoders, the opposite of
what blocking attention alone does in laya. Label Words are Anchors studies in-context demonstration labels in decoders, a different
setting from answer-option labels. Confidence Regulation Neurons describes entropy neurons that are explicitly not a temperature;
laya's late MLPs are. Wen et al. (2026) report early-decodable, late-decisive prompt-injection compliance in five decoder LLMs,
with a late representation that tracks intent; laya shares the timing, but not the robustness to surface style. Different seeds
learning different circuits is known (A Toy Model of Universality); our retrained seeds share the edge class but differ in layers. A BERT cross-encoder BM25 circuit
is the closest encoder precedent found. Full notes and the search log are in <code>LITERATURE_NOTES.md</code>.</p>""")

sections.append("""<h2>8 · Limitations and corrections</h2><ul>
<li>Laya was not built for multiple-choice QA (its model card notes near-chance zero-shot performance on typed decisions);
C1 and C2 describe how it handles options, not tuned QA behaviour.</li>
<li>Order-invariance is guaranteed by construction in the shared-position cells; positions alone could not be tested.</li>
<li>The extended guard was designed on earlier sets and has passed one confirmatory test (the frozen benchmark); contradicting
passages and jailbreak rewrites were checked by a second model, not by hand; SAE feature labels were contaminated and are not used.</li>
<li>Human audits of the GPT-written items, contradicting edits, jailbreak rewrites and style pairs are prepared
(<code>audit/</code>) but not yet done.</li>
<li>The all-layer guard's pre-registered prediction failed on retrained models (3 of 6 seeds ≤ 10%); RoBERTa has 3 seeds per arm,
so its comparison cannot reach p &lt; 0.10.</li>
<li>Withdrawn or corrected during the study: the early-timing account of label reading; "a confidence regulator" (now amplifier);
a single-seed "20% less competition" (now a non-significant trend); "steering works only late"; "label-following comes from
training data" (now: which training can create it); "same behaviour, different mechanisms" (now: the same edge classes in different
layers, partly entangled with normal answering in some seeds); "a redundant label is not used" (wrong: 4 of 6 seeds use it); the planned "position vs token order" test (an identity by construction, not evidence).</li>
</ul>""")

sections.append(f"""<h2>9 · Reproduction and cost</h2>
<p>Environment: laya 0.4.1, transformers 5.19.0, torch 2.14.1, nnsight 0.7.0, datasets 5.1.0, Modal client 1.6.1, Python 3.12
on H100 GPUs. Every result file under <code>release/results</code> is produced by
<code>modal run modal_interp.py --task &lt;name&gt;</code> (see <code>release/README.md</code>), and this report is rebuilt from those
files by <code>python release/scripts/build_report.py</code>. Modal compute: {metered} metered, {billed} billed at build time.
OpenAI usage (gpt-5.6-terra and gpt-5.6-luna) is listed in <code>release/README.md</code>.</p>""")

css = """
body{margin:0;background:#fff;color:#0b0b0b;font:11.5px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:900px;margin:0 auto;padding:24px 16px}
h1{font-size:26px;margin:0}.sub{font-size:14px;color:#52514e;margin:4px 0}.by{color:#898781;font-size:11px}
.lead{font-size:13px}h2{font-size:17px;margin:26px 0 6px;padding-top:10px;border-top:1px solid #e2e1dc;page-break-after:avoid}
h3{font-size:13.5px;margin:16px 0 4px;page-break-after:avoid}code{background:#efeee9;padding:0 4px;border-radius:3px;font-size:10.5px}
.tw{overflow-x:auto;border:1px solid #e2e1dc;border-radius:6px;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:10.5px}
th,td{text-align:left;vertical-align:top;padding:5px 7px;border-bottom:1px solid #e2e1dc}th{color:#52514e}
tr:last-child td{border-bottom:none}figure{margin:8px 0;page-break-inside:avoid}img{max-width:100%;display:block;margin:auto}
tr,img{page-break-inside:avoid}ul{padding-left:20px}li{margin:4px 0}@page{size:A4;margin:14mm 12mm}
"""
html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        f'initial-scale=1"><title>When Labels Override Evidence</title><style>{css}</style></head><body><main>'
        + "".join(sections) + "</main></body></html>")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "report.html", len(html))

edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
chrome = shutil.which("google-chrome") or shutil.which("chromium")
browser = edge if pathlib.Path(edge).exists() else chrome
if browser:
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT / 'report.pdf'}", (OUT / "report.html").as_uri()],
                   capture_output=True, timeout=180)
    print("wrote", OUT / "report.pdf")
else:
    print("no headless browser found; open report.html and print to PDF", file=sys.stderr)
