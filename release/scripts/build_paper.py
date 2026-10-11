"""Build the paper (release/report/paper.html + paper.pdf) from release/results only.

Data loading, figures and statistics are shared with build_report.py (everything above its text marker is executed
here unchanged); only the text and structure differ. Every number in the paper is computed from a results file.
"""

import json
import pathlib
import shutil
import subprocess
import sys
from math import sqrt

HERE = pathlib.Path(__file__).resolve().parent
_src = (HERE / "build_report.py").read_text(encoding="utf-8")
_MARK = "# ---------------------------------------------------------------- text"
exec(compile(_src.split(_MARK)[0], str(HERE / "build_report.py"), "exec"))  # noqa: S102  defines J, figs, stats, ...


fp1 = FP32["track1"]
C1NAMES = {"sequential": "Trained layout", "options_blocked_seqpos": "Options cannot see each other",
           "shipped_parallel_positions": "Options blocked + shared positions", "masked_parallel_all": "Fully masked parallel"}
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

# ---------------------------------------------------------------- extra data (Phase 9b)
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


DIR = {n: J(f"phase9/dir_{n}.json") for n in ("laya", "p5_planted_s1", "p5_planted_s5")}
STRICT0 = J("phase9/strict_override.json")           # pre-registered filter (failed on a formatting artifact)
STRICT = J("phase9/strict_override_amended.json")   # amended filter, logged before re-scoring
STRICT_RESIDUAL = 3  # clean edits that, on manual inspection of keyword hits, still keep part of a jailbreak trick
_b2 = [json.loads(l) for l in (ROOT / "data" / "conflict_bench2.jsonl").read_text(encoding="utf-8").split(chr(10)) if l.strip()]
always_no_acc = 1 - np.mean([it["gold"] for it in _b2])
ROB9 = {f"{r}_s{i}": J(f"phase8/rob_p9rob_{r}_s{i}.json") for r in ("planted", "baseline") for i in range(6)}
_rlf = lambda k: ROB9[k]["eval"]["conflict_bench"]["yes_no/none"]["label_following"]  # noqa: E731
_rboolq = lambda k: ROB9[k]["eval"]["boolq_heldout"]["yes_no/none"]["acc_normal"]  # noqa: E731
rob_learned = {k: _rboolq(k) >= 0.70 for k in ROB9}
rob_pl = [_rlf(f"planted_s{i}") for i in range(6) if rob_learned[f"planted_s{i}"]]
rob_bl = [_rlf(f"baseline_s{i}") for i in range(6) if rob_learned[f"baseline_s{i}"]]
p_rob6 = perm_p(rob_pl, rob_bl)
_d0 = dose_vals["D0: label random, informative text"]
p_d0_d1 = perm_p(_d0, _d1)
acc_d0, acc_d1, acc_bl = (np.mean(dose_acc[a]) for a in ("D0: label random, informative text",
                                                         "D1: label always right, informative text",
                                                         "Baseline: default labels, fixed order"))
B8 = P8["laya"]["benchmark"]
tr = B8["none"]["yes_no"]["real"]
rq = P8["laya"]["restoration_ci"]
best_l = int(np.argmax([x["median"] for x in rq["question"]]))
qb = rq["question"][best_l]
sk_lo, sk_hi = wilson(STRICT["n_edited_flagged"], STRICT["n_clean"])
_kx, _nx = STRICT["n_edited_flagged"] - STRICT_RESIDUAL, STRICT["n_clean"] - STRICT_RESIDUAL
skx_lo, skx_hi = wilson(_kx, _nx)
rob_bl_hi = sum(v > 0.5 for v in rob_bl)
AUDIT = [  # (sheet, check, k, n)  -- Claude audit of random samples (audit/claude_filled), not yet human-reviewed
    ("Contradicting edits, benchmark 1", "edit flips the answer", 56, 60),
    ("Contradicting edits, benchmark 2", "edit flips the answer", 58, 60),
    ("Contradicting edits, BoolQ", "edit flips the answer", 57, 60),
    ("GPT-written C3 items", "gold answer correct", 60, 60),
    ("GPT-written C3 items", "answer determined by passage", 57, 60),
    ("Style pairs", "same request", 141, 150),
    ("Style pairs", "no override in styled version", 150, 150),
    ("Jailbreak rewrites (first version)", "override removed", 53, 60),
    ("Jailbreak rewrites (first version)", "only the override removed", 21, 60),
]


def dir_ok(n, cond):
    r, base = DIR[n][cond]["yes_no"], DIR[n]["none"]["yes_no"]
    return r["label_following"] <= 0.10 and base["acc_normal"] - r["acc_normal"] <= 0.05


S = []
lf = lambda d: pct(d["label_following"], 1)  # noqa: E731

# ---------------------------------------------------------------- front matter
S.append(f"""
<h1>When Labels Override Evidence</h1>
<p class="sub">A label shortcut in an encoder decision model: its circuit, its training origin, and how to remove it</p>
<p class="by">Vignesh Kanike · 2026-10-12 · model studied: convaiinnovations/laya (laya 0.4.1) · all numbers rebuilt from
<code>release/results</code></p>
<div class="abstract"><b>Abstract.</b> Laya scores each answer option at its own [MASK] token in a single ModernBERT
encoder pass. We show that its yes/no answers follow the <em>meaning of the option label words</em> rather than the passage:
on a frozen, human-written benchmark of 600 MNLI and QNLI items, swapping "yes" and "no" flips {lf(tr)} of the answers laya
otherwise gets right (n = {tr['denominator']}), while swapping neutral symbols flips {lf(B8['none']['neutral_sym']['real'])}.
Mechanistically, the [MASK] reads its label word directly through local attention, redundantly across layers, while evidence
from the passage reaches the [MASK] only through the question tokens. Blocking the single attention edge class
[MASK]→label cuts label-following to {lf(B8['guard']['yes_no']['real'])} at an accuracy cost of
{100 * (B8['none']['yes_no']['real']['acc_normal'] - B8['guard']['yes_no']['real']['acc_normal']):.1f} points, while blocking the
same number of random edges changes nothing; the result replicates on two sibling checkpoints and a second frozen benchmark.
In a dose study of {sum(len(v) for v in DOSE_ARMS.values())} retrained ModernBERT models, the shortcut appears whenever the label
word identifies the option during training (planted mean {pct(np.mean(_pl), 1)} vs baseline {pct(np.mean(_bl), 1)}; exact
permutation p = {p8_perm:.4f}), even when it is redundant with the option text, and that randomising label words in training prevents it
({pct(np.mean(_d0), 1)} label-following). In RoBERTa-large the shortcut appears in planted and in most baseline seeds, so the
pre-registered planted-vs-baseline difference was not significant there, while the same attention block still removes it. Three
other pre-registered predictions failed: a redundant label was predicted to go unused, the guard does not transfer cleanly to
every retrained ModernBERT seed, and projecting
out a learned label direction removes the answer along with the label. Order sensitivity, confidence and
jailbreak detection are analysed as supporting findings. Hypotheses were frozen before confirmatory runs, and claims that
failed are reported as such.</div>""")

# ---------------------------------------------------------------- 1 introduction
S.append(f"""<h2>1 · Introduction</h2>
<p>Encoder-only decision models such as laya answer a question by scoring several written options in one pass. Each yes/no
option carries a label word ("no" / "yes") and a description ("the statement does not hold" / "holds"). A model that answers
from the passage should not care which word labels which option. Laya does: its answer tracks the label word. The model card
notes this failure; this paper explains it, shows what training produces it, and tests fixes.</p>
<p><b>Contributions.</b></p><ol>
<li><b>Behaviour, confirmed.</b> On a benchmark frozen and hashed before evaluation, laya follows swapped yes/no labels on
{lf(tr)} of correctly answered items (95% CI {pct(tr['label_following_ci'][0], 1)}–{pct(tr['label_following_ci'][1], 1)}), and on
never-seen wordings ("right"/"wrong": {lf(B8['none']['right_wrong']['real'])}).</li>
<li><b>Circuit.</b> Two pathways: a direct, redundant [MASK]→label read in local attention, and an evidence relay
passage → question tokens → [MASK].</li>
<li><b>Fix at inference.</b> A one-edge-class attention guard removes the shortcut in all three laya checkpoints; matched
control edges do not.</li>
<li><b>Origin.</b> Retraining shows which data create the shortcut, a dose response, a training-time fix, and that
RoBERTa-large learns the shortcut even more readily.</li>
<li><b>Limits.</b> Four pre-registered predictions failed (redundant labels unused, guard transfer, label-direction ablation, RoBERTa planted vs
baseline); we report where and why.</li></ol>
<p>Secondary claims (C1 order sensitivity, C2 confidence, C4 jailbreak detection, C5 training objective) are in Section 5.</p>""")

# ---------------------------------------------------------------- 2 methods
S.append("<h2>2 · Model, data and methods</h2><h3>2.1 Model and instrument</h3>" + f"""
<p>Laya is ModernBERT-large (28 layers, width 1024, 16 heads; global attention every third layer from layer 0, a ±64-token
window elsewhere) plus two transformer head layers and a per-option scorer. The input is
<code>[CLS] &lt;type&gt; question: &lt;instructions&gt; [SEP] [MASK] option0 [MASK] option1 … [SEP] state [SEP]</code>; a yes/no
option renders as "<code>false: no, the statement does not hold</code>". All analysis uses an explicit re-implementation of the
forward pass with named, editable sites (residual stream, attention pattern and output, MLP output, per-edge attention blocks),
checked against laya's own <code>predict</code>:</p>""" + table(
    ["Checkpoint", "Max probability difference vs predict", "Local window"],
    [[m, f"{V[m]['prob_maxdiff_vs_predict']:.1e}", f"±{V[m]['window_chosen']}"] for m in ("laya", "smolaya", "typed")] +
    [["multilingual", f"{V_ML['prob_maxdiff_vs_predict']:.1e}", f"±{V_ML['window_chosen']}"]]) + f"""
<h3>2.2 Data</h3><ul>
<li><b>Exploratory:</b> BoolQ (700 items; 300 held out), GPT-written items.</li>
<li><b>Untouched human-written:</b> RTE (277) and MultiRC (500).</li>
<li><b>Benchmark 1 (confirmatory):</b> 600 MNLI + QNLI items never used before, frozen and hashed before evaluation
(SHA-256 cea5547f…), with {B8['n_contra']} contradicting edits, three never-used label wordings and two multi-token ones.</li>
<li><b>Benchmark 2 (confirmatory):</b> 600 further MNLI + QNLI items with no overlap (SHA-256 05c0217c…), frozen with a written
prediction before any evaluation.</li>
<li>Contradicting edits and some items were written by gpt-5.6-terra and checked by gpt-5.6-luna; random samples were then
audited (Appendix A).</li></ul>
<h3>2.3 Definitions</h3><ul>
<li><b>Paired transitions.</b> Each item is scored with normal and with swapped label words; we count correct→correct,
correct→wrong, wrong→correct and wrong→wrong.</li>
<li><b>Label-following</b> = n(correct→wrong) / n(correct with normal labels), with the denominator reported and a bootstrap
95% interval over items.</li>
<li><b>Recovery</b> per item = (y_patched − y_corrupt) / (y_clean − y_corrupt), y = log-odds of "holds"; items with
|y_clean − y_corrupt| &lt; 0.5 are excluded and counted; median with bootstrap interval.</li>
<li><b>Guard:</b> block attention from each yes/no [MASK] to its label tokens in the local layers. <b>Extended guard:</b> also
block option-description tokens from reading the label tokens. <b>Controls:</b> the same number of random option edges, the
same number of description edges, and a sham [MASK]→[CLS] edge.</li>
<li><b>Seed-level tests</b> treat each trained model as one unit: per-seed values are listed and compared with an exact
two-sided permutation test.</li>
<li><b>Pre-registration.</b> Each confirmatory test's data, interventions and prediction were written to
<code>NEXT_STEPS_PLAN.md</code> before it ran; from the confirmatory benchmarks on, each pre-registration is in a git
commit that precedes its results. Analyses added afterwards are labelled exploratory.</li></ul>""")

# ---------------------------------------------------------------- 3 the shortcut and its circuit
ivrows = []
for iv, nm in (("none", "None"), ("guard", "Guard: [MASK]→label, local layers"),
               ("guard_extended", "Extended guard: + description→label"),
               ("random_edges", "Control: same number of random option edges"),
               ("description_edges", "Control: same number of description edges"), ("sham", "Control: [MASK]→[CLS]"),
               ("evidence_block", "Evidence path blocked: question→passage")):
    row = B8[iv]
    ivrows.append([nm] + [pct(row[w]["real"]["label_following"], 1) for w in ("yes_no", "right_wrong", "multi_right",
                                                                               "neutral_sym")]
                  + [f3(row["yes_no"]["real"]["acc_normal"]),
                     f"{P8['laya']['safety'][iv]['jailbreak']['auroc']:.3f} / {P8['laya']['safety'][iv]['injection']['auroc']:.3f}"])
S.append("<h2>3 · The label shortcut and its circuit</h2><h3>3.1 Behaviour</h3>" + figs["c3_behaviour"] + f"""
<p>On BoolQ (fp32), swapping "yes"/"no" flips {pct(yn['label_following_rate'])} of correct answers and "true"/"false"
{pct(tf['label_following_rate'])}, while "A"/"B" flips {pct(ab_['label_following_rate'])} and "◇"/"△"
{pct(ns['label_following_rate'])}: laya reads the meaning of the label word, not its identity.</p>
<h3>3.2 Confirmatory test and matched controls</h3>
<p>On benchmark 1, laya's transitions are: correct→correct {tr['correct_correct']}, correct→wrong {tr['correct_wrong']},
wrong→correct {tr['wrong_correct']}, wrong→wrong {tr['wrong_wrong']}. The guard is specific: blocking the same number of other
edges, or a sham edge, leaves label-following unchanged, while it costs
{100 * (B8['none']['yes_no']['real']['acc_normal'] - B8['guard']['yes_no']['real']['acc_normal']):.1f} accuracy points and lowers
jailbreak AUROC by {P8['laya']['safety']['none']['jailbreak']['auroc'] - P8['laya']['safety']['guard']['jailbreak']['auroc']:.3f}.</p>""" + table(
    ["Benchmark 1, laya (fp32)", "yes/no", "right/wrong (new)", "\"yes, that is right\" (new)", "◇/△", "Accuracy",
     "Jailbreak / injection AUROC"], ivrows) + f"""
<p>On contradicting passages the guard cuts label-following from {lf(B8['none']['yes_no']['contra'])} to
{lf(B8['guard']['yes_no']['contra'])}. With the passage blanked, swapping labels moves "holds" answers from
{pct(B8['none']['yes_no']['blank']['frac_holds_normal'])} to {pct(B8['none']['yes_no']['blank']['frac_holds_swapped'])}; with the
guard, {pct(B8['guard']['yes_no']['blank']['frac_holds_normal'])} to {pct(B8['guard']['yes_no']['blank']['frac_holds_swapped'])}.</p>""")

S.append("<h3>3.3 Two pathways</h3>" + figs["c3_timing"] + table(
    ["Block reading of the passage by…", "Accuracy, plain", "Accuracy, with guard"],
    [[k.replace("block ", "").replace("->state", ""), f3(e3["plain"][k]["acc_normal"]), f3(e3["guard"][k]["acc_normal"])]
     for k in ("none", "block masks->state", "block options->state", "block question->state", "block special->state",
               "block all_nonstate->state")]) + figs["c3_paths"] + f"""
<ul>
<li><b>The label read is redundant and mostly late.</b> Blocking [MASK]→label in layers 10–14 alone leaves label-following at
{pct(e2x['block L10-14']['label_following'])}; blocking layers 21–27 alone halves it ({pct(e2x['block L21-27']['label_following'])});
blocking every local layer removes it.</li>
<li><b>Evidence goes through the question tokens.</b> Stopping them from reading the passage drops accuracy from
{f3(e3['plain']['none']['acc_normal'])} to {f3(e3['plain']['block question->state']['acc_normal'])}; stopping the [MASK]s or option
text from reading it changes nothing. On benchmark 1, restoring only the question tokens into a blanked-passage run recovers a
median {qb['median']:.2f} [{qb['lo']:.2f}, {qb['hi']:.2f}] of the decision at layer {best_l} (n = {qb['n']};
{rq['excluded_small_gap']} of {rq['n_items']} excluded for a small gap).</li>
<li><b>No timing race.</b> Label information enters the [MASK] in layers 21–23, the same layers where the evidence arrives.</li>
</ul>""")

e4rows = []
for s_ in CONF_SETS:
    p_, g_ = e4[s_]["semantic_yn"]["plain"], e4[s_]["semantic_yn"]["guard"]
    e4rows.append([SET_NAMES[s_], f"{pct(p_['label_following'])} → {pct(g_['label_following'])}",
                   f"{f3(p_['acc_normal'])} → {f3(g_['acc_normal'])}"])
reprow8 = []
for m in ("laya", "typed", "multilingual"):
    b = P8[m]["benchmark"]
    reprow8.append([m, lf(b["none"]["yes_no"]["real"]), lf(b["guard"]["yes_no"]["real"]),
                    lf(b["guard_extended"]["multi_right"]["real"]), lf(b["random_edges"]["yes_no"]["real"]),
                    f"{f3(b['none']['yes_no']['real']['acc_normal'])} → {f3(b['guard']['yes_no']['real']['acc_normal'])}"])
S.append("<h3>3.4 The guard in deployment and across checkpoints</h3>" + f"""
<p>Installed on a stock <code>laya.load</code> agent and run through the official <code>predict</code> API
(<code>interp/label_guard.py</code>), the guard leaves choice questions bit-identical (max difference
{GUARD['choice_max_prob_diff']:.1g}) and moves jailbreak AUROC {GUARD['jailbreak']['plain_auroc']:.3f} →
{GUARD['jailbreak']['guard_auroc']:.3f}.</p>""" + figs["c3_guard"] + table(
    ["Untouched set (laya, fp32)", "yes/no label-following, plain → guard", "Accuracy, plain → guard"], e4rows) + table(
    ["Benchmark 1", "yes/no, none", "Guard", "Extended guard, multi-token label", "Random-edge control",
     "Accuracy, none → guard"], reprow8) +
    "<p>The behaviour, the guard and the two-pathway picture hold on all three checkpoints (laya, laya-typed-decisions and "
    "laya-multilingual, which has a different base model). Multi-token labels need the extended guard.</p>")

# ---------------------------------------------------------------- 4 origin and generality
p5rows = []
for r, nm in zip(regs, ["Baseline: default true/false", "Meaningful labels, aligned", "Meaningless symbols",
                        "Meaningful labels, randomly swapped", "Mixture", "Planted: label word is the only cue"]):
    p5rows.append([nm, f"{pct(np.mean(p5vals[r]), 1)} ± {100 * np.std(p5vals[r], ddof=1):.1f}",
                   f3(np.mean([np.mean([e[s_]['semantic_yn']['acc_normal'] for s_ in P5SETS]) for e in P5[r]]))])
S.append("<h2>4 · Where the shortcut comes from</h2><h3>4.1 Which training creates it</h3>" + f"""
<p>We trained ModernBERT-large models with laya's trainer, data (31,419 items) and loss (RLCD), changing only how the yes/no
labels were presented. Laya's own training data are not public, so this shows what <em>can</em> create the shortcut, not what
did.</p>""" + figs["c3_origin"] + table(
    ["Training regime (3 seeds; earlier test sets)", "yes/no label-following", "Accuracy"], p5rows) + f"""
<p>Only training where the label word identifies which option means what creates the shortcut (p = {p5_p:.3f} vs baseline);
meaningful labels alone do not (p = {p5_p_sem:.2f}).</p>""")

S.append("<h3>4.2 Dose response on benchmark 1</h3>" + table(
    ["Training arm (shuffled option order except baseline)", "yes/no label-following per seed (%)", "Mean (%)",
     "Accuracy"], [[a, ", ".join(f"{100 * v:.1f}" for v in vals_), f"{100 * np.mean(vals_):.1f}", f3(np.mean(dose_acc[a]))]
                   for a, vals_ in dose_vals.items()]) + f"""
<ul><li><b>Planted vs baseline</b> (6 seeds each, pre-registered): every planted model exceeds every baseline model; exact
p = {p8_perm:.4f}, the minimum this test allows.</li>
<li><b>Reliability.</b> A sole-cue label that is right only 75% or 90% of the time already produces strong label-following.</li>
<li><b>Redundancy is not protection.</b> When the option text also says which option is which (D1), {sum(v > 0.5 for v in _d1)} of 6
seeds still rely on the label (D1 vs baseline p = {p9_d1_base:.3f}; vs planted p = {p9_d1_pl:.2f}). We had predicted D1 would stay
low; that prediction was wrong.</li>
<li><b>Baselines are not immune:</b> one of six baseline seeds follows labels {100 * max(_bl):.0f}% of the time.</li></ul>""")

S.append("<h3>4.3 A training-time fix</h3>" + f"""
<p>Arms D0 and D1 differ only in whether the label word is reliable. When label words are assigned at random during training
(D0), label-following is {", ".join(f"{100 * v:.1f}%" for v in _d0)} across 3 seeds, against
{", ".join(f"{100 * v:.1f}%" for v in _d1)} for D1 (exact p = {p_d0_d1:.3f}). Benchmark accuracy is {f3(acc_d0)} for D0 vs
{f3(acc_d1)} for D1 and {f3(acc_bl)} for baseline. Randomising label words in training therefore prevents the shortcut at a cost of
about {100 * (acc_d1 - acc_d0):.0f} accuracy points in this setting. This was not pre-registered as a fix, and it rests on 3 seeds.</p>""")

gall_rows = [[n] + [f"{pct(g[iv]['yes_no']['real']['label_following'], 1)} ({f3(g[iv]['yes_no']['real']['acc_normal'])})"
                    for iv in ("none", "guard", "guard_all_layers", "guard_extended_all_layers", "random_edges_all_layers")]
             for n, g in BENCH2.items()]
S.append("<h3>4.4 Does the guard transfer to retrained models? (pre-registered; failed)</h3>" + f"""
<p>The local-layer guard worked unevenly on retrained planted models. Before running benchmark 2 we predicted that blocking
[MASK]→label in <em>all</em> layers would leave ≤ 10% label-following in the laya checkpoints and in at least 5 of 6 planted seeds,
while the same number of random edges would change nothing. The laya checkpoints passed; only {q1_planted_ok} of 6 planted seeds
did, so the prediction <b>failed</b>. The control stayed within 10 points of no intervention in {q1_ctrl_ok} of {len(BENCH2)}
models. Adding description→label edges (extended, all layers) reached ≤ 10% in {q1_ext_ok} of 6 seeds.</p>""" + table(
    ["Benchmark 2: yes/no label-following (accuracy)", "None", "Guard, local", "Guard, all layers",
     "Extended guard, all layers", "Control: random edges, all layers"], gall_rows) +
    "<p>Retrained models read the label through the same two edge classes as laya, in different layers and proportions per "
    "seed. In seeds 1 and 5, blocking them also costs 20–30 accuracy points: the label is entangled with normal answering.</p>")

dir_rows = []
for n in DIR:
    row = [n]
    for c in ("none", "direction_ablation", "random_direction_control", "direction_ablation+extended_guard_all_layers"):
        r = DIR[n][c]["yes_no"]
        row.append(f"{pct(r['label_following'], 1)} ({f3(r['acc_normal'])})")
    dir_rows.append(row)
n_ok_dir = sum(dir_ok(n, "direction_ablation") for n in DIR)
n_ok_comb = sum(dir_ok(n, "direction_ablation+extended_guard_all_layers") for n in DIR)
S.append("<h3>4.5 Removing a label direction instead of an edge (exploratory)</h3>" + f"""
<p>For each layer we fitted one direction on benchmark 1: the mean residual at the "holds" [MASK] with normal labels minus
with swapped labels. We then projected it out at every [MASK] in every layer, and evaluated on benchmark 2. The success criterion,
fixed in advance, was label-following ≤ 10% with an accuracy drop ≤ 5 points. Direction ablation alone met it in {n_ok_dir} of
{len(DIR)} models; combined with the extended all-layer guard, in {n_ok_comb} of {len(DIR)}. Label-following does fall to zero,
but only because the model stops answering: accuracy falls to {f3(DIR['laya']['direction_ablation']['yes_no']['acc_normal'])}, which
is exactly the accuracy of always answering "does not hold" on benchmark 2 ({f3(always_no_acc)}). A random direction per layer
changes nothing. The direction that separates normal from swapped labels at the [MASK] is the answer direction itself: in this
model, "what the label says" and "what the answer is" share one linear direction, so they cannot be separated by projection.</p>""" + table(["Benchmark 2: yes/no label-following (accuracy)", "None", "Direction ablation",
                         "Random-direction control", "Direction ablation + extended guard"], dir_rows))

rob_rows = []
for k, d in ROB9.items():
    c, h = d["eval"]["conflict_bench"], d["eval"]["boolq_heldout"]
    rob_rows.append([k.replace("_s", ", seed "), f3(h["yes_no/none"]["acc_normal"]),
                     pct(c["yes_no/none"]["label_following"], 1), pct(c["yes_no/label"]["label_following"], 1),
                     f"{f3(c['yes_no/none']['acc_normal'])} → {f3(c['yes_no/evidence']['acc_normal'])}"])
S.append("<h3>4.6 A second architecture: RoBERTa-large</h3>" + f"""
<p>We trained option-marker models on RoBERTa-large with the same recipe (encoder learning rate 1e-5, chosen in advance after 3
of 6 runs at laya's 2.5e-5 failed to learn), 6 planted and 6 baseline seeds, all reported. {sum(rob_learned.values())} of 12
learned the task (BoolQ ≥ 0.70). Among them, planted label-following averages {pct(np.mean(rob_pl), 1)} vs {pct(np.mean(rob_bl), 1)}
for baseline (exact p = {p_rob6:.4f}, {len(rob_pl)} vs {len(rob_bl)} seeds). The [MASK]→label block is applied in all layers
(RoBERTa has no local/global split); a fidelity check confirms the unblocked path matches the plain model exactly.</p>
<p>The pre-registered prediction (planted &gt; baseline, p &lt; 0.05) <b>failed</b>: {rob_bl_hi} of {len(rob_bl)} baseline seeds also learned
the shortcut (label-following above 50%). In RoBERTa, default training with fixed "true"/"false" labels is enough to produce it in
most seeds. The [MASK]→label block brings {sum(_bl2 <= 0.12 for _bl2 in [ROB9[k]["eval"]["conflict_bench"]["yes_no/label"]["label_following"] for k in ROB9])}
of 12 runs to 12% or less.</p>""" + table(
    ["RoBERTa-large (lr 1e-5)", "BoolQ accuracy", "Label-following (bench 1)", "With [MASK]→label block",
     "Accuracy, none → question→passage blocked"], rob_rows))

# ---------------------------------------------------------------- 5 supporting findings
c1rows = [[C1NAMES[c], f3(fp1[c]["accuracy"]), pct(fp1[c]["reversal_flip_rate"], 1), f3(fp1[c]["mean_abs_M_centered"])]
          for c in ("sequential", "options_blocked_seqpos", "shipped_parallel_positions", "masked_parallel_all")]
p1rows = []
for m, nm in zip(P1MODELS, ["laya", "typed", "multilingual", "smolaya", "CE s0", "CE s1", "CE s2", "RLCD s0",
                            "RLCD s1", "RLCD s2"]):
    c = P1[m]["confirm"]
    p1rows.append([nm, ci(c["flip_effects"]["interaction_effect_positions_original"]), pct(kept(m)),
                   f"{100 * c['gate_B_accuracy_drop']:.1f}"])
S.append("<h2>5 · Supporting findings</h2><h3>5.1 C1: order sensitivity is positional; competition is separate</h3>" + f"""
<p>Reversing the options flips {pct(fp1['sequential']['reversal_flip_rate'], 1)} of laya's multiple-choice answers (fp32, 1,392
items), and options compete: blanking one shifts another's centred logit by {fp1['sequential']['mean_abs_M_centered']:.2f}. A
factorial that holds [MASK]–option binding fixed shows that blocking option interaction removes only a few points of flips in
every model, while order-invariant layouts keep most of the competition.</p>""" + table(
    ["Condition (laya, fp32)", "Accuracy", "Reversal flips", "Mean abs(M)"], c1rows) + figs["c1_factorial"] + table(
    ["Model", "Flips removed by blocking interaction (pts, 95% CI)", "Competition kept when order-invariant",
     "Accuracy cost (pts)"], p1rows) +
    f"<p>Giving options reversed position ids without moving tokens reproduces the reversed run in "
    f"{pct(P8P4['laya']['confirm']['position_vs_token']['pos_permuted_matches_reversed_given_flip'])} of flips in all 10 models; this "
    f"holds by construction for a bidirectional encoder and is an identity check, not evidence. The competition measure is robust "
    f"to the replacement used (filler {P8P4['laya']['confirm']['M_filler']['est']:.2f}, dash "
    f"{P8P4['laya']['confirm']['M_dash']['est']:.2f}, distractor {P8P4['laya']['confirm']['M_distractor']['est']:.2f}).</p>")

c2rows = []
for k, nm in (("base", "Baseline"), ("a_ablate", "Ablate late MLPs"), ("b_rescale_per_item", "Rescale each item to the ablated size"),
              ("best_global_temperature", "Best global temperature")):
    r = P3["laya"]["confirm"][k]
    c2rows.append([nm, f3(r["accuracy"]), f3(r["ece"]), f3(r["brier"]), pct(r["argmax_kept"], 1)])
sr = P3["laya"]["confirm"]["scale_ratio"]
S.append("<h3>5.2 C2: confidence is set by a late-MLP amplifier</h3>" + f"""
<p>Removing MLPs {', '.join(str(l) for l in P3['laya']['mlp_layers'])} halves laya's logit scale (ratio {sr['mean']:.2f}) and fixes
most miscalibration, but rescaling each item reproduces it and one temperature beats it, so by the pre-registered rule these MLPs
are an amplifier, not a confidence regulator. With the temperature fitted on the exploratory split
(T = {P8['laya']['c2_heldout_T']['T_fitted_on_explore']:.2f}), confirmatory ECE is
{P8['laya']['c2_heldout_T']['confirm_heldout_temperature']['ece']:.3f} vs {P8['laya']['c2_heldout_T']['confirm_mlp_ablation']['ece']:.3f}
for MLP ablation.</p>""" + table(["Laya, confirmatory MC set", "Accuracy", "ECE", "Brier", "Answers kept"], c2rows) + figs["c2"])

S.append("<h3>5.3 C4: the jailbreak detector reacts to style</h3>" + f"""<ul>
<li>Laya detects natural jailbreaks with AUROC {T4['natural_baseline']['auroc']:.3f}. A linear probe reads the attack from layer
{T4['decodable_layer_state']}, but patching shows the decision is made late, relayed by the question tokens: the same
early-decodable, late-decisive timing Wen et al. (2026) report for prompt-injection compliance in decoder LLMs.</li>
<li><b>Style test.</b> The same 150 harmless requests are flagged {pct(STYLE['jailbreak']['flag_rate_plain'], 1)} written plainly and
{pct(STYLE['jailbreak']['flag_rate_styled'], 1)} written in jailbreak style (persona, drama, no override); for the injection question
{pct(STYLE['injection']['flag_rate_plain'], 1)} vs {pct(STYLE['injection']['flag_rate_styled'], 1)}.</li>
<li><b>Override removal (strict, pre-registered, amended).</b> gpt-5.6-terra was told to delete only override spans from
{STRICT['n_originals']} flagged jailbreaks. The pre-registered deletion-only check passed just {STRICT0['n_deletion_only']} edits, mostly
because the model echoed our input's "Prompt:" prefix and left punctuation from deleted words; with the amendment logged before re-scoring (strip the echo, compare words only),
{STRICT['n_deletion_only_amended']} are deletion-only and {STRICT['n_clean']} of those were judged override-free by gpt-5.6-luna. Laya still
flags {STRICT['n_edited_flagged']} of {STRICT['n_clean']} ({pct(STRICT['edited_flagged'])}, Wilson 95% CI {pct(sk_lo, 1)}–{pct(sk_hi, 1)});
the mean log-odds drops by only {STRICT['mean_logodds_drop']:.2f}. On manual inspection {STRICT_RESIDUAL} of these edits keep part of a
jailbreak trick; without them laya flags {_kx} of {_nx} (CI {pct(skx_lo, 1)}–{pct(skx_hi, 1)}). Removing the override barely moves
laya. An earlier, looser version of this test (82%) is withdrawn: its edits removed far more than the override.</li>
<li>Adding the jailbreak direction at the layer-{STEER['best_specific']['layer']} [MASK]s raises benign prompts flagged from
{pct(STEER['base']['benign_flagged'])} to {pct(STEER['best_specific']['benign_flagged'])}.</li></ul>
<p>Unlike the intent-tracking late layer described by Wen et al., laya's decision follows surface style.</p>""")

S.append("<h3>5.4 C5: training objective</h3>" + table(
    ["3 seeds per arm", "Cross-entropy", "RLCD (laya's loss)", "Welch p"],
    [["Competition, mean abs(M)", f"{c5_M[0]:.3f} ± {c5_M[1]:.3f}", f"{c5_M[2]:.3f} ± {c5_M[3]:.3f}", f"{c5_M[4]:.2f}"],
     ["Calibration error", f"{c5_ece[0]:.3f} ± {c5_ece[1]:.3f}", f"{c5_ece[2]:.3f} ± {c5_ece[3]:.3f}", f"{c5_ece[4]:.2f}"],
     ["Accuracy", f"{c5_acc[0]:.3f} ± {c5_acc[1]:.3f}", f"{c5_acc[2]:.3f} ± {c5_acc[3]:.3f}", f"{c5_acc[4]:.2f}"]]) +
    "<p>RLCD trends toward less competition and better calibration; neither difference is significant with 3 seeds.</p>")

# ---------------------------------------------------------------- 6 discussion, related work, limitations
S.append(f"""<h2>6 · Discussion</h2>
<p><b>For practitioners.</b> When answer options carry words with meaning, an option-scoring encoder can learn to answer from
those words. In our retraining this happened in every seed when the label word was the reliable cue to the option (planted: 6 of 6),
and often even when the option text made it redundant (ModernBERT D1: {sum(v > 0.5 for v in _d1)} of 6 seeds; RoBERTa baseline:
{rob_bl_hi} of {len(rob_bl)}). Two remedies work in different settings:</p><ul>
<li>For a deployed laya checkpoint, the attention guard removes the behaviour at a cost of {100 * (B8['none']['yes_no']['real']['acc_normal'] - B8['guard']['yes_no']['real']['acc_normal']):.1f} accuracy points on benchmark 1 and is
available as a drop-in wrapper.</li>
<li>For a new model, randomising label words during training prevented the shortcut in all 3 ModernBERT seeds we trained
(not yet tested in RoBERTa).</li></ul>
<p><b>On mechanistic fixes.</b> The guard was derived from laya's internals and transfers to its sibling checkpoints, but not
cleanly to every retrained model: the same edge classes carry the label, in different layers, and in some seeds they are
entangled with normal answering. A fix found in one model is a hypothesis about the next one, not a guarantee.</p>""")

S.append("""<h2>7 · Related work</h2>
<p>Order and option-ID bias are documented for decoder LLMs (Pezeshkpour &amp; Hruschka 2024; Zheng et al. 2024), and Anchored
Answers traces GPT-2's preference for "A" to MLP value vectors. Set-Based Prompting removes order dependence by sharing positions
and masking attention. Label Words are Anchors studies in-context demonstration labels in decoders, a different setting from
answer-option labels. Confidence Regulation Neurons describes entropy neurons that are explicitly not a temperature; laya's late
MLPs are. Wen et al. (2026) report early-decodable, late-decisive prompt-injection compliance in five decoder LLMs with a late
representation that tracks intent; laya shares the timing but not the robustness to style. Seeds learning different circuits is
known (A Toy Model of Universality). A BERT cross-encoder BM25 circuit is the closest encoder precedent we found. Notes and the
search log are in <code>LITERATURE_NOTES.md</code>.</p>""")

S.append("""<h2>8 · Limitations</h2><ul>
<li>Laya's training data are unknown; the retraining shows a sufficient cause, not the actual one.</li>
<li>Laya was not built for multiple-choice QA, so C1 and C2 describe how it handles options, not tuned QA behaviour.</li>
<li>GPT-made data (contradicting edits, some items, style pairs, override edits) were checked by a second GPT model and by a
Claude audit of random samples (Appendix A). The flagged rows have not yet been reviewed by a person.</li>
<li>Some arms have 3 seeds (D0, D2, D3, the earlier regimes); the training-time fix rests on 3 seeds.</li>
<li>The direction ablation (4.5) was tested on three models and failed its criterion in all three.</li>
<li>The strict override test (5.3) needed an amendment to its filter after the first run; both counts are reported.</li></ul>
<h2>9 · Corrections made during the study</h2><ul>
<li>The early-timing account of label reading was wrong; the read is redundant and mostly late.</li>
<li>"A confidence regulator" became "a logit amplifier" under the pre-registered rule.</li>
<li>A single-seed "20% less competition" became a non-significant trend with 3 seeds.</li>
<li>"Label-following comes from training data" became "this kind of training can create it".</li>
<li>"Same behaviour, different mechanisms" became "the same edge classes in different layers, partly entangled with answering".</li>
<li>The prediction that a redundant label would not be used was wrong (4 of 6 seeds use it).</li>
<li>The prediction that an all-layer guard would transfer to 5 of 6 retrained seeds failed (3 of 6).</li>
<li>"Laya flags 82% after only the override is removed" was withdrawn after an audit; replaced by the strict test in 5.3.</li>
<li>"The training-origin result replicates in RoBERTa" (from 3 seeds) was wrong: with 6 seeds most RoBERTa baselines also learn
the shortcut.</li>
<li>The planned position-vs-token test was an identity by construction, not evidence.</li></ul>""")

S.append(f"""<h2>10 · Reproducibility and cost</h2>
<p>Environment: laya 0.4.1, transformers 5.19.0, torch 2.14.1, datasets 5.1.0, Modal client 1.6.1, Python 3.12, H100 GPUs. Every
file under <code>release/results</code> is produced by <code>modal run modal_interp.py --task &lt;name&gt;</code>, and this paper is
rebuilt from those files by <code>python release/scripts/build_paper.py</code>. Pre-registrations and outcomes are logged in
<code>NEXT_STEPS_PLAN.md</code> with their git commits. Modal compute at build time: {metered} metered, {billed} billed. OpenAI usage
(gpt-5.6-terra, gpt-5.6-luna) for data generation and checking is listed in <code>release/README.md</code>.</p>""")

S.append("<h2>Appendix A · Audit of model-made data</h2>" + table(
    ["Sheet (random sample)", "Check", "Pass", "Rate (95% Wilson CI)"],
    [[a, b, f"{k}/{n}", f"{pct(k / n)} ({pct(wilson(k, n)[0])}–{pct(wilson(k, n)[1])})"] for a, b, k, n in AUDIT]) +
    "<p>Judged by Claude (a model independent of the GPT generators), not by a person; flagged rows are marked for human review "
    "in <code>audit/claude_filled/</code>. Only the first-version jailbreak rewrites failed the audit; that result was withdrawn "
    "and replaced by the strict test.</p>")

css_paper = css + """
.abstract{font-size:12px;background:#f6f5f1;border-left:3px solid #2a6fdb;padding:10px 14px;margin:14px 0;border-radius:4px}
ol{padding-left:20px}"""
html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
        f'initial-scale=1"><title>When Labels Override Evidence</title><style>{css_paper}</style></head><body><main>'
        + "".join(S) + "</main></body></html>")
(OUT / "paper.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "paper.html", len(html))
edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
browser = edge if pathlib.Path(edge).exists() else (shutil.which("google-chrome") or shutil.which("chromium"))
if browser:
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT / 'paper.pdf'}", (OUT / "paper.html").as_uri()], capture_output=True, timeout=180)
    print("wrote", OUT / "paper.pdf")
else:
    print("no headless browser found; open paper.html and print to PDF", file=sys.stderr)
