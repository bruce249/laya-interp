# Next steps: from experiment report to a defensible paper

Plan v4, written 2026-10-09 in response to the external research review (`Laya_Research_Review.md`).
Hypotheses and primary metrics below are frozen **before** any new run. Anything changed later is logged here with a date and reason.

## 0. Central claim and framing

**Central research question.** Can a non-autoregressive decision model's output be explained by separable causal pathways for (1) candidate interaction, (2) evidence retrieval and (3) confidence regulation, and can intervening on those pathways fix a failure without damaging other capabilities?

- **Flagship:** C3, label shortcut vs evidence retrieval. It has the clearest mechanism and a working fix.
- **Core support:** C1 (interaction vs position, completed as a factorial) and C2 (confidence regulation vs plain logit scaling).
- **Demoted:** C4 becomes a short stress-test section. C5 becomes a test of where label-following originates, not a fifth equal claim.
- **Working title:** *When Labels Override Evidence: Causal Pathways for Interaction, Evidence and Confidence in Non-Autoregressive Decision Models.* This is the review's Title B, broadened slightly so C1 and C2 can stay as support.
- **Target:** a mechanistic-interpretability workshop at a major ML/NLP venue first. Consider ACL/EMNLP Findings only after Phases 1–3 hold.

## 1. Immediate wording fixes (no compute)

| Item | Current claim | Corrected claim |
|---|---|---|
| C3 origin | "label-following comes from laya's training data" | "not inevitable for this architecture under either loss; its origin in training is untested" |
| C1 | "order flips come from positions" | "flips persist without option interaction; removed when interaction is blocked *and* positions shared; position alone untested" (until Phase 1) |
| C1 numbers | 0.0% in text vs 1.4% in appendix | label as fp32 vs bf16 runs of the same condition |
| C3 novelty | (implicit) discovery | failure documented on laya's model card; contribution = mechanism + fix |
| Related work | built from search snippets | read the closest papers in full (Phase 6) before stating any contrast, especially the Set-Based Prompting ablation claim |

## 2. Confirmatory data, fixed before any run

Untouched sets. Nothing in them is used to choose heads, layers, strengths, thresholds or templates.

| Set | Source | Use |
|---|---|---|
| MC-confirm | OpenBookQA test (4 options) + 500 items from an MMLU subject split, cast like the current MC set | C1, C2 confirmation |
| Noul-confirm A | GLUE RTE validation (human-written; "Does the text entail the hypothesis?") | C3 confirmation |
| Noul-confirm B | MultiRC or StrategyQA yes/no subset (human-written) | C3 confirmation |
| Label-adversarial | Templated by hand, then checked by a person: paraphrased labels (correct/incorrect, agree/disagree), multi-token labels, semantically related but non-equivalent labels, distractor passages | C3 generalization |

The existing sets remain the exploration sets. Every confirmatory result is reported separately.

## 3. Experiments, in priority order

### Phase 1 · C1 factorial (highest priority, ~$2, 0.5 day)

2×2 design on the same weights: cross-option attention {enabled, blocked} × option positions {original, shared}.

| Condition | Attention | Positions | Status |
|---|---|---|---|
| Baseline | enabled | original | done |
| A | blocked | original | done (flips 28.3%) |
| **B** | **enabled** | **shared** | **missing: run now** |
| C | blocked | shared | done (flips 0.0% fp32) |

- **Primary metrics:** reversal-flip rate and accuracy (reported jointly), plus mean abs(M).
- **Distribution-shift control:** a matched random position perturbation. Shift each option block by random offsets of the same average size as "sharing", so the damage is comparable but the result isn't order-invariant.
- **Report:** a 2×2 decomposition (position main effect, interaction main effect, their interaction term) with bootstrap CIs.
- **Models:** laya, smolaya, laya-multilingual, and the 6 retrained controls.
- **Decision gate:** if B collapses accuracy (more than 10 points below baseline), fine-tune a control with shared positions and enabled attention (laya's trainer supports custom layouts) and repeat. That separates "positions cause flips" from "shared positions break the model".

**Amendment, 2026-10-09 (after the first laya run).** Condition B as written (shared positions, unrestricted attention) is structurally degenerate. All [MASK]s are the same token at the same position attending to the same keys, so the encoder gives them identical vectors and identical scores (verified on 3 items: all 5 option logits equal). Positions are what bind each [MASK] to its own option.

The factorial is therefore redefined with [MASK] binding held fixed in all four cells (each [MASK] reads only its own option), varying option-text ↔ option-text attention {allowed, blocked} × positions {original, shared}. Encoder masks only; head layers are unmasked in all four cells. The degenerate condition and the binding mask's own cost are reported as references. The primary metrics and the gate are unchanged.

### Phase 2 · C3 complete causal account (highest priority, ~$4 + ~$3 API, 1.5 days)

1. **Label × evidence 2×2.** Label {consistent, swapped} × evidence {real passage, filler, contradicting passage}. Patch label and evidence activations separately and jointly.
2. **Timing test (causal, not correlational).** Block [MASK] → label only in layers 10–14 (the early read) vs only in layers ≥ 21. If the early block alone removes label-following while the late block does not, the timing account is supported.
3. **Necessity and sufficiency of the evidence path under the guard.**
   - With the guard on, block [MASK] → state in global layers; accuracy should fall toward chance (evidence path necessary).
   - Restore the evidence-path activations from a clean run into a corrupted run (evidence path sufficient).
4. **Generalization:** Noul-confirm A/B and Label-adversarial. Report label-following, accuracy, calibration (ECE, Brier, NLL) and swap gap, with and without the guard.
5. **Side effects of the guard:** jailbreak/injection AUROC (done), preset decisions (done), plus calibration on all noul sets.
6. **Primary metrics:** label-following rate, normal-label accuracy and swap gap.

**Outcome, 2026-10-09.**

- **Timing hypothesis rejected.** Blocking [MASK] → label only in layers 10–14 leaves label-following at 96%. The read is redundant across local layers, with the late band (21–27) most important.
- **Evidence path confirmed.** It runs through a question-token relay: blocking question → passage drops accuracy to the majority-class rate, and restoring the question tokens at layers 16–17 recovers 91%.
- **Guard confirmed** on RTE, MultiRC, held-out BoolQ and GPT items, across 5 new label wordings.
- **Guard failure found:** multi-token labels (32–38% label-following remain).
- **Exploratory, not confirmatory:** a guard extension that also blocks option text → labels reduces that to 4–13%. It was designed after seeing the confirmatory sets, so it needs a fresh untouched set before it is claimed.

### Phase 3 · C2 regulator vs amplifier (high priority, ~$1, 0.5 day)

Three interventions on the same items:
- (a) ablate MLPs 22/26/27;
- (b) keep the MLPs but rescale each item's logits to (a)'s norm;
- (c) ablate, then restore the MLP outputs at [MASK] positions only, and separately at non-[MASK] positions.

- **Metrics:** ECE, Brier, NLL, full-ranking retention (Kendall tau), argmax retention, and AUROC of the margin, overall and by dataset.
- **Key test:** whether the per-item scale change from (a) varies with difficulty or correctness (regulator) or stays near-constant (generic amplifier).
- **Decision:** if (b) reproduces every effect of (a), report C2 as "a late-MLP logit amplifier" and drop the word "regulator".

**Outcome, 2026-10-09.**

- **Decision rule triggered.** Per-item rescaling reproduces the calibration effect of ablating MLPs 22/26/27, and a global temperature matches or beats it. C2 is now reported as a logit amplifier, not a regulator.
- **Where it acts:** at the [MASK] positions only.
- **How it scales:** about 2× (ratio 0.47–0.49). The ratio does not depend on source, depends only weakly on correctness, and tracks the base margin.
- **Cost:** ablation flips about 4% of answers with no gain.

### Phase 4 · Cross-model replication (high priority, ~$3, 0.5 day)

- Run Phases 1–3 on **laya-multilingual**: a different pretrained base (mmBERT), its own training run, 22 layers.
- Run them on **laya-typed-decisions** as well.
- Stretch: an option-marker model built on a different encoder family (e.g. DeBERTa-v3-large) with laya's trainer. Use it only for C3, since C1's RoPE manipulation does not transfer.

**Outcome, 2026-10-09.**

- **Verification.** typed-decisions verified against `predict` (5e-5).
- **C1 replicates:**

  | | typed | multilingual |
  |---|---|---|
  | Flips removed by blocking interaction | 6.2 pts | 3.7 pts |
  | Competition kept under shared positions | 76% | 76% |
  | Accuracy cost of order-invariance | ≤ 1.4 pts | ≤ 1.4 pts |

- **C3 guard replicates:**
  - yes/no labels: 90–93% → 1–5%;
  - multi-token labels: 5–14% left, versus 33–38% in laya.
- **C3 evidence relay:**
  - **typed:** clean (restoring question tokens recovers 0.91);
  - **multilingual:** weaker (0.75), and option text also reads the passage.
- **Where the label read sits differs by model:**
  - blocking only layers 21–27 leaves 51% (laya), 23% (typed), 90% (multilingual) label-following;
  - blocking all local layers removes it in every model.
- **C2 verdict replicates** (only at the [MASK]s, rescaling reproduces it, temperature beats it). The selected MLPs carry nearly the whole logit scale in typed and multilingual (×0.003–0.007).

### Phase 5 · Training origin of the label shortcut (conditional, ~$12, 1 day)

Same architecture, init, budget and evaluation; only the label regime in the training data varies:
- semantic yes/no labels;
- neutral symbols;
- random label-to-answer mappings;
- a balanced mixture;
- semantic labels correlated with passage words (a planted shortcut).

3 seeds each, if the budget allows. **Measure:** label-following and the [MASK] → label pathway strength. Run only if Phases 1–3 hold. This turns "not inevitable" into a causal statement about training data.

**Frozen design, 2026-10-09 (before any run).**

- **Common setup:** RLCD loss (laya's), ModernBERT-large init, the same 31,419 training items as the retrained controls, 2 epochs, seeds 0/1/2. Only the yes/no (noul) items change between regimes; the multiple-choice items are identical in every run.
- **Regimes:**

  | Regime | Labels | Option order | Option descriptions |
  |---|---|---|---|
  | R0 baseline (the existing 3 RLCD controls) | default "true/false" | fixed | default |
  | R1 semantic | sampled per item from 6 meaningful pairs (true/false, yes/no, correct/incorrect, agree/disagree, valid/invalid, supported/unsupported), always correctly aligned | fixed | default |
  | R2 neutral | sampled from meaningless symbol pairs | fixed | default |
  | R3 random mapping | meaningful pairs, swapped on a random 50% of items, so the label word carries no information | fixed | default |
  | R4 mixture | each item drawn from R1, R2 or R3 at random | fixed | default |
  | R5 planted shortcut | meaningful pairs, correctly aligned | random | identical neutral text ("this option"), so the label word is the only cue to which option means "holds" |

- **Primary metric:** yes/no label-following on BoolQ (main and held-out), RTE and MultiRC. Also reported: true/false and correct/incorrect swaps, a symbol control, normal-label accuracy, and label-following with the guard on (does the shortcut use the same direct [MASK] → label read?). Every model is also evaluated with neutral descriptions.
- **Prediction:** R5 > R1 > R4 > R3 ≈ R2 ≈ R0.
- **Stated limit:** this tests whether label semantics in the training data can create the shortcut; it does not reconstruct laya's own training set.

**Outcome, 2026-10-09.**

- **Only R5 (planted) creates the shortcut:** yes/no label-following 88.8% ± 0.3 across seeds vs 4.4% for R0 (p = 0.002).
- **The other regimes match baseline:** R1, R2, R3 and R4 all give 1.4–5.8%.
- **The prediction "R1 > R0" was wrong:** meaningful labels alone do not create the shortcut.
- **The guard's effect on R5 differs by seed:** 6–21%, 26–31% and 51–95% remain. Same behaviour, different mechanisms.
- **Limit:** laya's real training data are unknown.
- **Operational note:** 18 jobs exceeded the workspace's 10-GPU concurrency limit, so 8 queued (no extra cost). Future launches are capped at 10.

### Phase 6 · Literature, done properly (no compute, 1 day)

Read in full and summarize the exact claims and setups of:
- Set-Based Prompting (2406.06581): check whether their ablation really found the attention mask more important than positions;
- Anchored Answers (ACL Findings 2025);
- Label Words are Anchors (EMNLP 2023);
- Wen et al. (2609.37737);
- Confidence Regulation Neurons (NeurIPS 2024);
- Pezeshkpour & Hruschka (NAACL Findings 2024);
- laya's model card.

Then search specifically for encoder / cross-encoder label-shortcut and option-interaction mechanisms. Record what was searched, so the novelty claims state their own limits.

**Outcome, 2026-10-09.** Notes and the search log are in `LITERATURE_NOTES.md`.

- **Set-Based Prompting contrast** confirmed from the paper itself: the mask mattered more than positions, but in decoders and on only 20 MMLU questions. Reworded accordingly.
- **Entropy neurons** are explicitly not a temperature; laya's MLPs are. This is now framed as a contrast.
- **Model card** confirms the noul label warning.
- **Cross-seed variation** of circuits is known (Toy Model of Universality); we claim only its consequence for fixes.
- **Wen et al.** was read at abstract level only; read it in full before submission.

### Phase 7 · Reproducible release (~0.5 day)

- **Environment:** laya 0.4.1, transformers 5.19.0, torch 2.14.1, nnsight 0.7.0; pinned model revisions from `laya.PINNED_REVISIONS`.
- **Data and results:** export data split files and raw aggregate results from the Modal volume into the repo (or a dataset repo).
- **Scripts:** one script per figure and table; keep the fidelity checks (3.8e-6 logit match, fp32 confirmation) in the README.

## 4. Statistics

- Item-level effects: bootstrap 95% CIs.
- Seed-level effects (C5, Phase 5): report between-seed SD and effect size, with seeds as the unit of analysis.
- Component screens: control the false discovery rate (Benjamini-Hochberg) when naming heads or MLPs.
- Every intervention is reported together with its accuracy cost and a random or matched control.

## 5. Budget and schedule

| Phase | Compute | API | Time |
|---|---|---|---|
| 1 C1 factorial | ~$2 | – | 0.5 day |
| 2 C3 causal account | ~$4 | ~$3 | 1.5 days |
| 3 C2 regulator test | ~$1 | – | 0.5 day |
| 4 Cross-model | ~$3 | – | 0.5 day |
| 5 Training origin (conditional) | ~$12 | – | 1 day |
| 6 Literature | – | – | 1 day |
| 7 Release + rewrite | – | – | 1 day |
| **Total** | **~$22** | **~$3** | **~6 days** |

Spent so far: $12.23 of the original $100 cap. This plan fits in the remainder.

## 6. What will not be done

- **No new unrelated tracks.**
- **No more jailbreak work** beyond keeping C4 as a short section.
- **No SAE feature labelling** unless it is redone blind and scored on held-out data.
- **Claim-only edits.** No claim is upgraded without its control; claims that fail their gate are downgraded in the text.

**Phase 7 outcome, 2026-10-09.** Release in `release/`:

- **Raw results:** JSON plus small pickles. SAE weights stay on the Modal volume.
- **Data:** splits, with jailbreak and injection texts replaced by hashed manifests.
- **Report:** `release/scripts/build_report.py` rebuilds the report (HTML and PDF) from `release/results` alone, so every number in it is computed rather than typed.
- **README:** environment, pinned revisions and per-task reproduction commands.
- **Doc:** reordered with C3 as the headline and retitled "When Labels Override Evidence".
- **Root copies:** `How_Laya_Chooses*.html/pdf` in the repo root now mirror the new report.

---

# Phase 8 · Response to the second review (frozen 2026-10-09, before any run)

**Primary claim** (stays C3). In laya, semantic answer labels can dominate passage evidence through a direct label → answer pathway. A targeted intervention suppresses it, and controlled retraining shows that a particular training information structure *can* induce it. This does not establish laya's actual training history.

## Metric definitions (used everywhere from here)

- **Paired transition.** Each item is scored with normal labels and with swapped label words. Cells: (normal correct, swapped correct), (correct, wrong), (wrong, correct), (wrong, wrong). Counts are reported for every cell.
- **Label-following.** P(swapped wrong | normal correct) = n(correct → wrong) / n(normal correct). The denominator n is always reported.
- **Swap gap.** acc(normal) − acc(swapped), over all items.
- **Recovery** (patching or restoration). Per item, r = (y_patched − y_corrupt) / (y_clean − y_corrupt), where y is the log-odds of the "holds" option. Items with |y_clean − y_corrupt| < 0.5 are excluded and the exclusion count is reported. Reported as the median over items, with a 95% bootstrap CI (2,000 resamples over items).
- **Seed-level tests.** The unit is the trained model (one seed). Welch t-test plus an exact two-sided permutation test on seed means; per-seed values are always listed.
- **Item-level uncertainty.** Bootstrap over items, reported separately from seed-level uncertainty.

## P1 · Confirmatory label–evidence conflict benchmark (data frozen before any guard is run)

- **Items.**
  - 300 MultiRC validation items, none sharing a paragraph with `noul_confirm`.
  - 300 GLUE QNLI validation items ("Does the sentence answer the question?").
  - All human-written, never used before.
- **Evidence factor.**
  - real passage;
  - contradicting edit (gpt-5.6-terra minimal edit; kept only if gpt-5.6-luna answers the flipped gold);
  - blanked passage.
- **Label factor.** Normal vs swapped label words.
- **Wordings.**
  - Old: yes/no, true/false.
  - New, never used before: right/wrong, accurate/inaccurate, confirmed/unconfirmed.
  - New multi-token: "yes, that is right" / "no, that is wrong", "it is the case" / "it is not the case".
  - Neutral: ◇/△ and A/B.
- **Freeze.** The SHA-256 of the frozen file is recorded below before evaluation.

## P2 · Matched intervention controls (on the P1 benchmark; laya, typed, multilingual)

- **Original guard:** [MASK] → own and other label tokens, all local layers.
- **Extended guard (still exploratory):** the original guard plus option text → label tokens, local layers. First test on data it was not designed on.
- **Random-edge control:** [MASK] → the same number of random non-label tokens inside the option spans, local layers.
- **Description control:** [MASK] → the same number of description tokens, local layers.
- **Evidence-path block:** question tokens → passage, all layers.
- **Sham:** [MASK] → [CLS] in local layers.
- **Measures:** paired transitions, label-following, accuracy, ECE/Brier, multi-token wordings, and jailbreak and injection AUROC.

## P3 · Training-origin dose response

Shuffled noul option order in every arm; same data, loss (RLCD), init and budget.

| Arm | Descriptions | Label word | Seeds |
|---|---|---|---|
| D0 uninformative | informative | random (correct p = 0.5) | 0–2 |
| D1 redundant | informative | always correct | 0–2 |
| D2 weak | neutral | correct p = 0.75 | 0–2 |
| D3 strong but insufficient | neutral | correct p = 0.9 | 0–2 |
| Planted (sole cue) | neutral | always correct | 0–2 done; add 3–5 |
| Baseline (default labels, fixed order) | default | default | 0–2 done; add 3–5 |

- **Primary test:** planted vs baseline with 6 seeds each, exact permutation test.
- **Prediction:** label-following increases with label reliability when descriptions are neutral; D1 stays low.

## P4 · Position vs interaction with binding preserved (10 models)

- **Position permutation:** canonical token order with option position ids taken from the reversed layout.
- **Token permutation:** reversed token order with the canonical layout's position ids.
- **Question answered:** does the decision follow positions (agree with the true reversed run) or token order?
- **Replacement types for the competition metric M:** neutral filler, "-" filler, and a length-matched distractor taken from another item's option.

## P5 · External architecture (RoBERTa-large option-marker models)

- **Training:** laya's trainer with `FacebookAI/roberta-large` as the encoder; planted vs baseline, 3 seeds each.
- **Tests:**
  - whether the shortcut emerges;
  - whether a [MASK] → label block in all layers removes it;
  - whether blocking question → passage drops accuracy.
- **Mask scope:** RoBERTa has no local/global split, so its masks apply to all layers.

## P6 · Audits and C2

- **Audit sheets for human checking** (CSV, kept out of git): 60 random contradicting edits, 60 GPT-written C3 items, 60 jailbreak rewrites.
- **Style test:** 150 benign prompts written in jailbreak style (persona or role-play, no rule override) vs 150 plain benign prompts; compare laya's flag rates.
- **C2:** global temperature fitted on the exploratory MC split and applied to the confirmatory split, compared with late-MLP ablation.

## Not done

- No new unrelated analyses.
- C1, C2, C4 and C5 remain framed as secondary, at their current evidence level.

**Amendment, 2026-10-09 (before any evaluation on the benchmark).** The MultiRC half of P1 is infeasible: MultiRC validation has only about 80 distinct paragraphs, and all of them are already used by `noul_confirm`. It is replaced by 300 MNLI validation-matched pairs (entailment vs contradiction, neutral dropped; "Does the text imply that …?"). These are human-written and have never been used. The draft file without MNLI was never evaluated and is deleted. QNLI items are unchanged.

**P1 benchmark frozen, 2026-10-09, before any evaluation.** `data/conflict_bench.jsonl`:

- 600 items (MNLI 300, QNLI 300);
- 519 with a verified contradicting edit;
- gold "holds" rate 0.543;
- SHA-256 `cea5547fffab4c62e77bf244245c1b63db0e94d2a206fee8c2a82229ceafd810`.

**Phase 8 outcomes so far (2026-10-09).**

- **P1/P2: frozen benchmark, laya, fp32.**
  - **yes/no label-following:** 96.7% (499 of 516 correct-normal items flip; 95% CI 95.2–98.1). Transitions: correct→correct 17, correct→wrong 499, wrong→correct 70, wrong→wrong 14.
  - **Never-used wordings:** right/wrong 80.4%, accurate/inaccurate 78.2%, confirmed/unconfirmed 85.2%, "yes, that is right" 96.9%, "it is the case" 93.8%. Symbols 4.1%, A/B 4.9%.
  - **Guard:** yes/no 2.6%, right/wrong 3.0%, multi_right 28.8%. Accuracy 0.860 → 0.842.
  - **Extended guard (first confirmatory test):** multi_right 7.6%, others 1.6–5.2%. Accuracy 0.852.
  - **Matched controls:** random edges, description edges and sham leave label-following at 92–99%. The evidence-path block drops accuracy to 0.645.
  - **Contradicting passages:** 97.5% → 4.5% with the guard.
  - **Blank passages:** the label pull (holds 33% → 95% when labels swap) disappears with the guard (5% → 11%).
  - **Safety cost:** jailbreak AUROC 0.940 → 0.917, injection 0.896 → 0.872.
  - **Replication:** typed and multilingual replicate (guard 1.6–5.6%, controls no effect).
- **Definitions.** Restoring the question tokens at layer 16 gives median recovery 0.907 [0.850, 0.962] (n = 275; 25 of 300 excluded for |gap| < 0.5). C2 with a held-out temperature: T = 2.61 fitted on the exploratory split gives confirm ECE 0.055 vs 0.080 for MLP ablation.
- **P4.** The position-permuted run equals the reversed run in 100% of flips in all 10 models. This holds by construction: a bidirectional encoder's output depends only on (token, position) pairs, so order effects are positional by definition. The substantive result remains the factorial one (option interaction carries 0–8 points). The competition metric is robust to the replacement type (laya: filler 0.388, dash 0.400, distractor 0.442).
- **P6.** On 150 GPT-written benign pairs, the same request in jailbreak style is flagged 54.0% vs 12.7% plain (injection question: 42.7% vs 2.7%). Audit sheets for human checking are in `audit/` (git-ignored).
- **P3 dose response (frozen benchmark, yes/no label-following per seed).**
  - Baseline (6 seeds): 0.2, 0.7, 18.1, 9.3, 54.9, 4.7% (mean 14.7%).
  - D0: 1.3, 0.0, 0.9%. D1: 2.8, 31.2, 98.5%. D2: 91.9, 41.1, 90.9%. D3: 95.3, 92.0, 85.8%.
  - Planted (6 seeds): 81.2, 86.3, 84.2, 94.7, 92.3, 95.0%.
  - Primary test, planted vs baseline: exact two-sided permutation p = 0.0022 (the minimum possible, 2/924); Welch p = 0.00018.
  - **Prediction partly wrong:** label-following does rise with label reliability under neutral descriptions, but D1 does not stay low (one seed of three at 98.5%).
- **Exploratory, post hoc: the same guard edges blocked in all layers.**
  - Planted s0–s5, local guard → all-layer guard: 33.3→2.0, 78.1→26.8, 16.1→5.3, 65.5→7.1, 28.8→3.6, 71.1→12.7%. Extended all-layer guard: 0.3–19.1%.
  - laya: 2.6% (local), 2.6% (all layers) and 0.6% (extended, all layers).
  - Two seeds lose 20+ accuracy points.
  - Revised reading: the same edge class, with the carrying layers differing by seed. This replaces "different mechanisms".
- **P5 RoBERTa-large (all-layer [MASK]→label block; fidelity check 0.0).**
  - Planted s0/s1/s2: 97.4 / 99.7 / 84.9% → 1.5 / 5.8 / 1.2%.
  - Baseline s0/s1/s2: 0.3 / 42.1 / 10.8%.
  - Did not learn (BoolQ ≈ 0.63–0.64, majority-class bench accuracy): baseline s0, baseline s2 and planted s1.
  - This is a partial replication; no learning-rate retune was done.

# Phase 9 · Follow-ups to Phase 8 (frozen 2026-10-10, before any run)

## Q1 · All-layer guard, confirmatory (fresh benchmark)

- **Data:** `conflict_bench2.jsonl`.
  - 300 MNLI validation-matched + 300 QNLI validation items, drawn with seed 22.
  - Excludes every id in `conflict_bench.jsonl`.
  - Contradicting edits are made the same way as before.
  - Frozen and hashed before any evaluation.
- **Models:** laya, typed, multilingual, and planted s0–s5.
- **Interventions:**
  - `none`;
  - `guard` (local layers);
  - `guard_all_layers`;
  - `guard_extended_all_layers`;
  - `random_edges_all_layers`, a new matched control: the same number of random option-text edges, blocked in every layer.
- **Prediction:**
  - With `guard_all_layers`, yes/no label-following is ≤ 10% in laya, typed and multilingual, and in at least 5 of 6 planted seeds.
  - `random_edges_all_layers` stays within 10 points of `none` in every model.

## Q2 · D1 (label redundant with informative text): seeds 3–5

- **Runs:** same recipe, evaluated on `conflict_bench.jsonl` as in P3.
- **Report:** all 6 seeds individually, plus exact permutation tests of D1 vs baseline and D1 vs planted (6 vs 6 each).
- **Prediction:** no directional prediction. The question is how often a redundant label gets used.

## Q3 · RoBERTa-large rerun with a lower encoder learning rate

- **Change:** `encoder_lr` = 1e-5 instead of 2.5e-5. laya's trainer has no warmup, and 2.5e-5 is tuned for ModernBERT. Everything else is unchanged.
- **Runs:** planted and baseline, seeds 0–2.
- **"Learned" criterion, fixed now:** BoolQ held-out accuracy ≥ 0.70. Every run is reported, including those that fail the criterion.
- **Prediction:** among runs that learned, planted label-following is higher than baseline, and the all-layer [MASK]→label block reduces planted runs to ≤ 10%.

**Q1 benchmark frozen, 2026-10-10, before any evaluation.** `data/conflict_bench2.jsonl`:

- 600 items (MNLI 300, QNLI 300), with no overlap with `conflict_bench.jsonl` (asserted);
- 524 with a verified contradicting edit;
- gold "holds" rate 0.478;
- SHA-256 `05c0217cbd11f7205f7bba88f15c1ae242e4155019a8d3f06e7f1b90de87fa48`.

**Phase 9 outcomes (2026-10-10).**

- **Q1 (fresh benchmark, yes/no label-following, all-layer guard).**
  - laya 4.8%, typed 2.3%, multilingual 0.6%.
  - Planted s0–s5: 3.6, 32.7, 5.8, 10.2, 4.7, 13.7%.
  - Only 3 of 6 planted seeds are ≤ 10%. **Prediction failed.**
  - The random-edge all-layer control stays within 10 points of no intervention in 9 of 9 models: the effect is specific.
  - Extended guard, all layers: ≤ 10% in 5 of 6 seeds; s1 stays at 23.6%.
  - s1 and s5 lose 24–30 accuracy points.
- **Q2 (D1, 6 seeds).**
  - Per seed: 2.8, 31.2, 98.5, 94.7, 92.1, 81.8% (mean 66.9%).
  - Exact permutation tests: D1 vs baseline p = 0.024; D1 vs planted p = 0.34.
  - A redundant label is relied on in 4 of 6 seeds.
- **Q3 (RoBERTa, encoder_lr 1e-5).**
  - All 6 runs learned (BoolQ 0.773–0.820).
  - Label-following: planted 97.8, 82.3, 92.4% vs baseline 66.8, 4.6, 6.7%. Every planted run exceeds every baseline run (exact p = 0.10, the minimum for 3 vs 3).
  - All-layer block on planted runs: 6.0, 2.8, 37.6%, so 2 of 3 are ≤ 10%. **Prediction partly met.**

**Audit of GPT-made data (2026-10-11).** Claude judged random samples; it is independent of the GPT generators, and this is not a human audit. Filled copies are in `audit/claude_filled/` (git-ignored), with flagged rows marked for human review.

| Sheet | Result |
|---|---|
| Benchmark contradicting edits | Flip the answer 56/60; minimal 58/60 |
| BoolQ edits | Flip 57/60; minimal 59/60 |
| GPT C3 items | Gold correct 60/60; answer determined by the passage 57/60 |
| Style pairs | Same request 141/150; no override 150/150 |
| Jailbreak rewrites | Override removed 53/60; *only* the override removed 21/60 |

- **Consequence for C4.** The headline "laya flags 82% after only the override is removed" is withdrawn.
- **Rescore of the same 145 rewrites** (`release/results/phase9/jb_audit_rescore.json`; reproduces 82.1% overall):
  - **Clean edits (override removed, nothing else changed):** 17/21 flagged (81%, Wilson 95% CI 60–92%).
  - **Edits that also cut the persona or setup:** 29/32 flagged.
  - **Edits that kept the override:** 7/7 flagged.
- **Reading.** Laya responds to the residual role-play framing or harmful request, not to the override instruction itself. This is exploratory; the style test remains the clean test.

# Phase 9b · Closing gaps (frozen 2026-10-12, before any run)

## Q4 · RoBERTa-large, seeds 3–5

- **Runs:** planted and baseline, same recipe as Q3 (encoder_lr 1e-5). Together with Q3 this gives 6 vs 6 seeds.
- **Learned criterion:** unchanged (BoolQ held-out ≥ 0.70); every run is reported.
- **Prediction:** planted label-following exceeds baseline. Exact two-sided permutation p < 0.05 over the 6 vs 6 seeds that learned.

## Q5 · C4 override removal, strict version

- **Edit:** gpt-5.6-terra edits the same 145 flagged jailbreaks, deleting only override spans (rules: delete, never rewrite).
- **Acceptance filter, fixed now:**
  - word-level deletion-only, checked in code;
  - gpt-5.6-luna judges that no override remains.
- **Report:** laya's flag rate on the accepted edits, with a Wilson 95% CI.
- **Prediction:** none. This replaces the withdrawn 82% with a cleaner estimate.

## Q6 · Label-direction ablation (exploratory)

- **Direction:** for each layer, the mean residual at the "holds" [MASK] with normal yes/no labels minus with swapped labels. It is fitted on `conflict_bench.jsonl` (600 items).
- **Ablation:** project that direction out at every [MASK] position in every layer, evaluated on `conflict_bench2.jsonl`.
- **Control:** random unit directions per layer (seed 0).
- **Combined condition:** direction ablation plus the extended all-layer guard.
- **Models:** laya, plus planted s1 and s5 (the seeds where the attention guard was not a clean fix).
- **Success criterion, fixed now:** yes/no label-following ≤ 10% with an accuracy drop ≤ 5 points.

## Q7 · Audit of the second benchmark's contradicting edits

- 60 random items (seed 0), judged the same way as the first-benchmark sheet.
