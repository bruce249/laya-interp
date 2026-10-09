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
