# How a Non-Autoregressive Decision Model Chooses: Option Competition and Confidence in Laya

Plan v3.1, **final and frozen**. **Everything runs on Modal, including data generation, LLM calls, plots, and the report. Nothing runs or is stored on the laptop.** **Budget: $100 hard cap, ~$62 planned.** Target: results in **2 to 3 days**.

No more redesigns. Changes after this point happen only when Phase 0 shows a fact in this plan is wrong.

Working title: *How a Non-Autoregressive Transformer Makes Decisions: Causal Circuits for Option Competition and Confidence.*

## 0. What changed from v2

| Change | Reason |
|---|---|
| Headline is no longer "where jailbreak forms" | Too close to existing work. A Sept 2026 paper (arXiv 2609.37737) already shows prompt-injection compliance is decodable early but causally decided late. Generic safety directions are also well studied. |
| RQ5 "questions interfere in one pass" removed | The public input layout holds **one question per sequence**: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`. Separate questions cannot attend to each other. |
| New core: **option competition** | All options of a question share one sequence, so they can interact. Reversing option order changes laya's ARC answer on ~34% of items (smolaya model card). That is a large, unexplained behavior. |
| New core: **choice vs confidence** | Test whether the mechanism picking the winner is separable from the one setting its probability. Framed carefully: laya's raw outputs are overconfident (model card: ECE 0.466 raw, 0.081 after per-(type, k) temperature refit). |
| New core: **label-following failure** | The model card notes noul answers sometimes follow the option labels instead of the state. A documented failure mode we can explain mechanistically. |
| Training-objective question made cheap | Instead of training a CE baseline, compare against **smolaya**, a public variant trained with CE + distillation and random option permutation, which is less order-sensitive (~22% flips). Confounds are noted. A full CE retrain is a stretch goal. |
| Jailbreak becomes a generalization test | Checks whether the competition and confidence mechanisms hold on a safety behavior, and compares an encoder to the LLM injection result. |
| Scope cut | intent/frustration dropped as separate studies. Multilingual reduced to one test. |

Methodology kept from v2: held-out template families, natural counterfactuals, attribution → exact → path patching, mean and resample ablation, probe control tasks, steering specificity checks, bootstrap CIs, fp32 confirmation.

**v3.1 additions (from the go/no-go review):**

1. **Parallel-layout control for C1.** Same weights, but options cannot attend to each other. If competition comes from option-option attention, the interaction matrix should collapse to ~0 and order flips should drop. This is a cleaner control than smolaya, which differs in depth and training too.
2. **Geometric definition of choice vs confidence.** Choice is the direction/ranking of the centered logit vector; confidence is its magnitude. This replaces the top-vs-runner-up margin as the formal criterion, since the margin jumps when the winner changes.
3. **Label-semantic control for C3.** True/False, A/B, and neutral-symbol labels, to separate semantic label bias from generic symbol or position bias.
4. **Narrower confidence claim.** We do not claim to discover confidence representations. We test whether confidence is mechanistically separable from winner selection in a purpose-built decision transformer.

## 1. Central claims (falsifiable)

**C1. Option competition.** Laya does not score options independently. An option's logit depends causally on the other options' content, and specific heads carry that interaction. Order sensitivity comes from a nameable mechanism (position, competition, or both).

**C1-Control.** Blocking option-option attention (parallel layout, same weights) drives the interaction matrix to ~0 and reduces order sensitivity.

**C2. Choice and confidence are partly separable.** Some interventions rotate the centered logit vector (change which option wins) while keeping its norm roughly fixed. Others change its norm (confidence) while keeping the argmax fixed. These map to different components. The claim is about separability in this architecture, not the existence of confidence representations.

**C3. Label vs state competition.** Laya contains a measurable causal competition between label-token evidence and state evidence. When noul answers ignore the state, a path from label tokens to `[MASK]` wins over the state-reading path, and the effect depends on label semantics, not only on symbols or position.

**C4 (generalization).** C1 to C3 hold on jailbreak detection. On an encoder decision model, safety evidence is or is not "decodable early, decisive late" like the LLM result.

**C5 (training contrast).** Permutation-trained smolaya has weaker option competition or a different position mechanism than laya.

A null result on any claim is reported as a null, e.g. "options are scored nearly independently; order effects are purely positional." Do not force a positive C1.

**Priority:** C1 + C1-Control > C2 > C3 > C4 > C5. Causal depth beats breadth.

**The target story:** state retrieval → option interaction → competition → winner selection → confidence, with named causal components at each stage.

## 2. Model facts

- **laya (English):** ModernBERT-large backbone (~28 layers, d=1024, 16 heads; local 128-token window and global attention alternate, global every 3rd layer; RoPE; GeGLU) → 2 pre-norm transformer layers → option-marker scorer (LayerNorm → Linear → GELU → …) → one logit per option. Separate act/escalate head.
- **Layout:** question first, then `[MASK] opt` pairs (up to 48 tokens per option, `head_max_len` total), then state. 512 tokens. State is truncated on the right.
- **Output:** raw logits ÷ temperature for (qtype, k) → softmax. choice = argmax; score = Σ i·pᵢ; noul = p[1].
- **Geometry consequence:** the `[MASK]`s sit near each other and the question, so local layers can mix options. Most of the state is more than 128 tokens from the `[MASK]`s, so state evidence must largely pass through global layers.
- **Comparison models:** `anon767tom/smolaya` (encoder truncated to layers 0 to 19, CE + KL-to-laya, random option order), `laya-multilingual` (mmBERT-base).

Phase 0 confirms all of this against the installed package version before anything else runs.

## 3. Metrics

Probabilities are a trap here: removing an option renormalizes the softmax and changes every pᵢ for trivial reasons. So the primary quantities are **raw logits**.

- **Option logit** sᵢ (before temperature).
- **Centered logits:** zᵢ = sᵢ − mean(s), since a shift shared by all options does nothing after softmax.
- **Geometric decomposition (formal criterion for C2):** write z = ‖z‖ · û.
  - **Choice geometry** is the unit direction û (it fixes the ranking and argmax).
  - **Confidence geometry** is the norm ‖z‖ (it sets how sharp the softmax is).
  - For an intervention z → z′: **rotation** = angle between û and û′; **scale change** = log(‖z′‖ / ‖z‖).
  - A **choice intervention** has large rotation with |scale change| small. A **confidence intervention** has |scale change| large, argmax unchanged, and small rotation.
- **Also reported:** argmax, margin s_top − s_second, entropy H(p), max p, each with raw logits and with the shipped temperature.
- **noul:** s_true − s_false.
- **Recovery:** `(patched − corrupt) / (clean − corrupt)` on the relevant metric.
- **Statistics:** bootstrap 95% CIs; 3 seeds for probes; everything also reported on the held-out family; headline numbers re-run in fp32.

## 4. Data

All cheap and public. Built while Phase 0 runs.

| Set | Use | Size |
|---|---|---|
| ARC, CommonsenseQA, QASC (cast to laya `choice`) | C1, C2, C5: order and competition | 2k items, all with 3 to 5 options |
| Noul items with swappable label wording + states that contradict the label cue | C3 | 1k, LLM-generated, model-filtered |
| Jailbreak minimal pairs (6 template families, one held out) | C4 | ≥400 kept from ~2k candidates |
| Jailbreak natural sets (JailbreakBench, benign ToxicChat/Alpaca) + natural counterfactuals | C4 | 3k + 200 |
| Injection set (`deepset/prompt-injections`) | C4, comparison to LLM paper | 1k |
| Hindi translations of 200 jailbreak pairs | multilingual check | 200 |

Option-set rules: options tokenized to equal length when doing patching across items, padded with neutral filler. Licenses in `data/README.md`. Baseline accuracy, ECE (raw and tempered), and order-flip rate measured per set.

## 5. Experiments

### Phase 0: Access and verify (hours 0 to 3, blocking)

1. `inspect` (CPU): package version, source, the `nn.Module`, `build_sequence`, marker tokens, temperature table, option slot limit, global layer indices, head structure. **Confirm one question per sequence** and how multi-question calls are batched. **Check whether the package ships a parallel option layout** (a mode or checkpoint where options don't attend to each other) and record exactly how it works.
2. `model.py` from the package and `model_fallback.py` from the raw checkpoint, built together. Eager attention.
3. `verify`: 200 inputs, custom forward matches `predict` within 1e-4 (fp32), for laya and smolaya.
4. Throughput and bf16/fp32 agreement. Re-cost section 8.

### Track 1: Option competition (C1), the main study

1.1 **Behavior map.**
- Order: for each item, all cyclic shifts and the reversal. Measure flip rate and how each option's logit varies with slot.
- Split order effects into **position** and **neighbor** parts: regress sᵢ on its slot index and on the identity of adjacent options.

1.2 **Interaction matrix.** For option i, replace option j with length-matched neutral filler (keeps positions fixed) and measure Δsᵢ (centered). This gives M_ij per item.
- Compare to removal of j (positions change) to separate content effects from position effects.
- Questions: Is M symmetric? Do strong distractors suppress the correct option? Do semantically similar options compete more (correlate M_ij with embedding similarity)? Does the correct option's margin go up when the strongest distractor is neutralized?
- If M ≈ 0 everywhere: options are scored independently, and order effects are pure position. Report and move on.

1.3 **Where competition happens.**
- Attention from each `[MASK]` to its own option, other options, other `[MASK]`s, question, and state, per head, local vs global layers, backbone vs head layers.
- Attribution patching, then exact patching: patch option j's tokens from a clean item into a corrupt one and track the effect on sᵢ across layers and positions.
- Path patching from option j tokens → candidate heads → `[MASK]_i`. Name **competition heads** (one option suppresses another) and **position heads** (attend by slot, not content).

1.4 **Causal test.** Ablate the competition heads (mean and resample).
- Prediction: M shrinks toward 0, order flip rate drops, accuracy changes by a measurable amount.
- Ablate position heads instead: order flips should drop while M stays.

1.5 **Parallel-layout control (C1-Control).** Run the 1.1 order map and 1.2 interaction matrices under two parallel conditions:
- **Masked parallel (always run):** same weights and same token sequence, with an attention mask so that each option span and its `[MASK]` cannot attend to other options' spans or `[MASK]`s, and question/state tokens cannot attend to any option (this closes the indirect path option j → state → option i). Each option then sees only the question, the state, and itself.
- **Shipped parallel (only if Phase 0 finds one):** the package's own parallel layout or checkpoint.

Predictions if competition is attention-mediated: M_parallel ≈ 0 with tight CIs, and order flip rate drops toward the pure-position level. Then **layer-wise masking**: block option-option attention only in a band of layers (early, middle, late, head layers; local vs global) to see which layers carry the interaction. This cross-checks the heads found in 1.3.

Caveat: masked parallel is off-distribution for weights trained on the sequential layout, so check accuracy under masking. If it collapses, report the masked results as an upper bound on dependence, not a clean ablation, and lean on 1.4 head ablations and the shipped parallel layout if available.

### Track 2: Choice vs confidence (C2)

2.1 **Decompose the logit vector** at every layer via logit lens and tuned lens at each `[MASK]`: where does the direction û stabilize, and where does the norm ‖z‖ grow? These can happen at different depths. The expected positive result is that choice stabilizes earlier than confidence.

2.2 **Component screen.** For every head and MLP (reusing Track 1 patching runs where possible), measure under ablation the mean rotation (angle between û and û′) and mean scale change log(‖z′‖/‖z‖), plus ΔArgmax rate. Plot every component on a 2D map: rotation vs scale change.

2.3 **Separability test (geometric criterion).** The claim holds if there exist component sets A and B with:
- **A ablated:** large rotation, argmax changes on many items, |scale change| small.
- **B ablated:** argmax unchanged on ≥95% of items, small rotation, large |scale change| (entropy moves accordingly).
Also test a direction-level version: find a residual direction at `[MASK]` that predicts ‖z‖ (regress on residuals, orthogonal to directions that predict û) and steer with it. Success means steering changes ‖z‖ along a dose-response curve while the argmax holds.

Report the opposite result plainly if it happens: no component set changes ‖z‖ without rotating û, meaning confidence is a by-product of the choice computation in this model.

2.4 **Does internal confidence track correctness?** AUROC of raw margin for predicting correct answers, before and after ablating set B. Raw and tempered ECE before and after. This tells you if set B carries information or only scale.

2.5 **Overconfidence.** Is laya's overconfidence a uniform scale (one global factor, which temperature fixes) or input-dependent? Check whether the residual norm at `[MASK]` or a specific late component sets the scale.

### Track 3: Label-following failure (C3)

3.1 Build noul items where the label cue points one way and the state points the other. Measure how often the answer follows the label. Run every item under **three label types**:
- **Semantic:** True/False (and Yes/No).
- **Symbolic:** A/B.
- **Neutral:** arbitrary symbols with no meaning (e.g. "◇"/"△" or random tokens of equal length).
Also swap which label sits in which slot. If label-following appears only with semantic labels, it is semantic label bias. If it appears equally with neutral symbols, it is symbol or position bias, which connects to Track 1's position heads. Option-symbol bias is studied in MC LLMs, so the claim here is the causal competition with state evidence, not the bias itself.

3.2 Patch the label tokens vs the state tokens between conflicting and consistent items. Which wins at which layer?

3.3 Path patching: label tokens → `[MASK]` (local, short distance) vs state → `[MASK]` (mostly global). Test the hypothesis that local-layer label reading beats global-layer state retrieval when state evidence is weak or distant.

3.4 Ablate the label path and check whether label-following drops without hurting normal accuracy.

### Track 4: Generalization (C4, C5)

4.1 **Jailbreak and injection.**
- Probes at each layer at state tokens and `[MASK]` (control tasks, held-out family): where is the attack decodable?
- Segment-level activation patching: where does causal control appear?
- Compare to the LLM result: is it also early-decodable and late-decisive in an encoder, or does the decision form earlier because the encoder sees everything at once?
- Do Track 1 and 2 components (competition heads, confidence set B) play the same role here? Ablate them on jailbreak items.
- One diff-of-means jailbreak direction with steering, a specificity check on ARC and benign accuracy, and an ablation. This is a supporting result, not a headline.

4.2 **laya vs smolaya (C5).** Rerun 1.1, 1.2, and the 1.3 attention summary on smolaya. Does permutation training shrink M, remove position heads, or both? Caveats: smolaya also differs in depth and objective, so this is suggestive, not causal. The causal control for C1 is 1.5; this is supporting evidence about training.

4.3 **Multilingual (one test).** Does the English jailbreak direction, refit on mmBERT English activations, steer Hindi inputs? Report as a single result.

**Stretch (only if under budget after Tracks 1 to 3):** fine-tune a CE-only control from the same ModernBERT-large init on the same option-marker format with fixed option order, then repeat 1.2 and 2.3. This is the clean training-objective comparison.

### SAE (cut to one, targeted)

One TopK SAE (16k latents) on the residual at `[MASK]` positions only, at the layer where Track 2 finds the margin forming. ~5M `[MASK]` tokens, collected from the cached datasets plus additional ARC-style items. Use: check whether confidence (set B) and choice correspond to distinct features. Run only if the budget table allows after Phase 0.

## 6. Infrastructure

```
interp/
  model.py, model_fallback.py  # laya, smolaya, laya-multilingual; eager attention
  inputs.py      # build_sequence clone; MASK, option, question, state spans
  hooks.py       # cache, patch, ablate, steer
  metrics.py     # centered logits, margin, entropy, M_ij, recovery, CIs
  data.py        # MC conversion, noul conflict items, jailbreak pairs, held-out split
  competition.py # order shifts, neutral-filler replacement, interaction matrices
  masks.py       # parallel-layout attention masks, full and layer-band
  confidence.py  # rotation/scale decomposition, component screen, norm direction, AUROC, ECE
  attribution.py # logit lens, tuned lens, attention summaries
  patching.py    # attribution, activation, path patching; mean + resample
  probes.py      # probes with controls, diff-of-means, steering
  sae.py         # MASK-only SAE
  llm.py         # OpenAI-compatible client (OpenAI by default, Grok by base URL swap), batching, retries, caching
  report.py      # builds the HTML report from results on the Volume
  cost.py, plots.py
modal_interp.py
```

### 6.1 Hardware (all Modal)

| Resource | Spec | Used for |
|---|---|---|
| CPU containers | 4 to 8 cores, 32 GB RAM | `inspect`, dataset building, LLM API calls, probes and tuned lens on cached activations, plots, report |
| **H100 80 GB** | 4 to 8 concurrent, one GPU per container | Forward passes, patching, ablations, attention masking, steering, SAE |
| L4 24 GB (optional) | Same fan-out | Plain sweeps, only if Phase 0 shows it is cheaper per sequence than H100 |
| Volume `laya-weights` | ~5 GB | laya, smolaya, laya-multilingual weights (HF cache) |
| Volume `laya-interp` | ~30 to 50 GB | `/data/`, `/acts/`, `/results/<run_id>/`, `/report/` |

Laya is 421M parameters (~1.7 GB in fp32), so a single H100 handles large batches. No multi-GPU jobs are needed. Before day 1, check the workspace GPU concurrency limit allows at least 4 H100s at once.

Image: `debian_slim` (Python 3.12) + `laya`, `torch`, `transformers`, `nnsight`, `einops`, `datasets`, `scikit-learn`, `matplotlib`, `pandas`, `openai`.

### 6.2 LLM APIs (called from Modal CPU functions)

**Default provider: OpenAI.** No Claude models are used.

| Task | Model | Notes |
|---|---|---|
| Template paraphrases, noul label-conflict items (3 label types), Hindi translation + back-translation, SAE feature labels | GPT-5.6 Luna | Bulk work |
| Natural counterfactuals (smallest edit that defuses a real jailbreak) | GPT-5.6 Terra | Needs careful minimal edits |

- Use the **Batch API** for all bulk generation (50% off, results within 24 hours). Only Phase 1's first small sample runs synchronously, so data checks can start right away.
- **Backup: xAI Grok** through the same OpenAI-compatible client, by changing the base URL and key. Check current model names on docs.x.ai first.
- **Refusals:** jailbreak data starts from JailbreakBench; the LLM is only asked for defusing edits and benign paraphrases, never for new jailbreaks. If a provider still refuses a step, run an open-weights model on a Modal GPU for that step (adds ~$1 to $2).
- Every LLM output is cached on the Volume with its prompt hash, so reruns cost nothing.
- Expected cost: a few million tokens in total, roughly **$2 to $5**. Verify current per-token prices when creating the key.

### 6.3 Secrets

- `OPENAI_API_KEY` (or `XAI_API_KEY` for Grok) as a Modal Secret.
- `HF_TOKEN` as a Modal Secret (optional; all models and datasets are public, but it avoids download rate limits).

### 6.4 No-laptop workflow

1. **One-time deploy from the browser:** open a Modal notebook (or GitHub Codespaces), clone the repo, set the secrets in the Modal dashboard, and run `modal deploy modal_interp.py`.
2. **Launch experiments from the browser:** each experiment is a deployed function, started from the Modal dashboard or a small web endpoint (`/run?exp=interaction_matrix`). A deployed `orchestrator` function can run a whole track in order.
3. **Read results in the browser:** `report.py` builds an HTML report (figures, tables, CIs, costs) on the Volume, served by a Modal web endpoint. Nothing is downloaded.
4. **Code changes:** edit in Codespaces or the Modal notebook and redeploy. The laptop is only a browser.

### 6.5 Cost discipline

- Model loaded once per container (`@modal.enter`); 4 to 8 large shards per job; activations cached once and reused; `scaledown_window=60`; no long-lived dev servers.
- Modal spending limit set at $100 before starting. Every run's cost goes in `manifest.json`, and a running total in `results/COSTS.md`, shown on the report page.

## 7. Schedule

| When | Work | Checkpoint |
|---|---|---|
| Day 1, hours 0 to 3 | Phase 0; data building | Layout confirmed (one question per sequence); verification passes |
| Day 1, hours 3 to 10 | Baselines; activation cache; 1.1 order map; 1.2 interaction matrices; 1.5 masked parallel; 2.1 lenses | **Key decision:** is M non-zero, and does it collapse under masked parallel? If M ≈ 0, C1 becomes a position story |
| Day 1, overnight | 1.3 patching; 1.5 layer-band masking; 2.2 component screen; 3.1 to 3.2 (all label types) | |
| Day 2 | 1.3 path patching, 1.4 ablations; 2.3 to 2.5; 3.3 to 3.4; 4.2 smolaya | Separability and label-path results |
| Day 3 | 4.1 jailbreak/injection; 4.3 multilingual; optional SAE; held-out replication; fp32 confirmation; write-up | Drop anything that does not replicate |

## 8. Budget

Assumes ~500 seq/s per H100 in bf16 at 512 tokens, ~$4.17/hr all-in. **Rescale after Phase 0.**

| Work | H100 hours | Cost |
|---|---|---|
| Phase 0 | 1 | ~$4 |
| Baselines + activation cache (laya, smolaya) | 1.5 | ~$6 |
| Track 1: order shifts, interaction matrices, patching, ablations | 4 | ~$17 |
| Track 1.5: parallel-layout control (full + layer bands) | 1 | ~$4 |
| Track 2: component screen, steering, ECE | 2 | ~$8 |
| Track 3: label-following, 3 label types | 1.5 | ~$6 |
| Track 4: jailbreak/injection, smolaya, multilingual | 2 | ~$8 |
| fp32 confirmation | 0.5 | ~$2 |
| CPU jobs (~$2) + OpenAI API, Batch (~$2 to $5) | | ~$7 |
| **Planned** | **~13.5** | **~$62** |
| SAE (optional) | 2 | ~$8 |
| Buffer | | ~$30 |
| **Hard cap** | | **$100** |

The CE retrain stretch goal probably costs $20 to $40 depending on data size. Attempt it only if the planned spend comes in under ~$50.

**Overrun cuts, in order:** SAE, multilingual, injection (keep jailbreak), smolaya comparison, Track 3 path patching, layer-band masking (keep full masked parallel). Tracks 1, 1.5, and 2 are protected.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Installed package packs several questions into one sequence after all | Phase 0 checks. If so, the old interference question returns as a side study. |
| Options are scored almost independently (M ≈ 0) | Still a result. The order effect then is a position mechanism, and Track 1 becomes "where position bias enters." |
| Removing options confounds content with position | Neutral-filler replacement keeps positions fixed; removal is reported separately |
| Softmax renormalization creates fake interactions | All interaction metrics use centered raw logits |
| Choice and confidence turn out inseparable | Reported as a finding: in this model, confidence is a by-product of the choice computation |
| Masked parallel is off-distribution and wrecks accuracy | Check accuracy under masking; treat as an upper bound; rely on head ablations and the shipped parallel layout if one exists |
| No shipped parallel layout exists | Masked parallel on the same weights is the control; this was always the primary version |
| Margin unstable when the winner changes | Formal C2 criterion uses rotation and norm of the centered logit vector, not the margin |
| Label effect is just symbol or position bias | Three label types plus slot swaps separate semantic from symbolic effects |
| Confidence effect is just a global scale | 2.5 tests this directly; temperature-corrected metrics reported alongside raw |
| smolaya differs in more than training | Labeled suggestive; CE retrain as stretch |
| Template artifacts | Held-out family, natural counterfactuals, MC benchmarks that are not templated |
| laya hides the model | Fallback loader built from hour 0 |
| Budget overrun | Re-cost after Phase 0, spending limit, ordered cuts |

## 10. Definition of done

- [ ] Layout and forward pass verified for laya and smolaya.
- [ ] Order flip rate, position vs neighbor decomposition, and interaction matrices M with CIs.
- [ ] Named competition and/or position heads, with ablations that change M and order flips as predicted (or a documented null).
- [ ] Parallel-layout control: M and order flips under masked parallel (and shipped parallel if it exists), plus the layer bands that carry the interaction.
- [ ] Choice vs confidence map on rotation vs scale change, a separability test in both directions, and a depth comparison of when û and ‖z‖ stabilize.
- [ ] Margin-correctness AUROC and ECE (raw and tempered) before and after interventions.
- [ ] Label-following rates across semantic, symbolic, and neutral labels, a mechanism, and an ablation that reduces it.
- [ ] Jailbreak: decodable-vs-decisive layers, compared to the LLM injection result; competition and confidence components tested on it.
- [ ] laya vs smolaya comparison.
- [ ] Every main claim replicated on held-out data and confirmed in fp32.
- [ ] Spend ≤ $100, logged.
- [ ] Every figure regenerates with a `modal run` command.
