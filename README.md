# Mechanistic Interpretability of Laya

This repository contains the code, results, and reports for a mechanistic interpretability study of **Laya**, a non-autoregressive encoder model that scores candidate options from a single input sequence.

The main study asks why Laya's binary `noul` questions can follow the semantic meaning of an option label such as `yes` or `no` rather than the accompanying option description. It traces the label pathway, tests an inference-time attention intervention, and fine-tunes controlled models to investigate conditions that can produce a similar shortcut.

## Main result

On a frozen benchmark of 600 human-written MNLI/QNLI examples, swapping `yes` and `no` flips 96.7% of answers that were correct under the normal labels. Swapping neutral symbols flips 4.1%. Blocking local attention from each `[MASK]` to its label reduces label-following to 2.6%, with an accuracy cost of 1.8 percentage points on that benchmark.

In a controlled fine-tuning study, the Planted condition averaged 88.9% label-following across six seeds, compared with 14.7% for the fixed-order baseline. This shows that a particular training setup can produce the shortcut. It does **not** establish that the released Laya checkpoint was trained on the same data distribution.

The guard is not a universal patch. The preregistered all-layer intervention met the 10% label-following threshold on only three of six retrained ModernBERT planted seeds, and in some runs it also damaged ordinary task accuracy. Results and failed predictions are reported rather than omitted.

## Repository guide

- `NEXT_STEPS_PLAN.md`: preregistered hypotheses, metrics, interventions, and subsequent amendments.
- `release/scripts/build_report.py`: rebuilds the report from the released result files.
- `release/results/`: result JSONs and small supporting artifacts used by the report.
- `release/report/`: report assets, including figure files.
- `interp/label_guard.py`: inference-time attention guard used in the study.
- `How_Laya_Chooses_paper.pdf` and related report files: full write-up and supporting analysis, when present in the repository checkout.

The exact paths available in a particular checkout may differ as the release is updated. Start by inspecting the files above before assuming that every experiment can be rerun from one command.

## Rebuilding the report

From the repository root, in the pinned study environment, run:

```bash
python release/scripts/build_report.py
```

This rebuilds the report from released results. It is not the same as rerunning all model training and interventions. The study plan records the environment used for the experiments as `laya 0.4.1`, `transformers 5.19.0`, `torch 2.14.1`, and `nnsight 0.7.0`, with model revisions pinned through `laya.PINNED_REVISIONS`. Treat these as the study's reported environment, not as a guarantee that the latest package release reproduces the same checkpoints.

Some jailbreak and prompt-injection text is represented by hashed manifests rather than redistributed verbatim. Those manifests help identify evaluation rows, but do not by themselves make it possible to rerun the text-level safety evaluations from scratch. Use the released aggregate results for those comparisons unless you have authorized access to the original data.

## Model and version notes

The English `laya` and `laya-typed-decisions` checkpoints use ModernBERT-large. `laya-multilingual` uses a different backbone, mmBERT-base, with 22 layers. Do not assume that layer numbers or circuit locations transfer one-to-one between the architectures.

The experiment environment used the Python package version `laya 0.4.1`. The current Hugging Face model card has separately listed runtime version `0.3.20`; do not describe `0.4.1` as the checkpoint version.

- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Laya Issue #156, reporting label sensitivity](https://github.com/NandhaKishorM/laya/issues/156)
- [InjecGuard: prompt-injection detection](https://arxiv.org/abs/2410.22770)

## How to interpret the findings

1. The strongest claim is specific to the released checkpoints and the tested option format: label words can dominate the decision even when the description is informative.
2. The retraining experiments show that shortcut formation depends on the training setup. They do not reconstruct Laya's proprietary training history.
3. The residual-stream direction experiment in §7.3 failed to preserve accuracy. Because the direction was estimated by contrasting normal-label and swapped-label runs that usually change the answer, it is confounded with the output. It does not establish a general limitation of linear concept erasure or LEACE.
4. The jailbreak-style tests show that dramatic style can increase flags on benign prompts. The plain, non-theatrical override condition was not tested, so the study does not establish that style is the detector's only cue.
5. The paper's jailbreak AUROC values (0.940, 0.919, and 0.947 in different parts of the draft) still need to be reconciled against saved predictions and exact split metadata before publication.

## Scope

This is a research artifact, not a production safety guarantee. Validate label-swap behavior and guard side effects on the exact checkpoint, prompt template, and task distribution you plan to use. Keep accuracy and safety metrics together when evaluating any intervention.

## Legacy serving demo

The original README described an Azure CPU VM demo using Ollaya and Streamlit. That deployment path is separate from this study's reproduction workflow. Those serving instructions are in `SERVING.md`.
