# Release: When Labels Override Evidence

Everything needed to check and rebuild the results of the laya mechanistic-interpretability study.

## Contents

| Path | What it is |
|---|---|
| `report/report.pdf`, `report/report.html` | The final report. Every number and figure is computed from `results/` by `scripts/build_report.py`. |
| `scripts/build_report.py` | Rebuilds the report from `results/`: `python release/scripts/build_report.py` |
| `results/` | Raw aggregate results (JSON, plus small pickles of per-item arrays), one folder per experiment |
| `data/` | Exact item sets and splits. See "Data" below. |
| `../modal_interp.py`, `../interp/` | The code that produced `results/` (runs on Modal) |
| `../NEXT_STEPS_PLAN.md` | Pre-registered hypotheses, metrics, amendments and the outcome of each phase |
| `../LITERATURE_NOTES.md` | Related work as checked against the papers, with the search log |

## Environment

| Component | Version |
|---|---|
| laya (PyPI) | 0.4.1 |
| transformers | 5.19.0 |
| torch | 2.14.1 |
| nnsight | 0.7.0 |
| datasets | 5.1.0 |
| Python (Modal image) | 3.12 |
| Modal client | 1.6.1 |
| GPU | NVIDIA H100 80GB |

### Checkpoints

| Name | Source | Pinned revision |
|---|---|---|
| laya | `convaiinnovations/laya` | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |
| laya-multilingual | `convaiinnovations/laya-multilingual` | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` |
| laya-typed-decisions | `convaiinnovations/laya`, subfolder `typed-decisions` | — |
| smolaya | `anon767tom/smolaya` | — |

The pinned revisions are laya 0.4.1's `PINNED_REVISIONS`.

### Instrument fidelity

The explicit forward pass matches `predict` to ≤ 5e-5 in probability on laya, smolaya, typed-decisions and multilingual (`results/phase0/`, `results/track4/multilingual.json`).

## Reproducing each result

Run from the repo root, with Modal authenticated:

```
modal run modal_interp.py --task <task> [--paths <models>]
```

| Task | Produces |
|---|---|
| `verify` | `results/phase0/verify_<model>.json` |
| `track1`, `track1b`, `track2`, `track3`, `track4`, `track43` | original C1–C4 studies (`results/track*`) |
| `followups`, `fp32`, `llm_steps`, `label_guard` | `results/followups/` (requires the `openai-secret` Modal secret for `llm_steps`) |
| `stretch`, `seeds` | `results/retrain/`, `results/sae/` (SAE weights are on the Modal volume only) |
| `phase1` | `results/phase1/` (C1 factorial) |
| `phase2`, `e4b` | `results/phase2/` (C3 causal tests; `phase2` also builds the confirmatory noul set and contradicting passages) |
| `phase3` | `results/phase3/` (C2 amplifier test) |
| `phase4` | Phase 1–3 on typed-decisions and multilingual |
| `phase5` | `results/phase5/` (label-regime retraining) |

Workspace GPU concurrency is 10. `phase5` launches 18 jobs, so 8 queue; this adds no cost.

## Data

| File | Contents | Text included? |
|---|---|---|
| `mc.jsonl` | ARC-Challenge/Easy (test), CommonsenseQA (validation), QASC (validation, 4 options): 1,989 items with splits | yes |
| `mc_confirm.jsonl` | OpenBookQA (test) + MMLU subset: 1,000 confirmatory items | yes |
| `boolq.jsonl` | BoolQ validation, passages ≤ 150 tokens: 1,000 items | yes |
| `noul_confirm.jsonl` | RTE validation (277) + MultiRC validation (500) | yes |
| `c3_llm.jsonl` | 971 yes/no items written by gpt-5.6-luna, answers checked by gpt-5.6-terra | yes |
| `boolq_contra.jsonl` | 187 BoolQ passages edited by gpt-5.6-terra to flip the answer, checked by gpt-5.6-luna | yes |
| `jb_natural`, `jb_pairs`, `jb_pairs_kept`, `injection`, `jb_counterfactuals` `.manifest.jsonl` | Jailbreak and prompt-injection sets | **no**: ids, labels, splits and SHA-256 of each text only |

The jailbreak and injection texts are left out because they contain harmful-request prompts and come from datasets with non-commercial or other licence terms. `interp/safety.py` and `interp/llm.py` rebuild them from the public sources, and the hashes let you verify the rebuild matches.

Each source dataset keeps its own licence (for example BoolQ CC BY-SA 3.0, Alpaca and ToxicChat CC BY-NC 4.0). Check the terms before reusing anything.

## Cost

- **Modal compute:** about $37 metered across all phases (see `report/report.pdf`, section 9, for the figure at build time), mostly covered by free credits.
- **OpenAI API:**
  - gpt-5.6-terra: 1,491 calls, 193,750 prompt + 125,199 completion tokens;
  - gpt-5.6-luna: 444 calls, 72,704 prompt + 156,462 completion tokens.

## Known limits

The full list is in `report/report.pdf`, section 8. The main ones:
- the multi-token guard extension is exploratory;
- rewritten passages and jailbreaks were checked by a second model, not by hand;
- Wen et al. 2026 was read at abstract level;
- laya's own training data are not public.
