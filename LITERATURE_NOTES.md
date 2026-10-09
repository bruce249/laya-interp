# Literature notes (plan v4, Phase 6)

Compiled 2026-10-09. Each paper was fetched from arXiv or the publisher and read through a summarizing fetch tool, so claims were checked against the paper's own text, but not every line was read by a human. Where only the abstract or part of a paper was available, that is noted. Before citing any of these in a submission, a person should read the relevant sections.

## Closest prior work, claim by claim

| Paper | What it actually shows | How much was read | Relation to this study |
|---|---|---|---|
| **Set-Based Prompting**: *Order-Independence Without Fine Tuning* (McIlroy-Young et al., NeurIPS 2024, arXiv 2406.06581) | Removes order dependence in decoder LLMs (GPT-2, Llama 2/3, Mistral) by (1) a mask that stops parallel sub-sequences attending to each other and (2) giving them shared positions. Ablation on the **first 20 MMLU questions**: "modifying the attention mask reduces variation much more than the positional encoding, although the effect varies significantly between models." Accuracy usually stays within the order-variation range. Decoders only; no encoders. | Full HTML | Our C1 "options blocked + shared positions" condition is their method transplanted to an encoder. **Contrast holds, with caveats:** in laya, blocking option attention alone leaves flips at 28%, the opposite of their mask-dominant finding. But their ablation is tiny (20 questions), and in our bidirectional encoder "positions only" cannot be tested directly because positions are what bind each [MASK] to its option (shared positions alone make all [MASK]s identical). |
| *Order Independence With Finetuning* (Brown & McIlroy-Young, ICLR 2025 Bi-Align workshop, arXiv 2503.23483) | Fine-tuning with set-based prompting reduces its out-of-distribution cost and improves robustness to answer permutations on MMLU, CSQA and ARC. | Search summary | laya already ships a "parallel" option layout as a trainable config. Our order-invariant condition costs 2–10 accuracy points zero-shot; this paper suggests training in that layout would close the gap. |
| **Anchored Answers** (Li & Gao, ACL Findings 2025, arXiv 2405.03205) | GPT-2 family favours option "A" in MCQs. Localized with logit lens to MLP value vectors (e.g. GPT2-Small layer 9) and heads (L8H1, L10H8); editing value vectors removes the bias. Attributes the bias to learned internal components tied to the "A" token, not to positional encoding. | First 100k of 122k characters | Different mechanism class: an option-ID token prior in a decoder. Laya shows almost no slot or symbol bias (A/B swaps ~3%, slot index explains 0.2% of order variance); its order effect is item-specific and positional. |
| *Large Language Models Sensitivity to the Order of Options in MCQs* (Pezeshkpour & Hruschka, NAACL Findings 2024, arXiv 2308.11483) | Reordering options changes LLM accuracy substantially. | Abstract and summary | Establishes the behaviour; we study its mechanism in an encoder decision model. |
| *Large Language Models Are Not Robust Multiple Choice Selectors* (Zheng et al., ICLR 2024, arXiv 2309.03882) | Selection bias in LLMs stems mainly from token-level priors on option IDs. | Abstract and summary | Same contrast as Anchored Answers: an ID-token prior, which laya largely lacks. |
| **Label Words are Anchors** (Wang et al., EMNLP 2023, arXiv 2305.14160) | In-context learning in decoders (GPT2-XL, GPT-J): demonstration label words aggregate information in shallow layers, deep layers read it. Blocking aggregation into label words in the first 5 layers matters; in the last 5 it does not. | Full HTML | Different setting: in-context *demonstration* labels in decoders, not *answer-option* labels in a scoring encoder. In laya the option label is read directly by the answer [MASK], redundantly across local layers and most strongly late; blocking an early band alone has no effect. |
| *Larger Language Models Do In-Context Learning Differently* (Wei et al., 2023, arXiv 2303.03846) | Small models follow semantic priors of labels and ignore flipped labels; large models can override them. | Search summary | Consistent with laya following label semantics. Our Phase 5 shows training can install the behaviour when the label word is the only cue to option identity. |
| **Confidence Regulation Neurons** (Stolfo et al., NeurIPS 2024, arXiv 2406.16254) | Entropy neurons in GPT-2, Pythia, Phi-2, Gemma and LLaMA2 write to a near-null space of the unembedding and scale logits through the final LayerNorm, changing entropy with little effect on rankings. The paper says this differs from a uniform temperature. | First 100k of 124k characters | **Contrast:** laya's late MLPs act at the [MASK]s, and their calibration effect is fully reproduced by per-item rescaling and beaten by a global temperature. They behave like a temperature, which is exactly what entropy neurons are argued not to be. |
| **Where Do LLMs Decide to Break the Rules?** (Wen et al., 2026, arXiv 2609.37737) | In five decoder LLMs (4B–32B), attack information is linearly decodable from the first layer, but causal leverage appears only at a late bottleneck in the final third; the mechanism is a rank-8 to rank-64 subspace; patching it reverses compliance in 77–92% of cases. No encoder or guard models. | **Abstract only** | We see the same early-decodable, late-decisive pattern in an encoder classifier, routed prompt → question tokens → [MASK]. Read the full paper before claiming more. |
| *Pathway to Relevance / Cross-Encoder Rediscovers a Semantic Variant of BM25* (arXiv 2502.04645) | Path patching in a BERT cross-encoder finds heads computing soft term frequency, saturation and length, aggregated BM25-style. | Search summary | Encoder mechanistic interpretability exists; this is the closest encoder precedent found. No option-competition or label-shortcut analysis. |
| *A Toy Model of Universality* (Chughtai et al., ICML 2023, arXiv 2302.03025) | Networks trained from different seeds learn different circuits for the same group-operation task. | Search summary | Our Phase 5 finding (identical label-following across seeds, different routes, so the guard transfers unevenly) is an instance of this known non-universality. What is new is the practical consequence: a mechanistically derived fix may not transfer to retrained siblings. |
| **laya model card** (convaiinnovations/laya) | "`noul` can follow its option labels instead of the state, most strongly on this English checkpoint." Recommends re-asking as a two-option `choice` with neutral keys. Training: RLCD with a strictly proper scoring reward. "Base checkpoints are near chance on typed-decisions zero-shot." | README | The failure is documented by the authors; our contribution is the mechanism, the guard, its evaluation and the training-origin experiment. The near-chance zero-shot note supports reading our MC accuracies (~0.45) as an off-task setting. |

## Searches run

Search terms used (web search, 2026-10-09):

- multiple-choice option-order sensitivity, positional bias mechanisms, attention heads
- LLM selection bias on option ID tokens
- confidence regulation and entropy neurons
- the arXiv ID 2609.37737
- mechanistic interpretability of encoder, cross-encoder and BERT classifiers
- jailbreak classifiers, linear probes, refusal directions
- the Order-Independence papers
- Label Words are Anchors
- semantic priors under flipped labels
- encoder label-word / verbalizer shortcuts
- circuit variation across random seeds

**Not found:** any work on answer-option label shortcuts or option competition in per-option [MASK] scoring encoders. Absence in these searches is not evidence that such work does not exist; social media, workshop papers and non-English venues were not searched.

## Consequences for the write-up

1. **C1:** keep the Set-Based Prompting contrast, but state that their ablation used 20 MMLU questions and decoders, and that our position-only cell is degenerate by construction. Cite Order Independence With Finetuning as the route to removing the zero-shot cost.
2. **C2:** frame it explicitly as the opposite of entropy neurons: a temperature-like amplifier, not a null-space regulator.
3. **C3:** cite the model card for the behaviour; claim the mechanism (redundant direct [MASK] → label read plus a question-token evidence relay), the guard, and the training-origin result. Cite Label Words are Anchors as a different setting, not as the same mechanism.
4. **Seeds:** cite the Toy Model of Universality for non-universality; claim only the practical consequence for fixes.
5. **C4:** treat as a replication of Wen et al. in an encoder; read the full paper first.
