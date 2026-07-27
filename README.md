# AI Research Notes

![Focus](https://img.shields.io/badge/Focus-AI%20%26%20ML%20Research-111827)
![Style](https://img.shields.io/badge/Style-Concept--First-2563eb)
![Status](https://img.shields.io/badge/Status-Active-16a34a)

A personal notebook for exploring modern machine learning through concept notes, paper studies, and reproducible experiments.

The current work follows three connected tracks:

1. Transformer foundations
2. Representation learning and neural retrieval
3. ML generalization, calibration, and reliability

## Start Here

### Transformer foundations

- [Transformers](concepts/transformers.md) — the architecture, its building blocks, and the encoder/decoder model families.
- [Attention](concepts/attention.md) — how token representations gather information from other tokens.
- [Positional encoding](concepts/positional-encoding.md) — how attention-based models represent order.
- [Residual connections](concepts/residual-connections.md) — why shortcut paths make deep networks easier to optimize.
- [Layer normalization](concepts/layer-normalization.md) — how each token representation is normalized across its hidden features.
- [Feed-forward networks](concepts/feed-forward-networks.md) — how Transformer blocks transform each token after attention mixes context.
- [Attention Is All You Need](papers/nlp-llms/attention-is-all-you-need/) — a structured study of the original Transformer paper.
- [Attention from scratch](experiments/attention-from-scratch/) — a runnable PyTorch walkthrough of Q/K/V projections, scaling, causal masking, and multi-head attention.
- [Tiny encoder-only Transformer](experiments/tiny-transformer/) — a runnable one-block encoder classifier that learns whether `A` appears before `B`.
- [Positional information ablation](experiments/positional-information/) — a five-seed controlled comparison of that classifier with and without learned positions.

### Representation learning and neural retrieval

- [Embedding models and bi-encoders](concepts/embedding-models-and-bi-encoders.md) — reusable text vectors for retrieval.
- [Contrastive learning](concepts/contrastive-learning.md) — how positive and negative pairs shape a useful embedding space.
- [Rerankers and cross-encoders](concepts/rerankers-and-cross-encoders.md) — joint pairwise scoring for higher-precision ranking.
- [Sentence-BERT](papers/nlp-llms/sentence-bert/) — an early foundational architecture that made BERT-style encoders practical for reusable semantic-search embeddings; it complements rather than replaces cross-encoder scoring.

## Repository Map

| Location | Contents |
| --- | --- |
| [`concepts/`](concepts/) | Concept-first notes that build intuition and link related ideas |
| [`papers/`](papers/) | Paper studies covering the problem, method, evidence, limitations, and follow-up questions |
| [`experiments/`](experiments/) | Focused, reproducible experiments tied to a research question |
| [`assets/`](assets/) | Diagrams used by the notes |

## Status

The Transformer and neural-retrieval foundations are documented. The focused experiments move from [attention mechanics](experiments/attention-from-scratch/) to a complete [tiny encoder-only classifier](experiments/tiny-transformer/), then use a [positional-information ablation](experiments/positional-information/) to test one component under controlled conditions. The [Sentence-BERT study](papers/nlp-llms/sentence-bert/) is the bridge from that encoder work to reusable representations; the next retrieval task will examine bi-encoder retrieval with cross-encoder reranking. Future experiments will be added only when they address a clear question from these notes.

---

[![GitHub](https://img.shields.io/badge/GitHub-SaqlainXoas-0f172a)](https://github.com/SaqlainXoas)
[![Medium](https://img.shields.io/badge/Medium-@saqlainjuna-0f172a)](https://medium.com/@saqlainjuna)
