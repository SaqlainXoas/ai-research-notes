# AI Research Notes

![Focus](https://img.shields.io/badge/Focus-AI%20%26%20ML%20Research-111827)
![Style](https://img.shields.io/badge/Style-Concept--First-2563eb)
![Status](https://img.shields.io/badge/Status-Active-16a34a)

A focused public notebook for understanding modern machine learning through connected concept notes, paper studies, and reproducible experiments.

The current work follows three connected tracks:

1. Transformer foundations
2. Representation learning and neural retrieval
3. ML generalization, calibration, and reliability

The aim is depth over coverage: explain a concept clearly, connect it to the relevant research, and use a small experiment when implementation can answer a concrete question.

## Start Here

### Transformer foundations

- [Transformers](concepts/transformers.md) — the architecture, its building blocks, and the encoder/decoder model families.
- [Attention](concepts/attention.md) — how token representations gather information from other tokens.
- [Positional encoding](concepts/positional-encoding.md) — how attention-based models represent order.
- [Residual connections](concepts/residual-connections.md) — why shortcut paths make deep networks easier to optimize.
- [Layer normalization](concepts/layer-normalization.md) — how each token representation is normalized across its hidden features.
- [Attention Is All You Need](papers/nlp-llms/attention-is-all-you-need/) — a structured study of the original Transformer paper.

### Representation learning and neural retrieval

- [Embedding models and bi-encoders](concepts/embedding-models-and-bi-encoders.md) — reusable text vectors for retrieval.
- [Rerankers and cross-encoders](concepts/rerankers-and-cross-encoders.md) — joint pairwise scoring for higher-precision ranking.

## Repository Map

| Location | Contents |
| --- | --- |
| [`concepts/`](concepts/) | Concept-first notes that build intuition and link related ideas |
| [`papers/`](papers/) | Paper studies covering the problem, method, evidence, limitations, and follow-up questions |
| [`experiments/`](experiments/) | Focused, reproducible experiments tied to a research question |
| [`assets/`](assets/) | Diagrams used by the notes |

## Status

The Transformer and neural-retrieval foundations are actively documented. The experiments directory is intentionally a starting point: new experiments will be added only when they address a clear question from these notes.

---

[![GitHub](https://img.shields.io/badge/GitHub-SaqlainXoas-0f172a)](https://github.com/SaqlainXoas)
[![Medium](https://img.shields.io/badge/Medium-@saqlainjuna-0f172a)](https://medium.com/@saqlainjuna)
