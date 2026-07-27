# Attention Mechanics

[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1706.03762)
[![Back](https://img.shields.io/badge/Back-Paper%20Overview-0f172a?style=flat-square&logo=bookstack&logoColor=white)](README.md)

This page focuses on how attention works inside the Transformer rather than on the full model layout.

## Scaled Dot-Product Attention

The core attention operation is:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Think of it as learned soft search.

| Component | Role |
| --- | --- |
| **Q - Query** | What this token is looking for |
| **K - Key** | What each token offers for matching |
| **V - Value** | Information each token passes forward |
| `QKᵀ` | Similarity scores between queries and keys |
| `/ √dₖ` | Scaling factor to keep scores stable |
| `softmax` | Converts scores into weights that sum to 1 |
| `V` | Values are mixed using those weights |

Simple flow:

```text
compare Q with K
-> get attention scores
-> scale scores
-> softmax into weights
-> weighted sum of V
-> output representation
```

The model does not select only one token.

It creates a distribution over positions and mixes information based on learned relevance.

### Why divide by `√dₖ`?

Without scaling, dot products can become large when the key/query dimension grows.

Large scores can push softmax into very sharp distributions, where one position receives almost all the weight.

That can make gradients smaller and training harder.

Dividing by `√dₖ` keeps the score scale more stable.

## Multi-Head Attention

One attention head gives the model one learned way to compare tokens.

The Transformer runs multiple attention heads in parallel.

Each head has its own learned projections for queries, keys, and values:

```text
input
  ├─→ head 1  with WQ, WK, WV
  ├─→ head 2  with WQ, WK, WV
  ├─→ ...
  └─→ head h  with WQ, WK, WV
        ↓
     concatenate head outputs
        ↓
     output projection WO
        ↓
     final output
```

In the base Transformer:

```text
d_model = 512
heads = 8
d_k = d_v = 64
```

Different heads can attend through different representation subspaces.

One head may capture local relationships.

Another may capture longer-range dependencies.

Another may learn alignment-like behavior.

The heads are not guaranteed to have clean human-readable roles, but multiple heads give the model more flexible ways to compare tokens.

## Three Uses of Attention in the Transformer

The same attention formula is reused in different places.

The difference is where `Q`, `K`, and `V` come from.

| Attention type | Where | Q comes from | K / V comes from | Purpose |
| --- | --- | --- | --- | --- |
| **Encoder self-attention** | Encoder | input tokens | input tokens | build contextual input representations |
| **Masked decoder self-attention** | Decoder | previous target tokens | previous target tokens | generate without seeing future tokens |
| **Encoder-decoder attention** | Decoder | decoder states | encoder outputs | let decoder look back at the source sequence |

The decoder mask is important.

Future positions are blocked before softmax, so the model cannot use target words it should not have seen yet.

That preserves autoregressive generation.

## Self-Attention Trade-offs

Self-attention improved parallelization and shortened the path between distant tokens, but it also introduced a major cost: full attention compares every token with every other token.

The paper compares self-attention, recurrence, and convolution using three ideas:

| Factor | Question |
| --- | --- |
| Computational complexity | How expensive is each layer? |
| Parallelization | How many sequential operations are required? |
| Path length | How far must information travel between positions? |

The Transformer’s advantage was not only performance.

It reduced the number of required sequential operations.

```text
RNN:
token₁ -> token₂ -> token₃ -> token₄
sequential operations grow with sequence length

Self-attention:
token₁ ↔ token₂ ↔ token₃ ↔ token₄
all positions can interact in one layer
```

This helps distant tokens exchange information more directly.

However, self-attention has a cost:

```text
full attention cost grows roughly with sequence length squared
```

That is why long-context and efficient-attention methods became important follow-up research areas.

## Related Concepts

- [`Attention`](../../../concepts/attention.md)
- [`Transformers`](../../../concepts/transformers.md)
