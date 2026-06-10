# Attention

[![Type](https://img.shields.io/badge/Type-Concept%20Note-2563eb?style=flat-square\&logo=bookstack\&logoColor=white)](.)
[![Area](https://img.shields.io/badge/Area-Transformers-16a34a?style=flat-square\&logo=buffer\&logoColor=white)](transformers.md)
[![Level](https://img.shields.io/badge/Level-Foundation-111827?style=flat-square\&logo=target\&logoColor=white)](.)
[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-b31b1b?style=flat-square\&logo=arxiv\&logoColor=white)](https://arxiv.org/abs/1706.03762)

> Attention lets a model learn where to look, instead of forcing all information into one fixed representation.

Attention is a mechanism that lets a neural network decide which parts of the input are most relevant when building a representation or producing an output.

It became important in sequence-to-sequence learning because earlier RNN/LSTM encoder-decoder models compressed the full input into one fixed-size vector. Attention reduced this bottleneck by keeping all encoder states accessible and learning how much each one should matter at each decoding step.

Later, the Transformer made attention the central architecture block.

---

## Quick View

| Question | Answer |
| --- | --- |
| Core idea | Learn to weight which input positions matter |
| Earlier setup | RNN/LSTM encoder-decoder models |
| Main problem | Fixed-vector bottleneck |
| First major use | Neural machine translation |
| Transformer use | Self-attention between tokens |
| Key formula | `softmax(QKᵀ / √dₖ) V` |
| Why it matters | Foundation of Transformers and most modern LLM architectures |

---

## Learning Flow

```text
RNN/LSTM encoder-decoder
→ fixed-vector bottleneck
→ attention over encoder states
→ self-attention
→ Transformer blocks
→ modern LLMs
```

---

## 1. What Came Before Attention

Before attention became widely used, many neural sequence-to-sequence systems used RNN or LSTM encoder-decoder models.

These models were used for tasks like machine translation, where the input and output are both sequences.

Basic flow:

```text
source sentence
  → RNN/LSTM encoder
  → fixed-size context vector
  → RNN/LSTM decoder
  → target sentence
```

The encoder reads the input sequence one token at a time.

It compresses the sequence into a fixed-dimensional vector.

The decoder then uses that vector to generate the output sequence one token at a time.

This was a major step because neural networks could now map variable-length input sequences to variable-length output sequences.

---

## 2. The Fixed-Vector Bottleneck

The weakness was the fixed-size context vector.

One vector had to carry information about the entire input sequence.

For short sentences, this could work.

For longer or more detailed sentences, useful information could be lost.

Example:

```text
The student who submitted the assignment late asked the professor for an extension.
```

When generating the translation, the decoder may need to focus on different words at different steps:

- `student`
- `submitted`
- `assignment`
- `late`
- `professor`
- `extension`

A single compressed vector is not a strong place to preserve all of that information.

This is the bottleneck attention was designed to reduce.

---

## 3. What Attention Changed

Attention did not immediately replace encoder-decoder models.

It first improved them.

Instead of forcing the decoder to depend only on the final compressed vector, attention gave the decoder access to all encoder hidden states.

New flow:

```text
source sentence
  → encoder
  → hidden states h₁, h₂, ..., hₙ

decoder step t
  → attention weights over h₁...hₙ
  → context vector for this step
  → decoder predicts next token
```

At each decoding step, the model asks:

```text
Which source positions are most useful right now?
```

The answer is not a hard rule.

It is a learned distribution over the input positions.

This is why attention is often described as a soft alignment mechanism.

---

## 4. Encoder-Decoder Attention

In encoder-decoder attention, the decoder attends to encoder outputs.

```text
encoder states = information from input tokens
decoder state  = current generation state
attention      = matching between decoder state and encoder states
```

Simple flow:

```text
Input tokens
  → Encoder
  → Encoder hidden states
  → Attention weights
  → Context vector for current decoder step
  → Decoder predicts next token
```

This is useful in translation because different output words often align with different input words.

Example:

```text
English input: The student submitted the assignment late.
Output:        ...
```

When generating the translated word for `student`, the decoder should focus more on the encoder state around `student`.

When generating the translated word for `assignment`, it should focus more on `assignment`.

Attention gives the model a learned way to choose relevant source positions at each step.

---

## 5. From Encoder-Decoder Attention to Self-Attention

Encoder-decoder attention connects two sequences:

```text
decoder tokens → encoder outputs
```

Self-attention works inside one sequence:

```text
tokens → other tokens in the same sequence
```

Example:

```text
The animal did not cross the street because it was tired.
```

When building the representation of `it`, the model should pull more useful information from `animal` than from `street`.

Self-attention lets every token exchange information with every other token.

```text
RNN/LSTM style:
t₁ → t₂ → t₃ → t₄

Self-attention style:
t₁ ↔ t₂ ↔ t₃ ↔ t₄
```

This shift is one of the main reasons Transformers are different from recurrent models.

---

## 6. Query, Key, and Value

Attention uses three learned projections of each token:

| Vector | Role |
| --- | --- |
| **Query (Q)** | What this token is looking for |
| **Key (K)** | What this token offers for matching |
| **Value (V)** | What this token passes forward |

Simple process:

```text
Query compares with Keys
→ scores
→ softmax weights
→ weighted sum of Values
→ new representation
```

Think of it like soft search.

A query matches against keys, and the corresponding values are returned according to learned relevance weights.

---

## 7. Scaled Dot-Product Attention

The Transformer uses scaled dot-product attention:

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

| Part | What it does |
| --- | --- |
| `QKᵀ` | Scores each query against every key |
| `/ √dₖ` | Scales scores so they do not become too large |
| `softmax` | Turns scores into weights that sum to 1 |
| `V` | Mixes values using those weights |

The output is a new representation where each token has gathered information from relevant tokens.

### Why divide by `√dₖ`?

As the key/query dimension grows, dot products can become large.

Large scores can push softmax into very sharp distributions, where one position gets almost all the weight.

That can make gradients smaller and training harder.

Dividing by `√dₖ` keeps the score scale more stable.

---

## 8. Three Types of Attention

| Type | Where it happens | What it does |
| --- | --- | --- |
| **Encoder-decoder attention** | Decoder attends to encoder outputs | Connects output generation to input information |
| **Self-attention** | Tokens attend to tokens in the same sequence | Builds contextual token representations |
| **Masked self-attention** | Decoder tokens attend only to previous/current tokens | Prevents future-token leakage during generation |

---

## 9. Multi-Head Attention

One attention head gives the model one learned way to compare tokens.

Multi-head attention runs several attention heads in parallel.

Each head has its own learned Q/K/V projections.

```text
input
  → head 1
  → head 2
  → head 3
  → ...
  → concatenate
  → linear projection
  → output
```

Different heads may learn different kinds of relationships, such as local patterns, long-range dependencies, or alignment-like behavior.

The outputs are concatenated and projected back into the model dimension.

The important point is not that every head has a clean human-readable role.

The useful idea is that multiple heads let the model attend through multiple representation subspaces.

---

## 10. Where the Transformer Fits

The Transformer did not invent attention.

It made attention the main architecture block.

Before Transformers, sequence models usually relied on recurrence:

```text
token 1 → token 2 → token 3 → token 4
```

The Transformer replaced recurrence with stacked attention-based blocks.

```text
tokens
→ self-attention
→ feed-forward network
→ contextual token representations
```

This made training more parallelizable and helped models capture relationships between distant tokens.

---

## 11. Transformer Encoder Block

A Transformer encoder reads the input sequence and builds contextual representations.

Simple encoder block:

```text
token embeddings + positional encoding
  → self-attention
  → add & norm
  → feed-forward network
  → add & norm
```

The encoder uses self-attention because each input token can look at other input tokens.

Example:

```text
The cat sat on the mat.
```

The representation of `cat` can use information from `sat`, `mat`, and the rest of the sentence.

---

## 12. Transformer Decoder Block

A Transformer decoder generates output tokens.

Simple decoder block:

```text
previous output tokens
  → masked self-attention
  → add & norm
  → encoder-decoder attention
  → add & norm
  → feed-forward network
  → add & norm
```

The decoder has two important attention parts:

| Part | Purpose |
| --- | --- |
| Masked self-attention | Lets generated tokens attend only to previous/current tokens |
| Encoder-decoder attention | Lets decoder tokens attend to encoder output states |

Masked self-attention is needed because during generation the model should not look at future tokens.

Encoder-decoder attention is used when the model has a separate input sequence to condition on, such as translation or summarization.

---

## 13. Decoder-Only LLMs

Many modern LLMs use decoder-only Transformer architectures.

In decoder-only models, there is no separate encoder-decoder attention block.

The model mainly uses masked self-attention.

```text
previous tokens
  → masked self-attention
  → feed-forward layers
  → next token prediction
```

The mask ensures each token can only attend to previous tokens, not future tokens.

This is what allows the model to generate text left-to-right.

---

## 14. Common Confusions

| Confusion | Correction |
| --- | --- |
| Attention means reasoning | Attention is a learned weighting mechanism; reasoning-like behavior depends on the full trained model |
| Attention is memory | Attention works over the current context window; it is not long-term persistent memory by itself |
| Attention started with Transformers | Attention existed before Transformers; Transformers made it the main architecture block |
| Self-attention and encoder-decoder attention are the same | They use similar math, but Q/K/V can come from different places |
| Multi-head attention means each head has a clear role | Some heads may show interpretable patterns, but not every head maps cleanly to a human label |
| Attention weights fully explain model behavior | They can help inspection, but they are not a complete explanation |

---

## Related Papers

- [**Sequence to Sequence Learning with Neural Networks**](https://arxiv.org/abs/1409.3215) - Used LSTM encoder-decoder sequence learning with a fixed-dimensional representation.
- [**Neural Machine Translation by Jointly Learning to Align and Translate**](https://arxiv.org/abs/1409.0473) - Introduced attention for neural machine translation by letting the decoder soft-search relevant source positions.
- [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) - Introduced the Transformer architecture based on attention mechanisms without recurrence or convolution.
- [**An Image is Worth 16x16 Words**](https://arxiv.org/abs/2010.11929) - Applied a pure Transformer to image patches for image classification.

## Related Concepts

- [`transformers.md`](transformers.md)
- [`positional-encoding.md`](positional-encoding.md)
- [`residual-connections.md`](residual-connections.md)

## Related Experiments

- [`attention-from-scratch/`](../experiments/attention-from-scratch/)
- [`mini-transformer-block/`](../experiments/mini-transformer-block/)

---

[![Home](https://img.shields.io/badge/Home-README-0f172a?style=flat-square\&logo=github\&logoColor=white)](../README.md)
[![Concepts](https://img.shields.io/badge/Back-Concepts-0f172a?style=flat-square\&logo=bookstack\&logoColor=white)](./)
[![Transformers](https://img.shields.io/badge/Related-Transformers-2563eb?style=flat-square\&logo=buffer\&logoColor=white)](transformers.md)
[![arXiv](https://img.shields.io/badge/Paper-1706.03762-b31b1b?style=flat-square\&logo=arxiv\&logoColor=white)](https://arxiv.org/abs/1706.03762)
