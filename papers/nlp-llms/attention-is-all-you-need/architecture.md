# Architecture

[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1706.03762)
[![Back](https://img.shields.io/badge/Back-Paper%20Overview-0f172a?style=flat-square&logo=bookstack&logoColor=white)](README.md)

This page focuses on the original Transformer as an encoder-decoder architecture for sequence transduction tasks such as translation.

## Architecture Overview

The original Transformer is an encoder-decoder model designed for sequence transduction tasks such as translation.

```text
source sequence
  ↓
encoder stack
  ↓
encoder output representations
  ↓
decoder stack
  ↓
target sequence
```

The encoder reads the input sequence and builds contextual representations.

The decoder generates the output sequence token by token.

In the base Transformer:

| Parameter | Value |
| --- | --- |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Model dimension `d_model` | 512 |
| Attention heads | 8 |
| Feed-forward hidden dimension `d_ff` | 2048 |
| Dropout | 0.1 |

## Encoder Block

Each encoder layer has two main sub-layers:

```text
multi-head self-attention
-> feed-forward network
```

With residual connections and layer normalization around each sub-layer:

```text
input tokens + positional encoding
  -> multi-head self-attention
  -> add & layer norm
  -> feed-forward network
  -> add & layer norm
  -> contextual output
```

Encoder self-attention is unmasked.

Each input token can attend to all other input tokens.

Example:

```text
The cat sat on the mat.
```

The representation of `cat` can use information from `sat`, `mat`, and the rest of the sentence.

This is what makes the representation contextual.

A static word embedding gives a word representation.

Self-attention helps build a word-in-context representation.

## Decoder Block

Each decoder layer has three main sub-layers:

```text
masked multi-head self-attention
-> encoder-decoder attention
-> feed-forward network
```

Simple flow:

```text
previous output tokens + positional encoding
  -> masked multi-head self-attention
  -> add & layer norm
  -> encoder-decoder attention
  -> add & layer norm
  -> feed-forward network
  -> add & layer norm
  -> next token prediction
```

The decoder has two attention stages.

First, it attends to the tokens already generated.

Then, it attends to the encoder output representations from the source sequence.

Example:

```text
Generated so far:
"The student"

Next token prediction can use:
"The", "student"

It cannot use future target words.
```

This is why decoder generation is still step by step.

## Feed-Forward Network

After attention, each position passes through a feed-forward network.

The paper defines it as:

```text
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

In the base model:

```text
input/output dimension = 512
inner feed-forward dimension = 2048
```

Attention and feed-forward layers do different jobs.

```text
attention:
  mixes information across positions

feed-forward network:
  transforms each position independently
```

This is important because the Transformer is not "only attention."

Attention is the main sequence-mixing operation, but the model also relies on feed-forward layers, residual connections, layer normalization, embeddings, and positional encodings.

## Positional Encoding

Self-attention does not naturally know token order.

Without position information, these two sentences contain the same words:

```text
dog bites man
man bites dog
```

But they do not mean the same thing.

Because the Transformer does not use recurrence or convolution, it adds positional encodings to token embeddings:

```text
input representation = token embedding + positional encoding
```

The original paper uses sinusoidal positional encodings:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

The paper also tested learned positional embeddings and found similar results.

Sinusoidal encodings were kept because they may allow the model to extrapolate to sequence lengths longer than those seen during training.

## Related Concepts

- [`Attention`](../../../concepts/attention.md)
- [`Transformers`](../../../concepts/transformers.md)
- [`Positional Encoding`](../../../concepts/positional-encoding.md)
- [`Residual Connections`](../../../concepts/residual-connections.md)
