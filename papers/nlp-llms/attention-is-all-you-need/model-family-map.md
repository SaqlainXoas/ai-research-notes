# Model Family Map

[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/1706.03762)
[![Back](https://img.shields.io/badge/Back-Paper%20Overview-0f172a?style=flat-square&logo=bookstack&logoColor=white)](README.md)

This page connects the original Transformer paper to the later model families built from the same core design ideas.

## Results

The Transformer was evaluated on WMT 2014 machine translation: English to German and English to French, two standard sequence transduction benchmarks at the time.

| Model | EN->DE BLEU | EN->FR BLEU | Training cost |
| --- | --- | --- | --- |
| Transformer base | 27.3 | 38.1 | ~12 hours on 8x P100 GPUs |
| Transformer big | 28.4 | 41.8 | ~3.5 days on 8x P100 GPUs |

The numbers tell part of the story. The more important point is what they meant in context.

For English to German, Transformer big improved over previous best results, including ensembles, by more than 2 BLEU.

For English to French, Transformer big achieved a new single-model state-of-the-art while using much less training compute than previous competitive systems.

Even the base model, trained in about 12 hours, already showed that attention-based sequence models could be both competitive and much faster to train.

## Where This Idea Shows Up Now

The Transformer paper introduced the encoder-decoder architecture for translation, but the same building blocks later appeared in different model families.

| Model family | Transformer form | How it connects to this paper |
| --- | --- | --- |
| **Translation Transformers** | Encoder-decoder | Closest to the original paper: encoder reads source text, decoder generates target text |
| **BERT / RoBERTa-style models** | Encoder-only | Use bidirectional self-attention to build strong text representations for understanding tasks |
| **GPT-style models** | Decoder-only | Use masked self-attention for left-to-right next-token prediction |
| **ChatGPT-style assistants** | GPT-style LLM + instruction/chat alignment | Built on the decoder-only language-model direction, then tuned to follow user instructions and produce helpful responses |
| **T5-style models** | Encoder-decoder | Convert many NLP tasks into text-to-text problems using a Transformer encoder-decoder design |
| **Vision Transformers** | Encoder-style over image patches | Treat image patches like tokens and apply Transformer blocks to vision tasks |

This is the practical link:

```text
Original Transformer:
encoder-decoder attention model for translation

BERT:
encoder-only Transformer for language understanding

GPT / ChatGPT-style models:
decoder-only Transformer for next-token generation and dialogue

T5 / translation models:
encoder-decoder Transformer for text-to-text generation

Vision Transformer:
Transformer blocks applied to image patches
```

The paper did not directly introduce BERT, GPT, ChatGPT, T5, or Vision Transformer.

It introduced the architecture pattern that made these later model families possible.

## Connection to Modern LLMs

The original Transformer is an encoder-decoder model.

Many modern LLMs use decoder-only Transformer architectures.

The connection is masked self-attention.

```text
Transformer paper:
encoder-decoder model for translation

Decoder-only LLMs:
masked self-attention + feed-forward layers
-> next-token prediction
```

Decoder-only models drop the encoder and the encoder-decoder attention block.

They keep the main Transformer decoder ingredients: masked self-attention, feed-forward networks, residual connections, layer normalization, and positional information.

Masked self-attention handles left-to-right generation.

The feed-forward layers and residual pathways help build richer token representations across many stacked layers.

Simple decoder-only flow:

```text
previous tokens
  -> masked self-attention
  -> feed-forward layers
  -> next token prediction
```

So the direct line is:

```text
Transformer paper
-> attention as the main architecture block
-> masked self-attention for generation
-> decoder-only language models
```

This paper did not introduce chat-style LLMs directly.

It introduced the architecture family that made later Transformer-based language models possible.
