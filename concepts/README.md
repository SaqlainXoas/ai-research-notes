# Concepts

This folder contains concept-first notes on core AI and ML ideas.

The goal here is to explain important topics clearly, build intuition, and connect them to papers and experiments.

## Transformer Foundations

| Concept | Main question |
| --- | --- |
| [`transformers.md`](transformers.md) | How do attention-based blocks build contextual token representations? |
| [`attention.md`](attention.md) | How does a model learn which tokens matter? |
| [`positional-encoding.md`](positional-encoding.md) | How does a Transformer represent token order and distance? |
| [`residual-connections.md`](residual-connections.md) | Why do shortcut paths make deep networks easier to optimize? |
| [`layer-normalization.md`](layer-normalization.md) | How are hidden features normalized independently at each token position? |
| [`feed-forward-networks.md`](feed-forward-networks.md) | How does a Transformer transform each token after attention mixes context? |

## Representation Learning and Retrieval

| Concept | Main question |
| --- | --- |
| [`embedding-models-and-bi-encoders.md`](embedding-models-and-bi-encoders.md) | How does an embedding model create a reusable vector for one text? |
| [`contrastive-learning.md`](contrastive-learning.md) | How do positive and negative comparisons shape a useful embedding space? |
| [`rerankers-and-cross-encoders.md`](rerankers-and-cross-encoders.md) | How does a reranker use joint text interaction to score a pair? |
