import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_head)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_values = attention_weights @ value

        combined_heads = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.output(combined_heads)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.feed_forward_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.attention_norm(x + self.attention(x))
        return self.feed_forward_norm(x + self.feed_forward(x))


class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, num_heads, d_ff, num_classes, use_positions):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.use_positions = use_positions
        self.encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)

    def embed(self, token_ids):
        embeddings = self.token_embedding(token_ids)
        if self.use_positions:
            positions = torch.arange(token_ids.size(1), device=token_ids.device)
            embeddings = embeddings + self.position_embedding(positions)
        return embeddings

    def forward(self, token_ids):
        encoder_output = self.encoder_block(self.embed(token_ids))
        return self.classifier(encoder_output[:, 0])
