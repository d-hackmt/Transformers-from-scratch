"""Scaled dot-product attention and multi-head attention (Section 3.2)."""

import math

import torch
import torch.nn as nn

from annotated_transformer.model.layers import clones


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot-Product Attention'.

        Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k)) V

    ``mask`` (when given) marks *disallowed* query/key pairs with ``0``; those
    scores are pushed to -1e9 before the softmax so they contribute ~0 weight.
    Returns the attended values and the attention weight matrix.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention: run ``h`` attention "heads" in parallel.

    Each head projects Q/K/V into a ``d_k``-dimensional subspace, attends there,
    and the results are concatenated and projected back to ``d_model``.  With
    ``d_k = d_model / h`` the total cost matches single-head attention at full
    dimensionality.

    Implementation detail: the four ``d_model -> d_model`` linear layers are the
    three input projections (Q, K, V) plus the final output projection.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k.
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # last attention matrix, kept for visualization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2."
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Project Q, K, V in one shot, then split into h heads:
        #    (batch, seq, d_model) -> (batch, h, seq, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" the heads back together and apply the final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
