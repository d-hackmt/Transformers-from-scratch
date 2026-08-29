"""Generic building blocks reused by both the encoder and the decoder."""

import copy

import torch
import torch.nn as nn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce ``n`` independent deep copies of ``module``.

    The encoder and decoder are stacks of *identical* (but not *shared*) layers,
    so we clone a prototype rather than reuse a single object.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """Layer normalization (Ba et al., 2016).

    Implemented by hand rather than with ``nn.LayerNorm`` to keep the maths
    visible: normalize each token vector to zero mean / unit std over the
    feature dimension, then apply a learned scale ``a_2`` and shift ``b_2``.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection around a sublayer, followed by layer norm.

    Note: for code simplicity the norm is applied *first* (to the sublayer
    input) rather than last, i.e. this is the "pre-norm" variant:

        output = x + dropout(sublayer(norm(x)))
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply the residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
