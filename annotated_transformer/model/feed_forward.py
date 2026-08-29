"""Position-wise feed-forward network (Section 3.3)."""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """The per-position MLP applied inside every encoder/decoder layer.

        FFN(x) = max(0, x W1 + b1) W2 + b2

    Same parameters at every position, different parameters per layer.
    Input/output width is ``d_model`` (512); the hidden layer is ``d_ff`` (2048).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
