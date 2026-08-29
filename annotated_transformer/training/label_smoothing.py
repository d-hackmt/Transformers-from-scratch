"""Label smoothing regularization (Section 5.4)."""

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """KL-divergence loss against a smoothed target distribution.

    Instead of a one-hot target, the correct token gets probability
    ``confidence = 1 - smoothing`` and the remaining ``smoothing`` mass is spread
    uniformly over the other tokens (the padding index gets 0).  This makes the
    model less over-confident: it hurts perplexity but improves accuracy / BLEU.
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None  # kept for the visualization scripts

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
