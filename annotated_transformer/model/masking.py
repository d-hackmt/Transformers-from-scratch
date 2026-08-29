"""Attention masks that belong to the model (as opposed to data padding masks)."""

import torch


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions for decoder self-attention.

    Returns a ``(1, size, size)`` boolean tensor where entry ``[0, i, j]`` is
    ``True`` iff position ``i`` is allowed to attend to position ``j`` - that is,
    iff ``j <= i``.  This enforces the auto-regressive property: the prediction
    for position ``i`` can only depend on known outputs at positions ``< i``.
    """
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.bool)
    return subsequent == 0
