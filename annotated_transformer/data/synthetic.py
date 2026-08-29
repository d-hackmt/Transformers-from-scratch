"""Synthetic data for the copy task ("A First Example" in the notebook)."""

import torch

from annotated_transformer.data.batch import Batch


def data_gen(V: int, batch_size: int, nbatches: int):
    """Yield random src == tgt batches for a src-to-tgt copy task.

    Each row is 10 random symbols in ``[1, V)`` with the first symbol forced to
    ``1`` (a start token).  The target is identical to the source, so a working
    model just has to learn to copy its input.  ``pad`` id is ``0``.
    """
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)
