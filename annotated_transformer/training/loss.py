"""Glue between the model's generator and the training criterion."""


class SimpleLossCompute:
    """Run the generator, compute the (normalized) loss, and report both.

    Called as ``loss_compute(x, y, norm)`` where ``x`` is the decoder output,
    ``y`` is ``batch.tgt_y`` and ``norm`` is ``batch.ntokens``.

    Returns ``(loss_for_logging, loss_node_for_backward)``:

    * the first is a detached scalar already multiplied back up by ``norm``
      (so callers can accumulate a total and divide by total tokens),
    * the second is the per-token loss tensor to call ``.backward()`` on.
    """

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
