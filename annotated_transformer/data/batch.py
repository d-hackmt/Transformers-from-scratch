"""The ``Batch`` object: holds one batch of data and its masks."""

from annotated_transformer.model.masking import subsequent_mask


class Batch:
    """Holds a batch of source / target sentences plus the masks training needs.

    Given padded ``src`` and ``tgt`` integer tensors of shape ``(batch, seq)``:

    * ``src_mask``  - ``True`` where ``src`` is a real (non-pad) token.
    * ``tgt``       - target input, i.e. ``tgt[:, :-1]`` (teacher forcing).
    * ``tgt_y``     - target output to predict, i.e. ``tgt[:, 1:]`` (shifted by one).
    * ``tgt_mask``  - combines the pad mask with the subsequent (causal) mask.
    * ``ntokens``   - number of real target tokens, used to normalize the loss.
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask hiding both padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
