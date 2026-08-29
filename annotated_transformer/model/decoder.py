"""The decoder stack (Section 3.1, right half of Figure 1)."""

import torch.nn as nn

from annotated_transformer.model.layers import LayerNorm, SublayerConnection, clones


class Decoder(nn.Module):
    """Generic ``N``-layer decoder with masking."""

    def __init__(self, layer: nn.Module, n: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """One decoder layer: self-attention, encoder-decoder attention, feed-forward.

    Three sublayers (vs. two in the encoder):

    1. masked self-attention over the target so far (``tgt_mask``),
    2. attention over the encoder output ``memory`` (``src_mask``),
    3. position-wise feed-forward.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        src_attn: nn.Module,
        feed_forward: nn.Module,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
