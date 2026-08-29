"""The encoder stack (Section 3.1, left half of Figure 1)."""

import torch.nn as nn

from annotated_transformer.model.layers import LayerNorm, SublayerConnection, clones


class Encoder(nn.Module):
    """Core encoder: a stack of ``N`` identical :class:`EncoderLayer` s.

    A final LayerNorm is applied after the last layer (needed because the
    sublayers use pre-norm residual connections).
    """

    def __init__(self, layer: nn.Module, n: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """One encoder layer: multi-head self-attention, then feed-forward.

    Each sublayer is wrapped in a residual connection + layer norm
    (see :class:`~annotated_transformer.model.layers.SublayerConnection`).
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
