"""Token embeddings and sinusoidal positional encoding (Sections 3.4 - 3.5)."""

import math

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Learned token embedding, scaled by ``sqrt(d_model)``.

    The scaling keeps the embedding magnitude comparable to the positional
    encoding that gets added to it.
    """

    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Inject absolute position information via fixed sine/cosine waves.

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Each embedding dimension is a sinusoid whose wavelength grows geometrically
    from 2*pi to 10000*2*pi.  Computed once up to ``max_len`` and stored as a
    buffer (moves with the model, but is not a trainable parameter).
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
