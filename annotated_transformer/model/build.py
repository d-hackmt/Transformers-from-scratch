"""``make_model`` - assemble a full Transformer from hyperparameters."""

import copy

import torch.nn as nn

from annotated_transformer.model.attention import MultiHeadedAttention
from annotated_transformer.model.decoder import Decoder, DecoderLayer
from annotated_transformer.model.embeddings import Embeddings, PositionalEncoding
from annotated_transformer.model.encoder import Encoder, EncoderLayer
from annotated_transformer.model.encoder_decoder import EncoderDecoder, Generator
from annotated_transformer.model.feed_forward import PositionwiseFeedForward


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Construct a Transformer model from hyperparameters.

    Defaults are the "base" configuration from the paper (Table 3): 6 layers,
    ``d_model=512``, ``d_ff=2048``, 8 heads, 10% dropout.

    A single attention / feed-forward / positional-encoding prototype is built
    and then deep-copied wherever a fresh copy is needed, so no parameters are
    accidentally shared between layers.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important in the original code: initialize with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
