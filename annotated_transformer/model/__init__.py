"""The Transformer network.

Figure 1 of the paper, bottom to top:

    Embeddings + PositionalEncoding
        -> Encoder  (N x EncoderLayer:  self-attn, feed-forward)
        -> Decoder  (N x DecoderLayer:  self-attn, src-attn, feed-forward)
        -> Generator (linear + log-softmax)

Every piece is a plain ``nn.Module``; nothing here knows about data loading or
training.  :func:`make_model` wires the pieces together from hyperparameters.
"""

from annotated_transformer.model.attention import MultiHeadedAttention, attention
from annotated_transformer.model.build import make_model
from annotated_transformer.model.decoder import Decoder, DecoderLayer
from annotated_transformer.model.embeddings import Embeddings, PositionalEncoding
from annotated_transformer.model.encoder import Encoder, EncoderLayer
from annotated_transformer.model.encoder_decoder import EncoderDecoder, Generator
from annotated_transformer.model.feed_forward import PositionwiseFeedForward
from annotated_transformer.model.layers import LayerNorm, SublayerConnection, clones
from annotated_transformer.model.masking import subsequent_mask

__all__ = [
    "attention",
    "MultiHeadedAttention",
    "make_model",
    "Decoder",
    "DecoderLayer",
    "Embeddings",
    "PositionalEncoding",
    "Encoder",
    "EncoderLayer",
    "EncoderDecoder",
    "Generator",
    "PositionwiseFeedForward",
    "LayerNorm",
    "SublayerConnection",
    "clones",
    "subsequent_mask",
]
