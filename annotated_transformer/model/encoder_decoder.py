"""The top-level encoder-decoder wrapper and the output generator."""

import torch.nn as nn
from torch.nn.functional import log_softmax


class EncoderDecoder(nn.Module):
    """A standard encoder-decoder architecture.

    Base class for this and many other sequence-transduction models.  It just
    holds the five components and defines how data flows through them:

        encode: src -> src_embed -> encoder -> memory
        decode: (memory, tgt) -> tgt_embed -> decoder -> hidden states

    The final projection to vocabulary logits is done separately by
    :class:`Generator` (so that training can call it once over a whole batch).
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Final linear layer + log-softmax: hidden state -> next-token log-probs."""

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
