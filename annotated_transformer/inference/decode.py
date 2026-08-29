"""Greedy decoding (the simplest possible search)."""

import torch

from annotated_transformer.model.masking import subsequent_mask


def greedy_decode(model, src, src_mask, max_len: int, start_symbol: int, eos_token=None):
    """Decode one sentence by always taking the highest-probability next token.

    Encodes ``src`` once, then repeatedly runs the decoder, appending the argmax
    token each step until ``max_len`` tokens have been produced.  Returns the
    generated id sequence (including the initial ``start_symbol``).

    ``eos_token`` is an optional optimisation: when given, decoding stops as soon
    as that token is produced (the notebook always ran the full ``max_len``).
    This roughly triples BLEU-scoring speed and does not change the output.
    """
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        if eos_token is not None and next_word == eos_token:
            break
    return ys
