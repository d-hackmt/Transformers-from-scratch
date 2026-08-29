"""Snapshot of the model's predicted distribution vs. the label-smoothed target.

This is the "how much do the two distributions overlap?" view. For one fixed
target position we compare:

* what the model currently predicts for the next token (softmax over the vocab), and
* the smoothed target distribution the training criterion is pulling it toward
  (``confidence`` on the true word, ``smoothing`` spread over the rest -
  see :mod:`~annotated_transformer.training.label_smoothing`).

As training proceeds the model bars should climb toward the target bars, and the
KL divergence between them should fall.
"""

import torch

from annotated_transformer.data.batch import Batch


@torch.no_grad()
def target_vs_prediction(model, criterion, src, tgt, pad_idx, vocab_tgt, device, topk: int = 10):
    """Compare model prediction and smoothed target at the first real target position.

    Returns a dict with the token labels, the two probability vectors (restricted
    to the union of the model's top-k and the true token), and the KL divergence
    ``KL(target || model)`` over the *full* vocabulary.
    """
    model.eval()
    if device is not None:
        src, tgt = src.to(device), tgt.to(device)
    batch = Batch(src, tgt, pad_idx)
    out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
    log_probs = model.generator(out)          # (B, T, V)  log-softmax
    model_p = log_probs[0, 0].exp()           # first sequence, first position
    true_id = int(batch.tgt_y[0, 0].item())
    vocab = model_p.size(0)

    # Rebuild the smoothed target the criterion would use here.
    target_p = torch.full((vocab,), criterion.smoothing / (vocab - 2), device=model_p.device)
    target_p[true_id] = criterion.confidence
    target_p[pad_idx] = 0.0

    kl = torch.sum(
        target_p * (torch.log(target_p.clamp_min(1e-9)) - log_probs[0, 0])
    ).item()

    top_ids = torch.topk(model_p, min(topk, vocab)).indices.tolist()
    if true_id not in top_ids:
        top_ids = [true_id] + top_ids[:-1]

    return {
        "context_token": vocab_tgt.get_itos()[int(batch.tgt[0, 0].item())],
        "true_token": vocab_tgt.get_itos()[true_id],
        "labels": [vocab_tgt.get_itos()[i] for i in top_ids],
        "model": [round(model_p[i].item(), 4) for i in top_ids],
        "target": [round(target_p[i].item(), 4) for i in top_ids],
        "kl": kl,
    }
