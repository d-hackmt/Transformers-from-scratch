"""Token-level perplexity over a data split."""

import math

import torch
import torch.nn.functional as F

from annotated_transformer.data.batch import Batch


@torch.no_grad()
def corpus_perplexity(model, dataloader, pad_idx, device=None):
    """Return the perplexity of ``model`` on everything in ``dataloader``.

    Perplexity = ``exp(mean NLL per token)`` where NLL is the plain negative
    log-likelihood of the correct next token (padding excluded). Lower is better;
    a uniform guess over a vocab of ``V`` words scores ``V``.

    This deliberately uses ``F.nll_loss`` on the generator's log-probs rather
    than the training criterion, because
    :class:`~annotated_transformer.training.label_smoothing.LabelSmoothing`
    reports a *smoothed* loss that would inflate the perplexity.

    Also returns **token accuracy** - the fraction of non-padding target tokens
    where the model's top prediction is exactly right. (A blunt metric for
    translation, but intuitive and it climbs steadily during training.)

    Returns ``{"perplexity", "nll", "accuracy", "tokens"}``.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0

    for src, tgt in dataloader:
        if device is not None:
            src, tgt = src.to(device), tgt.to(device)
        batch = Batch(src, tgt, pad_idx)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        log_probs = model.generator(out)  # (batch, seq, vocab), already log-softmax

        flat_logp = log_probs.reshape(-1, log_probs.size(-1))
        flat_tgt = batch.tgt_y.reshape(-1)
        nll = F.nll_loss(flat_logp, flat_tgt, ignore_index=pad_idx, reduction="sum")

        keep = flat_tgt != pad_idx
        preds = flat_logp.argmax(dim=-1)
        total_correct += (preds[keep] == flat_tgt[keep]).sum().item()
        total_nll += nll.item()
        total_tokens += keep.sum().item()

    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "perplexity": math.exp(mean_nll),
        "nll": mean_nll,
        "accuracy": total_correct / max(total_tokens, 1),
        "tokens": total_tokens,
    }
