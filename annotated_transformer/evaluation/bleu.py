"""Corpus BLEU via greedy decoding + sacrebleu."""

from annotated_transformer.inference.translate import translate_sentence


def corpus_bleu(
    model,
    pairs,
    vocab_src,
    vocab_tgt,
    spacy_src,
    device,
    max_len: int = 72,
    progress_every: int = 250,
):
    """Greedy-decode every source sentence in ``pairs`` and score against the refs.

    ``pairs`` is an iterable of ``(source_text, reference_text)`` - exactly what
    :class:`~annotated_transformer.data.dataset.Multi30kDataset` yields.

    BLEU (BiLingual Evaluation Understudy) measures n-gram overlap between the
    model's output and the reference translation, from 0 (no overlap) to 100
    (identical). We use ``sacrebleu`` so the score is computed with a fixed,
    published tokenization and is comparable across runs and papers.

    Returns ``{"bleu", "signature", "n", "hyps", "refs"}``.
    """
    from sacrebleu.metrics import BLEU

    pairs = list(pairs)
    hyps, refs = [], []
    for i, (src_text, ref_text) in enumerate(pairs):
        hyps.append(
            translate_sentence(
                model, src_text, vocab_src, vocab_tgt, spacy_src, device,
                max_len=max_len,
            )
        )
        refs.append(ref_text)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    BLEU: decoded {i + 1}/{len(pairs)}", flush=True)

    metric = BLEU()
    score = metric.corpus_score(hyps, [refs])  # sacrebleu 2.x API
    return {
        "bleu": score.score,
        "signature": str(metric.get_signature()),
        "n": len(pairs),
        "hyps": hyps,
        "refs": refs,
    }
