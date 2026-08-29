"""Quality metrics for a trained model.

The original notebook only ever looked at the (label-smoothed) validation loss.
This package adds the two numbers people actually report for machine translation:

* :mod:`~annotated_transformer.evaluation.perplexity` - ``corpus_perplexity``:
  ``exp(mean negative log-likelihood per token)``, using the *true* cross-entropy
  (no label smoothing).
* :mod:`~annotated_transformer.evaluation.bleu` - ``corpus_bleu``: greedy-decode a
  split and score it with ``sacrebleu``.

:mod:`~annotated_transformer.evaluation.report` turns the collected numbers into
Markdown tables and matplotlib plots.
"""

from annotated_transformer.evaluation.bleu import corpus_bleu
from annotated_transformer.evaluation.distributions import target_vs_prediction
from annotated_transformer.evaluation.perplexity import corpus_perplexity

__all__ = ["corpus_bleu", "corpus_perplexity", "target_vs_prediction"]
