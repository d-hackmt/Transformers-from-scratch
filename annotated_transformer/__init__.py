"""The Annotated Transformer, as a modular Python package.

This package is a line-by-line refactor of the Harvard NLP
``AnnotatedTransformer.ipynb`` notebook (an implementation of
"Attention Is All You Need", Vaswani et al. 2017).

The notebook's single flat namespace has been split into focused
sub-packages:

* :mod:`annotated_transformer.model`     - the network itself (pure ``nn.Module`` code)
* :mod:`annotated_transformer.data`      - batching, masking, tokenizers, vocab, datasets
* :mod:`annotated_transformer.training`  - loss, LR schedule, training loop, trainer
* :mod:`annotated_transformer.inference` - greedy decoding and translation helpers
* :mod:`annotated_transformer.viz`       - Altair charts, saved to disk instead of shown

Runnable entry points live in the top-level ``scripts/`` directory.
"""

from annotated_transformer.model import make_model

__all__ = ["make_model"]
