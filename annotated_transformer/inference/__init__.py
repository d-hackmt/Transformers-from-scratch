"""Decoding a trained model.

* :mod:`~annotated_transformer.inference.decode`    - ``greedy_decode``
* :mod:`~annotated_transformer.inference.translate` - batch translation + printing
"""

from annotated_transformer.inference.decode import greedy_decode
from annotated_transformer.inference.translate import translate_sentence

__all__ = ["greedy_decode", "translate_sentence"]
