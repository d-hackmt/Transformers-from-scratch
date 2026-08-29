"""Everything between raw text and a model-ready batch.

* :mod:`~annotated_transformer.data.batch`      - ``Batch`` (holds src/tgt + builds masks)
* :mod:`~annotated_transformer.data.synthetic`  - random copy-task data generator
* :mod:`~annotated_transformer.data.tokenizer`  - spaCy tokenizers
* :mod:`~annotated_transformer.data.vocab`      - ``Vocab`` + build/load helpers
* :mod:`~annotated_transformer.data.dataset`    - Multi30k dataset wrapper
* :mod:`~annotated_transformer.data.dataloader` - collate fn + ``create_dataloaders``
"""

from annotated_transformer.data.batch import Batch
from annotated_transformer.data.synthetic import data_gen

__all__ = ["Batch", "data_gen"]
