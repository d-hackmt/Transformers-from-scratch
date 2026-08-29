"""Training machinery: loss, LR schedule, the epoch loop, and the trainer.

* :mod:`~annotated_transformer.training.state`           - ``TrainState`` counters
* :mod:`~annotated_transformer.training.schedule`        - Noam LR schedule (``rate``)
* :mod:`~annotated_transformer.training.label_smoothing` - ``LabelSmoothing`` criterion
* :mod:`~annotated_transformer.training.loss`            - ``SimpleLossCompute``
* :mod:`~annotated_transformer.training.loop`            - ``run_epoch``
* :mod:`~annotated_transformer.training.trainer`         - end-to-end Multi30k training
"""

from annotated_transformer.training.label_smoothing import LabelSmoothing
from annotated_transformer.training.loop import run_epoch
from annotated_transformer.training.loss import SimpleLossCompute
from annotated_transformer.training.schedule import rate
from annotated_transformer.training.state import TrainState

__all__ = [
    "LabelSmoothing",
    "run_epoch",
    "SimpleLossCompute",
    "rate",
    "TrainState",
]
