"""A tiny mutable record of training progress."""

from dataclasses import dataclass


@dataclass
class TrainState:
    """Counters tracked across an entire training run.

    * ``step``       - optimizer steps taken this epoch
    * ``accum_step`` - gradient-accumulation steps taken
    * ``samples``    - total examples seen
    * ``tokens``     - total target tokens processed
    """

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0
