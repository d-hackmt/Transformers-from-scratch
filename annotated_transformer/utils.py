"""Small helpers shared across the package.

The notebook used ``show_example`` / ``execute_example`` wrappers to decide
whether a cell should run in an interactive session.  In a plain program that
logic disappears - scripts just call functions directly - so only the genuinely
reusable pieces are kept here.
"""

import torch


class DummyOptimizer(torch.optim.Optimizer):
    """A no-op optimizer.

    Used during evaluation passes so that :func:`annotated_transformer.training.loop.run_epoch`
    can share one code path for both "train" and "eval" modes without ever
    touching the real optimizer.
    """

    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):  # noqa: D102 - no-op
        pass

    def zero_grad(self, set_to_none=False):  # noqa: D102 - no-op
        pass


class DummyScheduler:
    """A no-op learning-rate scheduler, the counterpart to :class:`DummyOptimizer`."""

    def step(self):  # noqa: D102 - no-op
        pass


def pick_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA if available (and requested), otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
