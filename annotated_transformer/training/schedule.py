"""The "Noam" learning-rate schedule (Section 5.3)."""


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """Learning-rate multiplier for :class:`~torch.optim.lr_scheduler.LambdaLR`.

        lr = factor * model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    This ramps the rate up linearly for ``warmup`` steps, then decays it as the
    inverse square root of the step number.  ``step`` is forced to at least 1 so
    ``LambdaLR`` (which starts at step 0) does not raise 0 to a negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
