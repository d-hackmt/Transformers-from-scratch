"""``run_epoch`` - one pass over a data iterator, for training or evaluation."""

import time

from annotated_transformer.training.state import TrainState


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = None,
    log_fn=print,
):
    """Train (or evaluate) the model for a single epoch.

    ``mode`` is one of ``"train"``, ``"train+log"`` or ``"eval"``.  In eval mode
    pass a :class:`~annotated_transformer.utils.DummyOptimizer` /
    :class:`~annotated_transformer.utils.DummyScheduler` so the parameter-update
    branch is a no-op.

    Gradients are accumulated over ``accum_iter`` batches before each optimizer
    step.  Returns ``(average_loss_per_token, train_state)``.

    ``log_fn`` receives the periodic progress string (default ``print``); pass
    :meth:`~annotated_transformer.training.monitor.TrainingMonitor.log` to also
    tee it to a file and the live dashboard.
    """
    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            log_fn(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state
