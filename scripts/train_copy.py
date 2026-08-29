"""Train the toy copy task and greedy-decode one sequence ("A First Example").

Small enough to run on a CPU in a couple of minutes.  Given random symbols the
model must reproduce them; a trained model should print ``0 1 2 3 4 5 6 7 8 9``.

Usage:
    python scripts/train_copy.py [--epochs 20] [--vocab 11] [--batch-size 80]
"""

import argparse

import _path  # noqa: F401
import torch
from torch.optim.lr_scheduler import LambdaLR

from annotated_transformer.data.synthetic import data_gen
from annotated_transformer.inference.decode import greedy_decode
from annotated_transformer.model import make_model
from annotated_transformer.training import (
    LabelSmoothing,
    SimpleLossCompute,
    rate,
    run_epoch,
)
from annotated_transformer.utils import DummyOptimizer, DummyScheduler


def train_copy(vocab_size=11, epochs=20, batch_size=80):
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(vocab_size, vocab_size, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    for _ in range(epochs):
        model.train()
        run_epoch(
            data_gen(vocab_size, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(vocab_size, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--vocab", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=80)
    args = parser.parse_args()
    train_copy(vocab_size=args.vocab, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
