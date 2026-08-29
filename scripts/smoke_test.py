"""Sanity check: build an untrained model and run greedy decoding.

Mirrors the notebook's ``inference_test`` / ``run_tests``.  The output is
gibberish (the model is random) - the point is only that shapes line up and a
forward/decode pass runs without error.

Usage:
    python scripts/smoke_test.py [--runs 10]
"""

import argparse

import _path  # noqa: F401  (adds repo root to sys.path)
import torch

from annotated_transformer.model import make_model, subsequent_mask


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.full((1, 1), next_word, dtype=ys.dtype)], dim=1)

    print("Example Untrained Model Prediction:", ys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    for _ in range(args.runs):
        inference_test()


if __name__ == "__main__":
    main()
