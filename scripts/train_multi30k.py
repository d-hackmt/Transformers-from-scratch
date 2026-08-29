"""Train the real German-to-English model on Multi30k (Part 3).

This downloads the ``bentrevett/multi30k`` dataset and the spaCy models on first
run and writes checkpoints ``multi30k_model_00.pt`` ... ``multi30k_model_final.pt``.

Heads up: this is a full training run.  On CPU it is slow; the distributed
(multi-GPU, NCCL) path only works on Linux with several NVIDIA GPUs.

Usage:
    python scripts/train_multi30k.py [--epochs 8] [--batch-size 32] [--distributed]
"""

import argparse

import _path  # noqa: F401

from annotated_transformer.data.tokenizer import load_tokenizers
from annotated_transformer.data.vocab import load_vocab
from annotated_transformer.training.trainer import default_config, train_model


def main():
    config = default_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--accum-iter", type=int, default=config["accum_iter"])
    parser.add_argument("--base-lr", type=float, default=config["base_lr"])
    parser.add_argument("--max-padding", type=int, default=config["max_padding"])
    parser.add_argument("--warmup", type=int, default=config["warmup"])
    parser.add_argument("--file-prefix", default=config["file_prefix"])
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    config.update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        accum_iter=args.accum_iter,
        base_lr=args.base_lr,
        max_padding=args.max_padding,
        warmup=args.warmup,
        file_prefix=args.file_prefix,
        distributed=args.distributed,
    )

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)


if __name__ == "__main__":
    main()
