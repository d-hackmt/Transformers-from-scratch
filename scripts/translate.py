"""Translate Multi30k validation sentences (German -> English) with a checkpoint.

The checkpoint (local path or a shorthand like ``de-en:10``) and ``vocab.pt`` are
pulled from the Hugging Face repo if not present locally.  This helper is
DE->EN only; for EN->DE use ``scripts/evaluate.py en-de:10``.

Usage:
    python scripts/translate.py                       # de-en:10 from HF
    python scripts/translate.py --checkpoint de-en:15 --examples 8
"""

import argparse

import _path  # noqa: F401

from annotated_transformer.data.tokenizer import load_tokenizers
from annotated_transformer.data.vocab import load_vocab
from annotated_transformer.inference.translate import run_model_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--checkpoint", default="de-en:10")
    args = parser.parse_args()

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    run_model_example(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        n_examples=args.examples,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
