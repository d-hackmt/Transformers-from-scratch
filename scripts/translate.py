"""Translate Multi30k validation sentences with a trained checkpoint.

Requires ``multi30k_model_final.pt`` (produced by ``train_multi30k.py``) and the
cached ``vocab.pt``.

Usage:
    python scripts/translate.py [--examples 5] [--checkpoint multi30k_model_final.pt]
"""

import argparse

import _path  # noqa: F401

from annotated_transformer.data.tokenizer import load_tokenizers
from annotated_transformer.data.vocab import load_vocab
from annotated_transformer.inference.translate import run_model_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--checkpoint", default="multi30k_model_final.pt")
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
