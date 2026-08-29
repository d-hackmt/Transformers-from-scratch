"""Generate the notebook's charts as files under ``outputs/``.

Each chart is saved as ``.html`` (always) and ``.png`` (if ``vl-convert-python``
is installed).

Usage:
    python scripts/visualize.py mask
    python scripts/visualize.py positional
    python scripts/visualize.py schedule
    python scripts/visualize.py label-smoothing
    python scripts/visualize.py attention          # needs a trained checkpoint
    python scripts/visualize.py all
"""

import argparse

import _path  # noqa: F401

from annotated_transformer.viz import save_chart

OUT_DIR = "outputs"


def _simple_charts():
    """The four charts that need no trained model. Imported lazily per choice."""
    from annotated_transformer.viz.label_smoothing import (
        example_label_smoothing,
        penalization_visualization,
    )
    from annotated_transformer.viz.mask import example_mask
    from annotated_transformer.viz.positional import example_positional
    from annotated_transformer.viz.schedule import example_learning_schedule

    return {
        "mask": [("subsequent_mask", example_mask)],
        "positional": [("positional_encoding", example_positional)],
        "schedule": [("learning_rate_schedule", example_learning_schedule)],
        "label-smoothing": [
            ("label_smoothing_dist", example_label_smoothing),
            ("label_smoothing_penalty", penalization_visualization),
        ],
    }


def _run_simple(keys):
    charts = _simple_charts()
    for key in keys:
        for name, fn in charts[key]:
            written = save_chart(fn(), f"{OUT_DIR}/{name}")
            print("wrote", *written)


def _run_attention():
    from annotated_transformer.data.tokenizer import load_tokenizers
    from annotated_transformer.data.vocab import load_vocab
    from annotated_transformer.viz.attention import get_all_attention_charts

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    charts = get_all_attention_charts(vocab_src, vocab_tgt, spacy_de, spacy_en)
    for name, chart in charts.items():
        written = save_chart(chart, f"{OUT_DIR}/attention_{name}")
        print("wrote", *written)


def main():
    choices = ["mask", "positional", "schedule", "label-smoothing", "attention", "all"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=choices)
    args = parser.parse_args()

    simple_keys = ["mask", "positional", "schedule", "label-smoothing"]
    if args.what == "all":
        _run_simple(simple_keys)
        _run_attention()
    elif args.what == "attention":
        _run_attention()
    else:
        _run_simple([args.what])


if __name__ == "__main__":
    main()
