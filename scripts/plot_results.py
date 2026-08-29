"""(Re)generate the plots and Markdown reports from an existing metrics.json.

Useful if a run finished but you tweaked the report format, or a run crashed and
you want reports from the partial ``metrics.json``.

Usage:
    python scripts/plot_results.py                 # reads results/metrics.json
    python scripts/plot_results.py --out other_dir
"""

import argparse

import _path  # noqa: F401

from annotated_transformer.evaluation.report import (
    load_metrics,
    make_plots,
    write_reports,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    metrics = load_metrics(f"{args.out}/metrics.json")
    written = make_plots(metrics, args.out) + write_reports(metrics, args.out)
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
