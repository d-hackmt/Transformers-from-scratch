"""Turn collected experiment metrics into Markdown reports and plots.

Input everywhere is the ``metrics`` dict that ``scripts/run_experiments.py``
builds and saves to ``results/metrics.json``::

    {
      "config": {...},                       # hyperparameters used
      "checkpoint_epochs": [10, 20, 30],
      "epochs": [                             # one row per completed epoch
        {"epoch": 1, "train_loss": ..., "val_loss": ..., "val_perplexity": ...,
         "seconds": ...},
        ...
        {"epoch": 10, ..., "bleu_valid": ..., "bleu_test": ...,
         "bleu_signature": ...},
      ],
      "samples": {"10": [{"src","ref","hyp"}, ...], "20": [...], "30": [...]}
    }
"""

import json
import os

# Use a non-interactive backend so this works in a background script with no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- io

def save_metrics(metrics: dict, path: str) -> None:
    """Write ``metrics`` to ``path`` as pretty JSON (called after every epoch)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------ plots

def make_plots(metrics: dict, out_dir: str) -> list:
    """Write loss / perplexity / BLEU plots as PNGs under ``out_dir/plots``."""
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    rows = metrics["epochs"]
    epochs = [r["epoch"] for r in rows]
    written = []

    # 1. Loss curve --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train loss")
    ax.plot(epochs, [r["val_loss"] for r in rows], marker="o", label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("label-smoothed loss / token")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    p = os.path.join(plots_dir, "loss_curve.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)

    # 2. Perplexity curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["val_perplexity"] for r in rows], marker="o", color="#22c55e")
    for ce in metrics.get("checkpoint_epochs", []):
        ax.axvline(ce, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation perplexity  (lower is better)")
    ax.set_title("Validation perplexity")
    ax.grid(alpha=0.3)
    p = os.path.join(plots_dir, "perplexity_curve.png")
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    written.append(p)

    # 2b. Accuracy + KL-snapshot curve ---------------------------------
    acc_rows = [r for r in rows if "val_accuracy" in r]
    if acc_rows:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot([r["epoch"] for r in acc_rows], [r["val_accuracy"] * 100 for r in acc_rows],
                 marker="o", color="#f59e0b", label="token accuracy %")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("token accuracy %", color="#f59e0b")
        ax2 = ax1.twinx()
        ax2.plot([r["epoch"] for r in acc_rows], [r["kl_snapshot"] for r in acc_rows],
                 marker="s", color="#a855f7", label="KL(target‖model)")
        ax2.set_ylabel("KL divergence (snapshot)", color="#a855f7")
        ax1.set_title("Token accuracy and prediction/target KL")
        ax1.grid(alpha=0.3)
        p = os.path.join(plots_dir, "accuracy_kl_curve.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p)

    # 3. BLEU bars ------------------------------------------------------
    ck = [r for r in rows if "bleu_valid" in r]
    if ck:
        labels = [f"{r['epoch']} ep" for r in ck]
        x = range(len(ck))
        width = 0.38
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([i - width / 2 for i in x], [r["bleu_valid"] for r in ck], width, label="valid")
        ax.bar([i + width / 2 for i in x], [r["bleu_test"] for r in ck], width, label="test")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel("BLEU  (higher is better)")
        ax.set_title("BLEU by training length")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        for i, r in zip(x, ck):
            ax.text(i - width / 2, r["bleu_valid"], f"{r['bleu_valid']:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, r["bleu_test"], f"{r['bleu_test']:.1f}", ha="center", va="bottom", fontsize=8)
        p = os.path.join(plots_dir, "bleu_bars.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p)

    # 4. Prediction vs target distribution (final snapshot) -----------
    dist = metrics.get("distribution")
    if dist:
        labels = dist["labels"]
        x = range(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width / 2 for i in x], dist["model"], width, label="model", color="#4f8cff")
        ax.bar([i + width / 2 for i in x], dist["target"], width, label="smoothed target", color="#22c55e")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("probability")
        ax.set_title(
            f"After '{dist.get('context_token','')}' → true next word '{dist.get('true_token','')}' "
            f"(KL={dist.get('kl',0):.3f})"
        )
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        p = os.path.join(plots_dir, "distribution_snapshot.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        written.append(p)

    return written


# ------------------------------------------------------------------- markdown

def _epoch_table(rows: list) -> str:
    head = "| epoch | train loss | val loss (KL) | perplexity | accuracy | KL snap | sec |\n"
    head += "|------:|-----------:|-------------:|-----------:|---------:|--------:|----:|\n"
    body = ""
    for r in rows:
        acc = f"{r['val_accuracy'] * 100:.1f}%" if "val_accuracy" in r else "—"
        kl = f"{r['kl_snapshot']:.3f}" if "kl_snapshot" in r else "—"
        body += (
            f"| {r['epoch']} | {r['train_loss']:.4f} | {r['val_loss']:.4f} "
            f"| {r['val_perplexity']:.2f} | {acc} | {kl} | {r['seconds']:.0f} |\n"
        )
    return head + body


def _samples_table(samples: list) -> str:
    out = "| # | source (de) | reference (en) | model output (en) |\n"
    out += "|--:|-------------|----------------|-------------------|\n"
    for i, s in enumerate(samples, 1):
        src = s["src"].replace("|", "\\|")
        ref = s["ref"].replace("|", "\\|")
        hyp = s["hyp"].replace("|", "\\|")
        out += f"| {i} | {src} | {ref} | {hyp} |\n"
    return out


def write_reports(metrics: dict, out_dir: str) -> list:
    """Write comparison.md, experiment-*-epochs.md and sample_translations.md."""
    os.makedirs(out_dir, exist_ok=True)
    rows = metrics["epochs"]
    cfg = metrics.get("config", {})
    checkpoints = [r for r in rows if "bleu_valid" in r]
    written = []

    direction = cfg.get("direction", "de-en").replace("->", "-").replace("_", "-")
    _names = {"de": "German", "en": "English"}
    _s, _t = (direction.split("-") + ["en"])[:2]
    pair = f"{_names.get(_s, _s)} -> {_names.get(_t, _t)}"
    ck_epochs = ", ".join(str(r["epoch"]) for r in checkpoints) or "-"
    tag = direction.replace("-", "_")

    # --- comparison.md -------------------------------------------------
    lines = [f"# Experiment comparison — {pair} — epochs {ck_epochs}", ""]
    lines.append(f"{pair}, Multi30k. The model was checkpointed and fully evaluated "
                 f"(loss, perplexity, accuracy, KL, BLEU valid + test, sample translations) "
                 f"at epochs {ck_epochs}.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.append("| epochs | val loss | val perplexity | BLEU (valid) | BLEU (test) |")
    lines.append("|-------:|---------:|---------------:|-------------:|------------:|")
    for r in checkpoints:
        lines.append(
            f"| {r['epoch']} | {r['val_loss']:.4f} | {r['val_perplexity']:.2f} "
            f"| {r['bleu_valid']:.2f} | {r['bleu_test']:.2f} |"
        )
    lines.append("")
    if checkpoints:
        sig = checkpoints[-1].get("bleu_signature", "")
        lines.append(f"BLEU signature: `{sig}`")
        lines.append("")
    lines.append("## Plots")
    lines.append("")
    for img, cap in [
        ("loss_curve.png", "training vs. validation loss"),
        ("perplexity_curve.png", "validation perplexity"),
        ("accuracy_kl_curve.png", "token accuracy and prediction/target KL"),
        ("bleu_bars.png", "BLEU by training length"),
        ("distribution_snapshot.png", "model vs. smoothed-target distribution (final)"),
    ]:
        lines.append(f"![{cap}](plots/{img})")
        lines.append("")
    lines.append("## Reading of the results")
    lines.append("")
    lines.append(_auto_commentary(checkpoints))
    lines.append("")
    lines.append("## Full per-epoch history")
    lines.append("")
    lines.append(_epoch_table(rows))
    p = os.path.join(out_dir, "comparison.md")
    _write(p, "\n".join(lines)); written.append(p)

    # --- experiment-N-epochs.md --------------------------------------
    samples = metrics.get("samples", {})
    for r in checkpoints:
        n = r["epoch"]
        el = [f"# Experiment: {pair}, {n} epochs", ""]
        el.append(f"Checkpoint: `checkpoints/multi30k_{tag}_{n}ep.pt`")
        el.append("")
        el.append("| metric | value |")
        el.append("|--------|------:|")
        el.append(f"| validation loss (label-smoothed) | {r['val_loss']:.4f} |")
        el.append(f"| validation perplexity | {r['val_perplexity']:.2f} |")
        el.append(f"| BLEU — validation (1014 sents) | {r['bleu_valid']:.2f} |")
        el.append(f"| BLEU — test (1000 sents) | {r['bleu_test']:.2f} |")
        el.append(f"| wall-clock to reach this epoch | {_cumulative_seconds(rows, n) / 60:.1f} min |")
        el.append("")
        el.append("## Sample translations")
        el.append("")
        el.append(_samples_table(samples.get(str(n), [])))
        pth = os.path.join(out_dir, f"experiment-{n}-epochs.md")
        _write(pth, "\n".join(el)); written.append(pth)

    # --- sample_translations.md (side by side) -----------------------
    st = [f"# Sample translations across checkpoints — {pair}", ""]
    st.append(f"The same fixed {_names.get(_s, _s)} sentences, decoded by each checkpoint.")
    st.append("")
    ref_list = samples.get(str(checkpoints[-1]["epoch"]), []) if checkpoints else []
    for i, base in enumerate(ref_list):
        st.append(f"### {i + 1}. `{base['src']}`")
        st.append("")
        st.append(f"**Reference:** {base['ref']}")
        st.append("")
        st.append("| checkpoint | output |")
        st.append("|-----------|--------|")
        for r in checkpoints:
            n = str(r["epoch"])
            hyp = samples.get(n, [])[i]["hyp"] if i < len(samples.get(n, [])) else ""
            st.append(f"| {n} epochs | {hyp} |")
        st.append("")
    p = os.path.join(out_dir, "sample_translations.md")
    _write(p, "\n".join(st)); written.append(p)

    return written


def _auto_commentary(checkpoints: list) -> str:
    if len(checkpoints) < 2:
        return "_Not enough checkpoints for a comparison._"
    first, last = checkpoints[0], checkpoints[-1]
    dppl = first["val_perplexity"] - last["val_perplexity"]
    dbleu = last["bleu_test"] - first["bleu_test"]
    best = max(checkpoints, key=lambda r: r["bleu_test"])
    parts = [
        f"- From {first['epoch']} to {last['epoch']} epochs, validation perplexity "
        f"{'fell' if dppl > 0 else 'rose'} by {abs(dppl):.2f} "
        f"({first['val_perplexity']:.2f} -> {last['val_perplexity']:.2f}).",
        f"- Test BLEU {'improved' if dbleu > 0 else 'dropped'} by {abs(dbleu):.2f} "
        f"({first['bleu_test']:.2f} -> {last['bleu_test']:.2f}).",
        f"- Best test BLEU: **{best['bleu_test']:.2f}** at {best['epoch']} epochs.",
    ]
    mid = checkpoints[len(checkpoints) // 2]
    late_gain = last["bleu_test"] - mid["bleu_test"]
    if late_gain < 0.5:
        parts.append(
            f"- Gains flatten after {mid['epoch']} epochs (only +{late_gain:.2f} BLEU "
            f"from {mid['epoch']} to {last['epoch']}) — likely near convergence / "
            f"mild overfitting on this small dataset."
        )
    return "\n".join(parts)


def _cumulative_seconds(rows: list, upto_epoch: int) -> float:
    return sum(r["seconds"] for r in rows if r["epoch"] <= upto_epoch)


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
