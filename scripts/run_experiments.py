"""Train the German -> English Multi30k model once and evaluate it at 10/20/30 epochs.

Because the learning-rate schedule is purely step-based (fixed 3000-step warmup),
the first N epochs of a 30-epoch run are identical to a standalone N-epoch run.
So a single run, checkpointed at epochs 10/20/30, gives the same comparison as
three separate runs for half the compute.

Every epoch it logs train loss, validation loss and validation perplexity to
``<out>/metrics.json`` (rewritten each epoch, so a crash still leaves data).
At each checkpoint epoch it also saves the model and computes BLEU on the
validation and test splits, plus a fixed set of sample translations.

GPU run (recommended, ~1 hour on a 6 GB card):
    python scripts/run_experiments.py --epochs 30

CPU run (the full model is far too slow on CPU - shrink it):
    python scripts/run_experiments.py --epochs 10 \
        --n-layers 2 --d-model 128 --d-ff 512 --heads 4 --batch-size 16

Outputs go to <out>/ (default: results_de_en/ or results_ende/ by --direction):
    <out>/metrics.json, train.log, live.html
    <out>/checkpoints/multi30k_<dir>_{N}ep.pt        (also uploaded to Hugging Face)
    <out>/comparison.md, experiment-*-epochs.md, sample_translations.md
    <out>/plots/*.png
"""

import argparse
import json
import os
import random
import time
import traceback

import _path  # noqa: F401
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from annotated_transformer.data.batch import Batch
from annotated_transformer.data.dataloader import create_dataloaders
from annotated_transformer.data.dataset import Multi30kDataset
from annotated_transformer.data.tokenizer import load_tokenizers
from annotated_transformer.data.vocab import load_vocab
from annotated_transformer.evaluation.bleu import corpus_bleu
from annotated_transformer.evaluation.distributions import target_vs_prediction
from annotated_transformer.evaluation.perplexity import corpus_perplexity
from annotated_transformer.evaluation.report import make_plots, save_metrics, write_reports
from annotated_transformer.inference.translate import translate_sentence
from annotated_transformer.model import make_model
from annotated_transformer.training import (
    LabelSmoothing,
    SimpleLossCompute,
    rate,
    run_epoch,
)
from annotated_transformer.training.hardware import (
    CooldownAbort,
    cooldown,
    gpu_stats,
    save_atomic,
)
from annotated_transformer.training.monitor import TrainingMonitor
from annotated_transformer.utils import DummyOptimizer, DummyScheduler, pick_device

N_SAMPLES = 8  # how many validation sentences to show as qualitative examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_samples(model, pairs, vocab_src, vocab_tgt, spacy_src, device, max_len):
    """Translate the source side of each (src, tgt) pair; return {src, ref, hyp}."""
    rows = []
    for src_text, ref_text in pairs:
        hyp = translate_sentence(model, src_text, vocab_src, vocab_tgt, spacy_src, device, max_len=max_len)
        rows.append({"src": src_text, "ref": ref_text, "hyp": hyp})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-iter", type=int, default=10)
    parser.add_argument("--base-lr", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--max-padding", type=int, default=72)
    # model size (defaults = paper "base"; shrink for CPU)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="output dir (default: results_de_en / results_ende by --direction)")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    parser.add_argument("--max-bleu-sentences", type=int, default=0,
                        help="cap BLEU decoding for a quick check (0 = use the whole split)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out>/checkpoints/_resume.pt if it exists")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="finish this epoch (incl. checkpoint/BLEU) then stop cleanly; resume later with --resume")
    # --- crash / overheat protection ---------------------------------
    parser.add_argument("--cooldown-seconds", type=int, default=0,
                        help="pause this long after every epoch so the PC's power/thermals recover")
    parser.add_argument("--temp-limit", type=int, default=80,
                        help="GPU temp (C) above which the cooldown pause is doubled")
    parser.add_argument("--temp-abort", type=int, default=90,
                        help="GPU temp (C) at/above which the run checkpoints and stops")
    parser.add_argument("--direction", choices=["de-en", "en-de"], default="de-en",
                        help="translation direction on Multi30k (default de-en)")
    args = parser.parse_args()

    src_lang, tgt_lang = args.direction.split("-")
    if args.out is None:
        args.out = "results_de_en" if args.direction == "de-en" else "results_ende"
    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Device: {device}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.json")

    # --- data -----------------------------------------------------------
    spacy_de, spacy_en = load_tokenizers()
    vocab_de, vocab_en = load_vocab(spacy_de, spacy_en)
    # map the language vocabs / tokenizer onto source / target for this direction
    vocab_src, vocab_tgt = (vocab_de, vocab_en) if src_lang == "de" else (vocab_en, vocab_de)
    spacy_src = spacy_de if src_lang == "de" else spacy_en
    pad_idx = vocab_tgt["<blank>"]
    print(f"Direction: {args.direction}  (src {src_lang} vocab {len(vocab_src)}, "
          f"tgt {tgt_lang} vocab {len(vocab_tgt)})", flush=True)

    train_dl, valid_dl = create_dataloaders(
        device, vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=args.batch_size, max_padding=args.max_padding, is_distributed=False,
        direction=args.direction,
    )
    valid_pairs = list(Multi30kDataset("validation", src_lang, tgt_lang))
    test_pairs = list(Multi30kDataset("test", src_lang, tgt_lang))
    sample_pairs = valid_pairs[:N_SAMPLES]  # fixed sentences for qualitative tracking
    if args.max_bleu_sentences:
        valid_pairs = valid_pairs[: args.max_bleu_sentences]
        test_pairs = test_pairs[: args.max_bleu_sentences]

    # one fixed batch, reused every epoch for the "prediction vs target" snapshot
    snap_src, snap_tgt = next(iter(valid_dl))

    # --- model / optimizer -------------------------------------------
    model = make_model(
        len(vocab_src), len(vocab_tgt),
        N=args.n_layers, d_model=args.d_model, d_ff=args.d_ff,
        h=args.heads, dropout=args.dropout,
    ).to(device)
    criterion = LabelSmoothing(len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step, args.d_model, factor=1, warmup=args.warmup))

    resume_path = os.path.join(ckpt_dir, "_resume.pt")
    start_epoch = 1
    prior_metrics = None
    if args.resume and os.path.exists(resume_path):
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        try:
            torch.set_rng_state(state["rng_cpu"])
            if torch.cuda.is_available() and state.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception:
            pass
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                prior_metrics = json.load(f)
        print(f"Resuming from epoch {start_epoch} (loaded {resume_path})", flush=True)

    monitor = TrainingMonitor(
        args.out, total_epochs=args.epochs, fresh=(start_epoch == 1),
        steps_per_epoch=len(train_dl),
    )
    monitor.log(f"device={device}  model N={args.n_layers} d_model={args.d_model} "
                f"d_ff={args.d_ff} heads={args.heads}  batch={args.batch_size}")
    monitor.log(f"direction: {args.direction}  src vocab={len(vocab_src)}  tgt vocab={len(vocab_tgt)}")

    metrics = {
        "config": {
            "direction": args.direction, "dataset": "bentrevett/multi30k",
            "epochs": args.epochs, "checkpoint_epochs": args.checkpoint_epochs,
            "batch_size": args.batch_size, "accum_iter": args.accum_iter,
            "base_lr": args.base_lr, "warmup": args.warmup, "max_padding": args.max_padding,
            "n_layers": args.n_layers, "d_model": args.d_model, "d_ff": args.d_ff,
            "heads": args.heads, "dropout": args.dropout, "seed": args.seed,
            "device": str(device),
            "vocab_src": len(vocab_src), "vocab_tgt": len(vocab_tgt),
        },
        "checkpoint_epochs": args.checkpoint_epochs,
        "epochs": [],
        "samples": {},
    }
    if prior_metrics is not None:
        # keep the already-computed epochs / samples when resuming
        metrics["epochs"] = prior_metrics.get("epochs", [])
        metrics["samples"] = prior_metrics.get("samples", {})
        metrics["distribution"] = prior_metrics.get("distribution")
    monitor.render(metrics)  # so the dashboard shows the config from the first tick

    total_start = time.time()
    try:
        _train_loop(
            args, model, criterion, optimizer, scheduler, monitor, metrics,
            metrics_path, resume_path, ckpt_dir, train_dl, valid_dl, pad_idx,
            vocab_src, vocab_tgt, spacy_src, valid_pairs, test_pairs, sample_pairs,
            snap_src, snap_tgt, device, start_epoch,
        )
    except KeyboardInterrupt:
        monitor.log("interrupted — resume point from the last completed epoch is safe. rerun with --resume.")
        return
    except Exception:
        monitor.log("CRASH:\n" + traceback.format_exc())
        raise

    last_epoch = metrics["epochs"][-1]["epoch"] if metrics["epochs"] else 0
    fully_done = last_epoch >= args.epochs
    if fully_done:
        metrics["total_minutes"] = round((time.time() - total_start) / 60, 1)
        save_metrics(metrics, metrics_path)
        monitor.render(metrics)

    # always (re)generate the report + plots for whatever epochs we have
    monitor.log("writing reports and plots ...")
    written = make_plots(metrics, args.out) + write_reports(metrics, args.out)
    for w in written:
        monitor.log(f"  wrote {w}")
    if fully_done:
        monitor.log(f"DONE in {metrics['total_minutes']} min. See {args.out}/comparison.md")
    else:
        monitor.log(f"PAUSED at epoch {last_epoch}/{args.epochs}. reports for epochs 1-{last_epoch} "
                    f"written. resume with --resume. See {args.out}/comparison.md")


def _train_loop(
    args, model, criterion, optimizer, scheduler, monitor, metrics,
    metrics_path, resume_path, ckpt_dir, train_dl, valid_dl, pad_idx,
    vocab_src, vocab_tgt, spacy_src, valid_pairs, test_pairs, sample_pairs,
    snap_src, snap_tgt, device, start_epoch,
):
    for epoch in range(start_epoch, args.epochs + 1):
        monitor.set_epoch(epoch)
        monitor.log(f"===== Epoch {epoch}/{args.epochs} =====")
        t0 = time.time()

        model.train()
        train_loss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dl),
            model, SimpleLossCompute(model.generator, criterion),
            optimizer, scheduler, mode="train+log", accum_iter=args.accum_iter,
            log_fn=monitor.log,
        )

        model.eval()
        val_loss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dl),
            model, SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(), DummyScheduler(), mode="eval",
        )
        ppl = corpus_perplexity(model, valid_dl, pad_idx, device)
        dist = target_vs_prediction(model, criterion, snap_src, snap_tgt, pad_idx, vocab_tgt, device)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_perplexity": ppl["perplexity"],
            "val_accuracy": ppl["accuracy"],
            "kl_snapshot": dist["kl"],
            "seconds": round(time.time() - t0, 1),
        }
        metrics["distribution"] = dist  # latest prediction-vs-target snapshot
        monitor.log(
            f"epoch {epoch} done: train {row['train_loss']:.4f} | "
            f"val {row['val_loss']:.4f} | perplexity {row['val_perplexity']:.2f} | "
            f"acc {row['val_accuracy'] * 100:.1f}% | KL {row['kl_snapshot']:.3f} | "
            f"{row['seconds']:.0f}s"
        )
        hw = gpu_stats()
        if hw:
            row["gpu_temp_c"] = hw["temp_c"]
            row["gpu_power_w"] = hw["power_w"]
            monitor.log(f"GPU after epoch {epoch}: {hw['temp_c']:.0f} C, {hw['power_w']:.0f} W, {hw['mem_mb']:.0f} MiB")

        if epoch in args.checkpoint_epochs:
            tag = args.direction.replace("-", "_")  # de_en or en_de
            ckpt = os.path.join(ckpt_dir, f"multi30k_{tag}_{epoch}ep.pt")
            save_atomic(model.state_dict(), ckpt, keep_prev=False)
            monitor.log(f"checkpoint saved: {ckpt}  —  scoring BLEU (this takes a few minutes) ...")

            bv = corpus_bleu(model, valid_pairs, vocab_src, vocab_tgt, spacy_src, device, max_len=args.max_padding)
            bt = corpus_bleu(model, test_pairs, vocab_src, vocab_tgt, spacy_src, device, max_len=args.max_padding)
            row["bleu_valid"] = round(bv["bleu"], 2)
            row["bleu_test"] = round(bt["bleu"], 2)
            row["bleu_signature"] = bv["signature"]
            monitor.log(f"BLEU @ {epoch}ep — valid {row['bleu_valid']} | test {row['bleu_test']}")

            metrics["samples"][str(epoch)] = decode_samples(
                model, sample_pairs, vocab_src, vocab_tgt,
                spacy_src, device, args.max_padding,
            )

        metrics["epochs"].append(row)
        save_metrics(metrics, metrics_path)
        monitor.render(metrics)

        # crash-safe resume point, written atomically (keeps a .prev fallback)
        save_atomic(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            resume_path,
        )
        monitor.log(f"resume point saved (epoch {epoch}). safe to stop or lose power here.")

        # planned pause: stop cleanly after this epoch, continue later with --resume
        if args.stop_after and epoch >= args.stop_after:
            monitor.log(
                f"PAUSED after epoch {epoch} (--stop-after {args.stop_after}). "
                f"resume with:  python scripts/run_experiments.py ... --resume"
            )
            return

        # let the machine's power/thermals recover before the next epoch
        if epoch < args.epochs:
            try:
                cooldown(args.cooldown_seconds, log_fn=monitor.log,
                         temp_limit=args.temp_limit, temp_abort=args.temp_abort)
            except CooldownAbort as exc:
                monitor.log(f"ABORTING: {exc}. resume point is safe — rerun with --resume once cool.")
                raise KeyboardInterrupt  # handled in main(): clean exit, reports still written from partial data


if __name__ == "__main__":
    main()
