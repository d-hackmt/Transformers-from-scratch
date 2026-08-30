"""End-to-end training for the Multi30k German-English example (Part 3).

Public entry points:

* :func:`default_config`        - the hyperparameter dict used by the notebook
* :func:`train_model`           - train (single-process or distributed)
* :func:`load_trained_model`    - train if needed, then load ``multi30k_model_final.pt``
* :func:`average`               - average several checkpoints into one model

NOTE ON HARDWARE: the distributed path uses the NCCL backend and
``torch.multiprocessing.spawn``, which require multiple NVIDIA GPUs and Linux.
On a CPU-only or Windows machine use ``config["distributed"] = False``; training
still works but is slow for a dataset this size.
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from annotated_transformer.data.batch import Batch
from annotated_transformer.data.dataloader import create_dataloaders
from annotated_transformer.model import make_model
from annotated_transformer.training.label_smoothing import LabelSmoothing
from annotated_transformer.training.loop import run_epoch
from annotated_transformer.training.loss import SimpleLossCompute
from annotated_transformer.training.schedule import rate
from annotated_transformer.training.state import TrainState
from annotated_transformer.utils import DummyOptimizer, DummyScheduler


def default_config() -> dict:
    """Return a fresh copy of the default Multi30k training configuration."""
    return {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    """Train on a single device (``gpu``); one of these runs per process."""
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")
    print(f"Train worker process using device: {device}", flush=True)

    if cuda_available:
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model = model.to(device)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    ).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        if cuda_available:
            torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        if cuda_available:
            torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    """Spawn one :func:`train_worker` process per detected GPU (NCCL / Linux)."""
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    """Train either distributed (``config["distributed"]``) or single-process."""
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


def load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config=None):
    """Return a trained model: use a local checkpoint, else pull from HF, else train."""
    from annotated_transformer.hf import resolve

    if config is None:
        config = default_config()
    model_path = "%sfinal.pt" % config["file_prefix"]

    src = None
    if os.path.exists(model_path):
        src = model_path
    else:
        try:
            src = resolve(model_path)
        except Exception:
            train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
            src = model_path

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # map_location="cpu" so a GPU-trained checkpoint also loads on a CPU-only box.
    model.load_state_dict(torch.load(src, map_location="cpu", weights_only=True))
    return model


def average(model, models):
    """Average the parameters of ``models`` into ``model`` (checkpoint ensembling)."""
    for ps in zip(*[m.parameters() for m in [model] + models]):
        ps[0].copy_(torch.sum(torch.stack(ps[1:]), dim=0) / len(ps[1:]))
