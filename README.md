# The Annotated Transformer — modular edition

A refactor of Harvard NLP's [`AnnotatedTransformer.ipynb`](AnnotatedTransformer.ipynb)
(an implementation of *Attention Is All You Need*, Vaswani et al. 2017) into a
plain, importable Python package with small command-line scripts.

The notebook is kept unchanged for reference; all of its code now lives in
`annotated_transformer/`, split by concern.

> **New to the Transformer?** Read [`docs/`](docs/README.md) first — every
> component explained in order, in plain words, with diagrams, worked examples on
> one sentence (`ein hund spielt` → `a dog plays`), and links to the relevant
> papers.

---

## Learn more

- 🎓 **[Join our Advanced AI Engineering batch](https://www.krishnaik.in/liveclass2/Advanded-Route?id=12)**
  — a hands-on live cohort where we build things exactly like this, end to end.
- 📖 **[Learn the Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
  — the original Harvard NLP line-by-line walkthrough this repo is refactored from.
- 📄 **[Explore the paper](https://arxiv.org/abs/1605.00459)**
  — *Multi30K: Multilingual English-German Image Descriptions* (Elliott et al., 2016),
  the dataset behind the German→English example here.
- 🗂️ **[Explore the dataset](https://huggingface.co/datasets/bentrevett/multi30k)**
  — `bentrevett/multi30k` on Hugging Face, loaded automatically by the training scripts.

---

<img width="1612" height="973" alt="image" src="https://github.com/user-attachments/assets/21df4ddd-8217-4074-ac57-22d7bb0e155c" />

<img width="1878" height="553" alt="image" src="https://github.com/user-attachments/assets/4407a888-4c4d-44c4-8aef-294d37265558" />

<img width="1071" height="758" alt="image" src="https://github.com/user-attachments/assets/c401d007-ef85-41e9-962c-e5052fd81839" />

---

## Install

```bash
pip install -r requirements.txt

# spaCy models for the Multi30k example (Part 3 only):
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

No packaging step — the scripts add the repo root to `sys.path` themselves
(see `scripts/_path.py`), so just run them from anywhere.

---

## Package layout

```
annotated_transformer/
├── model/                     the network (pure nn.Module code)
│   ├── layers.py              clones, LayerNorm, SublayerConnection
│   ├── attention.py           attention(), MultiHeadedAttention
│   ├── feed_forward.py        PositionwiseFeedForward
│   ├── embeddings.py          Embeddings, PositionalEncoding
│   ├── encoder.py             Encoder, EncoderLayer
│   ├── decoder.py             Decoder, DecoderLayer
│   ├── encoder_decoder.py     EncoderDecoder, Generator
│   ├── masking.py             subsequent_mask
│   └── build.py               make_model(...)
├── data/
│   ├── batch.py               Batch (builds src/tgt masks)
│   ├── synthetic.py           data_gen  (copy-task data)
│   ├── tokenizer.py           load_tokenizers, tokenize
│   ├── vocab.py               Vocab, build_vocabulary, load_vocab
│   ├── dataset.py             Multi30kDataset
│   └── dataloader.py          collate_batch, create_dataloaders
├── training/
│   ├── state.py               TrainState
│   ├── schedule.py            rate()  (Noam LR schedule)
│   ├── label_smoothing.py     LabelSmoothing
│   ├── loss.py                SimpleLossCompute
│   ├── loop.py                run_epoch  (+ optional log_fn)
│   ├── monitor.py             live train.log + auto-refreshing <out>/live.html
│   └── trainer.py             train_model, load_trained_model, average, default_config
├── inference/
│   ├── decode.py              greedy_decode
│   └── translate.py           check_outputs, run_model_example, translate_sentence
├── evaluation/                (added — not in the notebook)
│   ├── perplexity.py          corpus_perplexity  (true token cross-entropy)
│   ├── bleu.py                corpus_bleu  (greedy decode + sacrebleu)
│   └── report.py              metrics.json → Markdown tables + matplotlib plots
├── viz/
│   ├── chart_io.py            save_chart (html + optional png)
│   ├── mask.py / positional.py / schedule.py / label_smoothing.py
│   └── attention.py           per-head attention heatmaps
├── utils.py                   DummyOptimizer, DummyScheduler, pick_device
└── hf.py                     resolve() — checkpoints/vocab: local first, else Hugging Face

scripts/
├── smoke_test.py              untrained forward/decode sanity check
├── train_copy.py              toy copy task ("A First Example")
├── train_multi30k.py          full DE→EN training (Part 3)
├── run_experiments.py         30-epoch run, eval at 10/20/30, live dashboard
├── evaluate.py                score one checkpoint (perplexity + BLEU + samples)
├── plot_results.py            rebuild plots/reports from <out>/metrics.json
├── translate.py               decode validation sentences with a checkpoint
└── visualize.py               regenerate the notebook charts as files

docs/                          component-by-component explainer (start at docs/README.md)
results_de_en/  results_ende/  run outputs (reports + plots committed; checkpoints on HF)
```

---

## The pipeline

```
raw text ──tokenizer──▶ tokens ──vocab──▶ ids ──collate_batch──▶ padded tensors
                                                                      │
                                                              Batch (+ masks)
                                                                      │
   make_model ─▶ EncoderDecoder ──run_epoch(loss=SimpleLossCompute,   │
                                            criterion=LabelSmoothing, ◀┘
                                            sched=LambdaLR(rate))
                                                                      │
                                                             checkpoint .pt
                                                                      │
                                             greedy_decode / check_outputs
                                                                      │
                                                          translations, attention maps
```

---

## Usage

### Sanity check (no training)
```bash
python scripts/smoke_test.py
```
Builds a random model and decodes a sequence. Output is gibberish by design;
it only proves the shapes and the forward/decode path work.

### Toy copy task — runs on CPU
```bash
python scripts/train_copy.py --epochs 20
```
Trains a 2-layer model to echo its input. A trained model prints
`0 1 2 3 4 5 6 7 8 9`.

### Multi30k German → English (Part 3)
```bash
python scripts/train_multi30k.py --epochs 8 --batch-size 32
```
Downloads the dataset + spaCy models on first run, caches the vocab to
`vocab.pt`, and writes `multi30k_model_00.pt … multi30k_model_final.pt`.

> **Hardware.** This is a real training run. On CPU it is slow. The
> `--distributed` flag uses the NCCL backend with `torch.multiprocessing.spawn`
> and needs multiple NVIDIA GPUs on Linux; the default single-process path works
> everywhere.

### Translate with a trained checkpoint
```bash
python scripts/translate.py --examples 5
```

### Regenerate the charts
```bash
python scripts/visualize.py all          # or: mask | positional | schedule
                                         #     label-smoothing | attention
```
Writes `outputs/*.html` (and `*.png` if `vl-convert-python` is installed).
`attention` needs a trained `multi30k_model_final.pt`.

---

## Experiments — 10 vs 20 vs 30 epochs, with BLEU & perplexity

Full walkthrough (GPU **and** CPU): [`docs/19-running-experiments.md`](docs/19-running-experiments.md).
What the metrics mean: [`docs/18-evaluation-bleu-and-perplexity.md`](docs/18-evaluation-bleu-and-perplexity.md).

### Setup

```bash
# GPU (driver >= 560 -> cu126, >= 570 -> cu128):
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
# CPU only:
uv pip install torch

uv pip install -r requirements.txt
python -m spacy download de_core_news_sm en_core_web_sm
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Run the study

```bash
# German -> English, 30-epoch run, checkpointed + evaluated at epochs 10/20/30
python scripts/run_experiments.py --epochs 30                       # -> results_de_en/

# English -> German
python scripts/run_experiments.py --direction en-de --epochs 15 \
    --checkpoint-epochs 5 10 15                                     # -> results_ende/

# CPU — the full model is impractical; train a small one instead
python scripts/run_experiments.py --epochs 10 --checkpoint-epochs 3 6 10 \
    --n-layers 2 --d-model 128 --d-ff 512 --heads 4 --batch-size 16 --warmup 400
```

Output goes to `results_de_en/` or `results_ende/` (by `--direction`), or wherever
`--out` points.

### Watch it live

While it runs, open **`<out>/live.html`** (e.g. `results_de_en/live.html`) in a
browser — it refreshes itself every 10 s with the metrics table, GPU temp,
loss/accuracy/perplexity/KL curves, and latest sample translations. Or tail the
text log: `tail -f <out>/train.log`.

### Outputs (in `results_de_en/` or `results_ende/`)

| File | Contents |
|------|----------|
| `live.html` | auto-refreshing dashboard (during the run) |
| `train.log` | full timestamped log |
| `metrics.json` | every epoch's numbers (train/val loss, perplexity, accuracy, KL, BLEU) |
| `comparison.md` | **the summary** — table + plots + commentary |
| `experiment-{N}-epochs.md` | per-checkpoint detail + sample translations |
| `sample_translations.md` | same sentences, each checkpoint side by side |
| `plots/*.png` | loss / accuracy / perplexity / KL / BLEU / GPU-temp |
| `checkpoints/multi30k_<dir>_{N}ep.pt` | saved models — **gitignored, mirrored on Hugging Face** |

The reports/plots/logs are committed to this repo; the checkpoints live on
[Hugging Face](https://huggingface.co/Diveshj/transformer-from-scratch) (see below).

### Score or use a saved checkpoint later

```bash
python scripts/evaluate.py de-en:10   # perplexity + BLEU + samples for the epoch-10 DE->EN model
python scripts/evaluate.py en-de:15   # ... epoch-15 EN->DE model
python scripts/plot_results.py        # rebuild reports from metrics.json
```

---

## Trained checkpoints (Hugging Face)

The trained models and the vocab cache are too big for git, so they live in a
Hugging Face model repo:
**[`Diveshj/transformer-from-scratch`](https://huggingface.co/Diveshj/transformer-from-scratch)**
(public — no token needed).

```
results_deen/checkpoints/multi30k_de_en_{10,15}ep.pt     German -> English
results_ende/checkpoints/multi30k_en_de_{5,10,15}ep.pt   English -> German
vocab.pt                                                 shared vocab cache
```

Nothing to download by hand — [`annotated_transformer/hf.py`](annotated_transformer/hf.py)
resolves paths **local-first, then HF**. If `results*/checkpoints/*.pt` and
`vocab.pt` aren't on disk they are fetched from the repo and cached in
`~/.cache/huggingface/`. So you can delete the local copies and everything still
runs:

```bash
python scripts/evaluate.py de-en:10        # shorthand -> downloaded from HF
python scripts/translate.py --checkpoint en-de:10
```

```python
from annotated_transformer.hf import resolve
import torch
path = resolve("en-de:15")   # local file, or downloaded from HF
state = torch.load(path, map_location="cpu", weights_only=True)
```

---

## Using it as a library

```python
from annotated_transformer.model import make_model
from annotated_transformer.data.batch import Batch
from annotated_transformer.inference.decode import greedy_decode

model = make_model(src_vocab=32000, tgt_vocab=32000, N=6)   # paper "base" config
```

---

## Differences from the notebook

- The `show_example` / `is_interactive_notebook` gating is gone; scripts call
  functions directly.
- Notebook globals (`vocab_src`, `spacy_de`, `model`, …) are now explicit
  function arguments and return values.
- `viz` functions save charts to disk instead of returning them for Jupyter to
  render.
- `GPUtil.showUtilization()` (a debug print in the training loop) was dropped,
  removing that dependency.
- `average()` was fixed to actually stack-and-mean the checkpoints.
- `TrainState` is a `@dataclass`.
- **Added** (not in the notebook): `annotated_transformer/evaluation/` (BLEU +
  perplexity), `scripts/run_experiments.py` / `evaluate.py` / `plot_results.py`,
  a live log + `<out>/live.html` dashboard, and `docs/`.
- `run_epoch` gained an optional `log_fn=print` argument so progress can be teed
  to the log/dashboard; default behaviour is unchanged.

Nothing about the model maths or the training procedure changed.
