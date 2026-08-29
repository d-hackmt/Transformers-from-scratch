"""Score a single trained checkpoint: perplexity + BLEU + sample translations.

Usage:
    python scripts/evaluate.py results/checkpoints/multi30k_de_en_30ep.pt
    python scripts/evaluate.py <ckpt> --split test --n-layers 6 --d-model 512

The model-size flags must match how the checkpoint was trained (defaults = base).
"""

import argparse

import _path  # noqa: F401
import torch

from annotated_transformer.data.dataloader import create_dataloaders
from annotated_transformer.data.dataset import Multi30kDataset
from annotated_transformer.data.tokenizer import load_tokenizers
from annotated_transformer.data.vocab import load_vocab
from annotated_transformer.evaluation.bleu import corpus_bleu
from annotated_transformer.evaluation.perplexity import corpus_perplexity
from annotated_transformer.inference.translate import translate_sentence
from annotated_transformer.model import make_model
from annotated_transformer.utils import pick_device

SAMPLES = [
    "Zwei Hunde spielen im Schnee.",
    "Eine Frau kocht in der Küche.",
    "Ein Mann fährt mit dem Fahrrad die Straße entlang.",
    "Ein kleines Mädchen läuft über eine Wiese.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-padding", type=int, default=72)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Device: {device}")

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    pad_idx = vocab_tgt["<blank>"]

    model = make_model(
        len(vocab_src), len(vocab_tgt),
        N=args.n_layers, d_model=args.d_model, d_ff=args.d_ff, h=args.heads,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # perplexity needs a dataloader (uses the validation split)
    _, valid_dl = create_dataloaders(
        device, vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=32, max_padding=args.max_padding, is_distributed=False,
    )
    ppl = corpus_perplexity(model, valid_dl, pad_idx, device)
    print(f"\nValidation perplexity: {ppl['perplexity']:.2f}  (nll {ppl['nll']:.4f}, {ppl['tokens']} tokens)")

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    for split in splits:
        pairs = list(Multi30kDataset(split))
        res = corpus_bleu(model, pairs, vocab_src, vocab_tgt, spacy_de, device, max_len=args.max_padding)
        print(f"BLEU ({split}, {res['n']} sents): {res['bleu']:.2f}   {res['signature']}")

    print("\nSample translations:")
    for s in SAMPLES:
        print(f"  DE: {s}")
        print(f"  EN: {translate_sentence(model, s, vocab_src, vocab_tgt, spacy_de, device, max_len=args.max_padding)}")


if __name__ == "__main__":
    main()
