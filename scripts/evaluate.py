"""Score a single trained checkpoint: perplexity + BLEU + sample translations.

The checkpoint can be:
  * a local path   -> results/checkpoints/multi30k_de_en_15ep.pt
  * a shorthand    -> de-en:10   /   en-de:15    (direction is inferred)
Anything not found locally is downloaded from the Hugging Face repo
(Diveshj/transformer-from-scratch) and cached.

Usage:
    python scripts/evaluate.py de-en:10
    python scripts/evaluate.py en-de:15
    python scripts/evaluate.py results_ende/checkpoints/multi30k_en_de_10ep.pt --direction en-de

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
from annotated_transformer.hf import direction_of, resolve
from annotated_transformer.inference.translate import translate_sentence
from annotated_transformer.model import make_model
from annotated_transformer.utils import pick_device

SAMPLES = {
    "de": [
        "Zwei Hunde spielen im Schnee.",
        "Eine Frau kocht in der Küche.",
        "Ein Mann fährt mit dem Fahrrad die Straße entlang.",
        "Ein kleines Mädchen läuft über eine Wiese.",
    ],
    "en": [
        "Two dogs are playing in the snow.",
        "A woman is cooking in the kitchen.",
        "A man is riding a bicycle down the street.",
        "A little girl is running across a meadow.",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="local path, or shorthand like 'de-en:10' / 'en-de:15'")
    parser.add_argument("--direction", choices=["de-en", "en-de"], default=None,
                        help="inferred from the checkpoint name if omitted")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-padding", type=int, default=72)
    parser.add_argument("--max-bleu-sentences", type=int, default=0,
                        help="cap BLEU decoding for a quick check (0 = whole split)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    direction = args.direction or direction_of(args.checkpoint) or "de-en"
    ckpt_path = resolve(args.checkpoint)
    src_lang, tgt_lang = direction.split("-")
    device = torch.device("cpu") if args.cpu else pick_device()
    print(f"Device: {device}  |  direction: {direction}  |  checkpoint: {ckpt_path}")

    spacy_de, spacy_en = load_tokenizers()
    vocab_de, vocab_en = load_vocab(spacy_de, spacy_en)
    vocab_src, vocab_tgt = (vocab_de, vocab_en) if src_lang == "de" else (vocab_en, vocab_de)
    spacy_src = spacy_de if src_lang == "de" else spacy_en
    pad_idx = vocab_tgt["<blank>"]

    model = make_model(
        len(vocab_src), len(vocab_tgt),
        N=args.n_layers, d_model=args.d_model, d_ff=args.d_ff, h=args.heads,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    _, valid_dl = create_dataloaders(
        device, vocab_src, vocab_tgt, spacy_de, spacy_en,
        batch_size=32, max_padding=args.max_padding, is_distributed=False,
        direction=direction,
    )
    ppl = corpus_perplexity(model, valid_dl, pad_idx, device)
    print(f"\nValidation perplexity: {ppl['perplexity']:.2f}  "
          f"(nll {ppl['nll']:.4f}, accuracy {ppl['accuracy'] * 100:.1f}%, {ppl['tokens']} tokens)")

    splits = ["validation", "test"] if args.split == "both" else [args.split]
    for split in splits:
        pairs = list(Multi30kDataset(split, src_lang, tgt_lang))
        if args.max_bleu_sentences:
            pairs = pairs[: args.max_bleu_sentences]
        res = corpus_bleu(model, pairs, vocab_src, vocab_tgt, spacy_src, device, max_len=args.max_padding)
        print(f"BLEU ({split}, {res['n']} sents): {res['bleu']:.2f}   {res['signature']}")

    print("\nSample translations:")
    for s in SAMPLES[src_lang]:
        out = translate_sentence(model, s, vocab_src, vocab_tgt, spacy_src, device, max_len=args.max_padding)
        print(f"  {src_lang.upper()}: {s}")
        print(f"  {tgt_lang.upper()}: {out}")


if __name__ == "__main__":
    main()
