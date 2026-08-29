"""A minimal vocabulary, plus helpers to build/cache one for Multi30k.

This is a stand-in for ``torchtext.vocab`` (which recent notebook copies dropped).
The four special tokens are fixed at the front:

    0 = <s>   1 = </s>   2 = <blank> (pad)   3 = <unk>
"""

from collections import Counter
from os.path import exists

import torch

from annotated_transformer.data.tokenizer import tokenize

SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]


class Vocab:
    """Simple string<->index mapping with an <unk> fallback.

    ``stoi`` maps token -> id, ``itos`` maps id -> token.  Calling the object on
    a list of tokens returns their ids; unknown tokens map to ``default_idx``
    (3 = ``<unk>`` by default).
    """

    def __init__(self, stoi: dict, itos: dict, default_idx: int = 3):
        self.stoi = stoi
        self.itos = itos
        self.default_idx = default_idx

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_idx)

    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_idx) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def set_default_index(self, idx):
        self.default_idx = idx


def _build_one(counter: Counter) -> Vocab:
    """Turn a token frequency counter into a :class:`Vocab` (specials first)."""
    stoi = {w: i for i, w in enumerate(SPECIALS)}
    for w, _ in counter.most_common():
        if w not in stoi:
            stoi[w] = len(stoi)
    itos = {i: w for w, i in stoi.items()}
    return Vocab(stoi, itos)


def build_vocabulary(spacy_de, spacy_en):
    """Build the German and English vocabularies from Multi30k.

    Returns ``(vocab_de, vocab_en)`` - always in that order.  The caller decides
    which is source and which is target based on the translation direction.
    """
    from datasets import load_dataset

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German / English vocabularies ...")
    ds = load_dataset("bentrevett/multi30k", trust_remote_code=False)
    counter_de, counter_en = Counter(), Counter()
    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            counter_de.update(tokenize_de(ex["de"]))
            counter_en.update(tokenize_en(ex["en"]))

    return _build_one(counter_de), _build_one(counter_en)


def load_vocab(spacy_de, spacy_en, cache_path: str = "vocab.pt"):
    """Load ``(vocab_de, vocab_en)`` from ``cache_path``, building + saving if absent."""
    if not exists(cache_path):
        vocab_de, vocab_en = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_de, vocab_en), cache_path)
    else:
        vocab_de, vocab_en = torch.load(
            cache_path, map_location="cpu", weights_only=False
        )
    print("Finished.\nVocabulary sizes:")
    print("  de:", len(vocab_de))
    print("  en:", len(vocab_en))
    return vocab_de, vocab_en
