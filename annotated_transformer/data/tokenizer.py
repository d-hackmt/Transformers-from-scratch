"""spaCy tokenizers for the German-English Multi30k example."""

import os

import spacy


def load_tokenizers():
    """Load the German and English spaCy models, downloading them if missing.

    Returns ``(spacy_de, spacy_en)``.
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text: str, tokenizer) -> list:
    """Split ``text`` into a list of token strings using a spaCy tokenizer."""
    return [tok.text for tok in tokenizer.tokenizer(text)]
