"""A map-style wrapper around the Hugging Face Multi30k dataset."""

from torch.utils.data import Dataset


class Multi30kDataset(Dataset):
    """Map-style access to one split of ``bentrevett/multi30k``.

    Each item is a ``(source_text, target_text)`` tuple.  ``src_lang`` / ``tgt_lang``
    are ``"de"`` or ``"en"`` and pick the translation direction:

    * ``src_lang="de", tgt_lang="en"``  -> German -> English (the default)
    * ``src_lang="en", tgt_lang="de"``  -> English -> German

    The split is loaded eagerly into a list so ``__getitem__`` is cheap.
    """

    def __init__(self, split: str, src_lang: str = "de", tgt_lang: str = "en"):
        from datasets import load_dataset

        assert {src_lang, tgt_lang} == {"de", "en"}, "languages must be 'de' and 'en'"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        ds = load_dataset(
            "bentrevett/multi30k", split=split, trust_remote_code=False
        )
        self.data = list(ds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return (row[self.src_lang], row[self.tgt_lang])
