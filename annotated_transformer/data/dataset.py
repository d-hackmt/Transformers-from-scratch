"""A map-style wrapper around the Hugging Face Multi30k dataset."""

from torch.utils.data import Dataset


class Multi30kDataset(Dataset):
    """Map-style access to one split of ``bentrevett/multi30k``.

    Each item is a ``(german_text, english_text)`` tuple.  The split is loaded
    eagerly into a list so ``__getitem__`` is cheap.
    """

    def __init__(self, split: str):
        from datasets import load_dataset

        ds = load_dataset(
            "bentrevett/multi30k", split=split, trust_remote_code=False
        )
        self.data = list(ds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx]["de"], self.data[idx]["en"])
