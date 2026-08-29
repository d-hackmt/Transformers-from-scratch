"""Turn Multi30k text pairs into padded tensor batches."""

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from annotated_transformer.data.dataset import Multi30kDataset
from annotated_transformer.data.tokenizer import tokenize


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding: int = 128,
    pad_id: int = 2,
):
    """Collate a list of ``(de_text, en_text)`` pairs into ``(src, tgt)`` tensors.

    Each sentence becomes ``<s> tokens... </s>`` then is right-padded with
    ``pad_id`` to exactly ``max_padding`` positions, so every batch is a clean
    rectangular tensor.
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)), dtype=torch.int64, device=device
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)), dtype=torch.int64, device=device
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id)
        )
        tgt_list.append(
            pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id)
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size: int = 12000,
    max_padding: int = 128,
    is_distributed: bool = True,
):
    """Build train / validation :class:`~torch.utils.data.DataLoader` s for Multi30k.

    When ``is_distributed`` is True a :class:`DistributedSampler` is used (one
    shard per process); otherwise the loaders simply shuffle.
    """

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_dataset = Multi30kDataset("train")
    valid_dataset = Multi30kDataset("validation")

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader
