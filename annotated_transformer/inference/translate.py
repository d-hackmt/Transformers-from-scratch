"""Translate validation examples with a trained model and print the results."""

import torch

from annotated_transformer.data.batch import Batch
from annotated_transformer.data.dataloader import create_dataloaders
from annotated_transformer.data.tokenizer import tokenize
from annotated_transformer.inference.decode import greedy_decode
from annotated_transformer.model import make_model


@torch.no_grad()
def translate_sentence(
    model,
    sentence: str,
    vocab_src,
    vocab_tgt,
    spacy_src,
    device,
    max_len: int = 72,
    pad_idx: int = 2,
    bos: int = 0,
    eos: int = 1,
) -> str:
    """Translate one raw source string to a target string (greedy decoding).

    Steps: tokenize -> look up ids -> wrap in ``<s> ... </s>`` -> greedy-decode
    -> map ids back to words, dropping ``<s>`` and stopping at ``</s>``.
    """
    model.eval()
    ids = [bos] + vocab_src(tokenize(sentence, spacy_src)) + [eos]
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = (src != pad_idx).unsqueeze(-2)

    out = greedy_decode(model, src, src_mask, max_len, start_symbol=bos, eos_token=eos)

    words = []
    for token_id in out[0].tolist():
        if token_id == bos:
            continue
        if token_id == eos:
            break
        words.append(vocab_tgt.get_itos()[token_id])
    return " ".join(words)


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples: int = 15,
    pad_idx: int = 2,
    eos_string: str = "</s>",
):
    """Greedy-decode the first ``n_examples`` validation sentences and print them.

    Returns a list of ``(batch, src_tokens, tgt_tokens, model_out_ids, model_text)``
    tuples so callers (e.g. the attention-visualization scripts) can reuse the
    decoded batch.
    """
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[int(x)] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[int(x)] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : " + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[int(x)] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(
    vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples: int = 5,
    checkpoint: str = "multi30k_model_final.pt",
):
    """Load ``checkpoint`` on CPU and run :func:`check_outputs` on Multi30k validation."""
    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=True)
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data
