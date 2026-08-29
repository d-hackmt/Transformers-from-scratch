"""Visualize the sinusoidal positional encoding."""

import altair as alt
import pandas as pd
import torch

from annotated_transformer.model.embeddings import PositionalEncoding


def example_positional():
    """Line chart of four positional-encoding dimensions vs. position.

    Each dimension is a sine/cosine wave with a different frequency and phase.
    """
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )
