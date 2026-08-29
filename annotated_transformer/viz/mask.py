"""Visualize the subsequent (causal) mask used in decoder self-attention."""

import altair as alt
import pandas as pd

from annotated_transformer.model.masking import subsequent_mask


def example_mask():
    """Heatmap: for each target position (row), which positions it may attend to.

    Bright cells (mask == 1) are allowed; dark cells are blocked future words.
    """
    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(ls_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )
