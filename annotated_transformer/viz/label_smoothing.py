"""Visualize what label smoothing does to the target distribution."""

import altair as alt
import pandas as pd
import torch

from annotated_transformer.training.label_smoothing import LabelSmoothing


def example_label_smoothing():
    """Heatmap of the smoothed target distribution for five example targets."""
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.tensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ],
        dtype=torch.float32,
    )
    crit(
        x=torch.log(predict),
        target=torch.tensor([2, 1, 0, 3, 3], dtype=torch.long),
    )
    ls_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(ls_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color("target distribution:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


def _loss(x, crit):
    d = x + 3 * 1
    predict = torch.tensor(
        [[0, x / d, 1 / d, 1 / d, 1 / d]], dtype=torch.float32
    )
    return crit(torch.log(predict), torch.tensor([1], dtype=torch.long)).data


def penalization_visualization():
    """Line chart showing that label smoothing penalizes over-confident guesses."""
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [_loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(x="Steps", y="Loss")
        .interactive()
    )
