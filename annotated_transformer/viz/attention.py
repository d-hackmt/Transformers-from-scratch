"""Visualize the attention matrices of a trained model, layer by layer, head by head.

After a forward pass, each :class:`~annotated_transformer.model.attention.MultiHeadedAttention`
module stashes its last attention matrix on ``.attn``.  The getters below pull
those out; :func:`visualize_layer` turns a selection of heads into a column of
heatmaps and :func:`get_all_attention_charts` runs the whole example end to end.
"""

import altair as alt
import pandas as pd

from annotated_transformer.inference.translate import run_model_example


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "Convert a dense attention matrix into a tidy DataFrame with token labels."
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    "Heatmap of a single (layer, head) attention matrix."
    df = mtx2df(attn[0, head].data, max_dim, max_dim, row_tokens, col_tokens)
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    "Stack heads 1, 3, 5, 7 of one layer into a single vertical chart."
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(attn, 0, h, row_tokens=row_tokens, col_tokens=col_tokens, max_dim=ntokens)
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0] | charts[2] | charts[4] | charts[6]
        # layer + 1 because humans count layers from 1
    ).properties(title="Layer %d" % (layer + 1))


def _viz_stack(model, example, getter_fn, layers, use_target_cols):
    """Build charts for the given ``layers`` and hconcat them.

    ``example`` is one tuple from
    :func:`~annotated_transformer.inference.translate.check_outputs`;
    ``example[1]`` is the source tokens, ``example[2]`` the target tokens.

    For encoder/decoder self-attention the columns are the source tokens; for
    decoder source-attention (``use_target_cols=True``) the query rows are source
    tokens and the key columns are target tokens.
    """
    row_tokens = example[1]
    col_tokens = example[2] if use_target_cols else example[1]
    ntokens = (
        max(len(example[1]), len(example[2])) if use_target_cols else len(example[1])
    )
    layer_viz = [
        visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens)
        for layer in layers
    ]
    chart = layer_viz[0]
    for lv in layer_viz[1:]:
        chart = chart & lv
    return alt.hconcat(chart)


def get_all_attention_charts(vocab_src, vocab_tgt, spacy_de, spacy_en,
                             checkpoint="de-en:10"):
    """Run one validation example and return the three attention charts.

    Returns a dict ``{"encoder_self": ..., "decoder_self": ..., "decoder_src": ...}``.
    Matches the notebook's selection: encoder self-attention shows layers 1/3/5,
    the two decoder views show all six layers.
    """
    model, example_data = run_model_example(
        vocab_src, vocab_tgt, spacy_de, spacy_en, n_examples=1, checkpoint=checkpoint
    )
    example = example_data[-1]  # the final decoded example
    return {
        "encoder_self": _viz_stack(
            model, example, get_encoder, layers=[0, 2, 4], use_target_cols=False
        ),
        "decoder_self": _viz_stack(
            model, example, get_decoder_self, layers=range(6), use_target_cols=False
        ),
        "decoder_src": _viz_stack(
            model, example, get_decoder_src, layers=range(6), use_target_cols=True
        ),
    }
