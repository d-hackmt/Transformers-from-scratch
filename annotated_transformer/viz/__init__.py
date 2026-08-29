"""Altair visualizations from the notebook, adapted to save to disk.

In the notebook each of these functions returned an ``alt.Chart`` that Jupyter
rendered automatically.  Here they still return the chart, but the helper
:func:`~annotated_transformer.viz.chart_io.save_chart` (used by
``scripts/visualize.py``) writes it to an ``.html`` file - and a ``.png`` too if
``vl-convert-python`` is installed.

* :mod:`~annotated_transformer.viz.mask`            - subsequent-mask heatmap
* :mod:`~annotated_transformer.viz.positional`      - positional-encoding waves
* :mod:`~annotated_transformer.viz.schedule`        - learning-rate schedule curves
* :mod:`~annotated_transformer.viz.label_smoothing` - target distribution + penalty
* :mod:`~annotated_transformer.viz.attention`       - per-head attention maps
"""

from annotated_transformer.viz.chart_io import save_chart

__all__ = ["save_chart"]
