"""Signed spatial attribution and pairwise interaction heatmaps."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ._config import BLUE, LINES, NEUTRAL, RED

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from shapiq.interaction_values import InteractionValues

__all__ = ["interaction_heatmap_plot", "region_attribution_plot"]


def _order_values(interaction_values: InteractionValues, order: int) -> np.ma.MaskedArray:
    """Keep explicitly stored zeros distinct from unrecorded entries."""
    n_players = interaction_values.n_players
    shape = (n_players,) * order
    values = np.zeros(shape)
    present = np.zeros(shape, dtype=bool)
    for interaction, value in interaction_values.dict_values.items():
        if len(interaction) != order:
            continue
        if len(set(interaction)) != order or any(not 0 <= i < n_players for i in interaction):
            msg = f"Invalid interaction {interaction} for {n_players} players."
            raise ValueError(msg)
        if order == 2:
            if present[interaction] and values[interaction] != value:
                msg = f"Conflicting values for pair {tuple(sorted(interaction))}."
                raise ValueError(msg)
            reverse = interaction[::-1]
            values[reverse] = value
            present[reverse] = True
        values[interaction] = value
        present[interaction] = True
    if not present.any():
        msg = f"No order-{order} interaction values available to plot."
        raise ValueError(msg)
    return np.ma.array(values, mask=~present)


def _plot_signed_heatmap(
    values: np.ma.MaskedArray,
    *,
    ax: Axes | None,
    max_score: float | None,
    colorbar: bool,
    colorbar_label: str,
) -> Axes:
    """Draw values with a symmetric scale and a distinct shade for unavailable values."""
    recorded = values.compressed()
    if not np.isfinite(recorded).all():
        msg = "The interaction values to plot must be finite."
        raise ValueError(msg)
    max_abs = float(np.max(np.abs(recorded)))
    if max_score is not None and (not np.isfinite(max_score) or max_score <= 0):
        msg = "max_score must be finite and strictly positive."
        raise ValueError(msg)
    if max_score is not None and max_score < max_abs:
        msg = "max_score must be at least the largest absolute value to display."
        raise ValueError(msg)
    limit = max_score if max_score is not None else (max_abs or 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "shapiq_signed", [BLUE.hex, NEUTRAL.hex, RED.hex], N=257
    ).with_extremes(bad=LINES.hex)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    heatmap = ax.imshow(
        values,
        norm=Normalize(vmin=-limit, vmax=limit),
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    if colorbar:
        fig = cast("Figure", ax.get_figure())
        ticks = [0.0] if max_score is None and max_abs == 0 else None
        fig.colorbar(heatmap, ax=ax, label=colorbar_label, ticks=ticks)
    return ax


def _region_boundaries(regions: np.ndarray) -> LineCollection:
    """Trace the edges between adjacent pixels with different labels."""
    vertical_rows, vertical_cols = np.nonzero(regions[:, :-1] != regions[:, 1:])
    vertical_edges = [
        ((col + 0.5, row - 0.5), (col + 0.5, row + 0.5))
        for row, col in zip(vertical_rows, vertical_cols, strict=True)
    ]
    horizontal_rows, horizontal_cols = np.nonzero(regions[:-1] != regions[1:])
    horizontal_edges = [
        ((col - 0.5, row + 0.5), (col + 0.5, row + 0.5))
        for row, col in zip(horizontal_rows, horizontal_cols, strict=True)
    ]
    return LineCollection(vertical_edges + horizontal_edges, colors=LINES.hex, linewidths=0.6)


def region_attribution_plot(
    interaction_values: InteractionValues,
    regions: np.ndarray,
    *,
    ax: Axes | None = None,
    max_score: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    show: bool = False,
) -> Axes | None:
    """Draw first-order attributions in the spatial layout of a region map.

    Every pixel has the value of its region. Positive contributions are red, negative
    contributions are blue, and explicit zeros are white. Regions with no recorded first-order
    value are gray.

    Args:
        interaction_values: An explanation with at least one first-order contribution.
        regions: A nonempty 2D integer array using all region IDs from zero to
            ``interaction_values.n_players - 1``. Regions may have irregular shapes.
        ax: Axes to draw on. If omitted, create an axes in a new figure.
        max_score: Positive symmetric color limit, at least as large as every absolute value
            displayed. If omitted, scale to the largest absolute value. Use the same limit
            across plots to compare colors directly.
        colorbar: Whether to add a colorbar.
        colorbar_label: Custom label for the colorbar, such as a name and unit.
        show: Whether to call ``plt.show()`` instead of returning the axes.

    Returns:
        The plot axes if ``show=False``, otherwise ``None``.

    Raises:
        ValueError: If the regions do not match the explanation, first-order values are absent
            or nonfinite, or the supplied color limit is invalid.

    Examples:
        >>> from shapiq import InteractionValues
        >>> scores = InteractionValues(
        ...     {(0,): 0.3, (1,): -0.2}, index="SV", n_players=2, min_order=1, max_order=1
        ... )
        >>> ax = region_attribution_plot(scores, np.array([[0, 0], [0, 1]]))
    """
    regions = np.asarray(regions)
    if (
        regions.ndim != 2
        or regions.size == 0
        or not np.issubdtype(regions.dtype, np.integer)
        or not np.array_equal(np.unique(regions), np.arange(interaction_values.n_players))
    ):
        msg = "regions must be a nonempty 2D integer array with one label for every player."
        raise ValueError(msg)
    scores = _order_values(interaction_values, order=1)
    ax = _plot_signed_heatmap(
        scores[regions],
        ax=ax,
        max_score=max_score,
        colorbar=colorbar,
        colorbar_label="Region contribution" if colorbar_label is None else colorbar_label,
    )
    ax.add_collection(_region_boundaries(regions), autolim=False)
    ax.set_axis_off()
    if show:
        plt.show()
        return None
    return ax


def interaction_heatmap_plot(
    interaction_values: InteractionValues,
    *,
    feature_names: Sequence[str] | None = None,
    ax: Axes | None = None,
    max_score: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    show: bool = False,
) -> Axes | None:
    """Draw pairwise interaction values for one explanation as a symmetric matrix.

    Each cell shows the stored interaction of its row and column features. The diagonal and
    unrecorded pairs are gray, while an explicitly stored zero is white. Blues and reds encode
    negative and positive interactions. Either orientation of a pair can be stored. When both
    orientations are present, their values must agree.

    Args:
        interaction_values: An explanation with at least one pairwise interaction.
        feature_names: Optional labels in player order. Defaults to the player indices.
        ax: Axes to draw on. If omitted, create an axes in a new figure.
        max_score: Positive symmetric color limit, at least as large as every absolute value
            displayed. If omitted, scale to the largest absolute value. Use the same limit
            across plots to compare colors directly.
        colorbar: Whether to add a colorbar.
        colorbar_label: Custom label for the colorbar, such as a name and unit.
        show: Whether to call ``plt.show()`` instead of returning the axes.

    Returns:
        The plot axes if ``show=False``, otherwise ``None``.

    Raises:
        ValueError: If the feature names do not match the number of players, pairwise values
            are absent, nonfinite, or conflicting, or the supplied color limit is invalid.

    Examples:
        >>> from shapiq import InteractionValues
        >>> scores = InteractionValues(
        ...     {(0, 1): 0.3}, index="k-SII", n_players=2, min_order=2, max_order=2
        ... )
        >>> ax = interaction_heatmap_plot(scores, feature_names=["Feature A", "Feature B"])
    """
    n_players = interaction_values.n_players
    names = [str(i) for i in range(n_players)] if feature_names is None else list(feature_names)
    if len(names) != n_players:
        msg = "The number of feature names must match the number of players."
        raise ValueError(msg)
    ax = _plot_signed_heatmap(
        _order_values(interaction_values, order=2),
        ax=ax,
        max_score=max_score,
        colorbar=colorbar,
        colorbar_label="Pairwise interaction" if colorbar_label is None else colorbar_label,
    )
    ax.set_xticks(range(n_players), names)
    ax.set_yticks(range(n_players), names)
    if feature_names is not None:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if show:
        plt.show()
        return None
    return ax
