"""Check the emitted region and pairwise heatmaps against known interaction values."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from shapiq import InteractionValues, interaction_heatmap_plot, region_attribution_plot
from shapiq.plot import (
    interaction_heatmap_plot as plot_package_interaction_heatmap_plot,
    region_attribution_plot as plot_package_region_attribution_plot,
)


@pytest.fixture
def scores():
    return InteractionValues(
        {(0,): 0.4, (1,): -0.3, (2,): 0.0, (0, 1): -0.2, (1, 2): 0.0, (0, 1, 2): 0.9},
        index="SII",
        n_players=3,
        min_order=1,
        max_order=3,
    )


def test_region_values_and_boundaries_follow_irregular_segments(scores):
    assert region_attribution_plot is plot_package_region_attribution_plot
    regions = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 1]])
    fig, ax = plt.subplots()
    try:
        assert region_attribution_plot(scores, regions, ax=ax, colorbar=False) is ax
        np.testing.assert_array_equal(
            ax.images[0].get_array(), [[0.4, 0.4, -0.3], [0.4, -0.3, -0.3], [0.0, 0.0, -0.3]]
        )
        assert ax.images[0].get_interpolation() == "nearest"
        # An L-shaped region has only local boundaries where pixel labels change.
        segments = {tuple(map(tuple, segment)) for segment in ax.collections[0].get_segments()}
        assert segments == {
            ((1.5, -0.5), (1.5, 0.5)),
            ((0.5, 0.5), (0.5, 1.5)),
            ((1.5, 1.5), (1.5, 2.5)),
            ((0.5, 0.5), (1.5, 0.5)),
            ((-0.5, 1.5), (0.5, 1.5)),
            ((0.5, 1.5), (1.5, 1.5)),
        }
        assert fig.axes == [ax]
    finally:
        plt.close(fig)


def test_interaction_heatmap_distinguishes_zero_and_missing_pairs(scores):
    assert interaction_heatmap_plot is plot_package_interaction_heatmap_plot
    fig, ax = plt.subplots()
    try:
        interaction_heatmap_plot(scores, feature_names=["a", "b", "c"], ax=ax, colorbar=False)
        image = ax.images[0]
        values = image.get_array()
        np.testing.assert_array_equal(
            values.mask, [[True, False, True], [False, True, False], [True, False, True]]
        )
        assert values[0, 1] == values[1, 0] == -0.2
        assert values[1, 2] == values[2, 1] == 0.0
        assert image.cmap(image.norm(0.0)) == to_rgba("white")
        assert image.cmap.get_bad().tolist() != list(to_rgba("white"))
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b", "c"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["a", "b", "c"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("pair_values", [{(0, 1): 0.7}, {(1, 0): 0.7}, {(0, 1): 0.7, (1, 0): 0.7}])
def test_interaction_heatmap_mirrors_both_pair_orientations(pair_values):
    scores = InteractionValues(pair_values, index="SII", n_players=2, min_order=2, max_order=2)
    fig, ax = plt.subplots()
    try:
        interaction_heatmap_plot(scores, ax=ax, colorbar=False)
        values = ax.images[0].get_array()
        np.testing.assert_array_equal(values.mask, [[True, False], [False, True]])
        assert values[0, 1] == values[1, 0] == 0.7
    finally:
        plt.close(fig)


def test_interaction_heatmap_rejects_conflicting_pair_orientations():
    scores = InteractionValues(
        {(0, 1): 0.7, (1, 0): 0.0}, index="SII", n_players=2, min_order=2, max_order=2
    )
    with pytest.raises(ValueError, match="Conflicting values for pair"):
        interaction_heatmap_plot(scores)


def test_region_heatmap_distinguishes_zero_and_missing_attributions():
    sparse_scores = InteractionValues(
        {(1,): 0.0}, index="SV", n_players=2, min_order=1, max_order=1
    )
    fig, ax = plt.subplots()
    try:
        region_attribution_plot(sparse_scores, np.array([[0, 1]]), ax=ax, colorbar=False)
        values = ax.images[0].get_array()
        np.testing.assert_array_equal(values.mask, [[True, False]])
        assert values[0, 1] == 0
    finally:
        plt.close(fig)


def test_manually_shared_symmetric_scale_and_colorbar_label(scores):
    fig, (region_ax, matrix_ax) = plt.subplots(1, 2)
    try:
        kwargs = {"max_score": 0.5, "colorbar_label": "Score [unit]"}
        region_attribution_plot(scores, np.array([[0, 1, 2]]), ax=region_ax, **kwargs)
        interaction_heatmap_plot(scores, ax=matrix_ax, **kwargs)
        for ax in (region_ax, matrix_ax):
            image = ax.images[0]
            assert image.get_clim() == (-0.5, 0.5)
            np.testing.assert_allclose(image.cmap(image.norm(-0.5)), to_rgba("#1e88e5"))
            np.testing.assert_allclose(image.cmap(image.norm(0.5)), to_rgba("#ff0d57"))
        assert [ax.get_ylabel() for ax in fig.axes[2:]] == ["Score [unit]", "Score [unit]"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("max_score", [0, -1, np.inf, np.nan, 0.1])
def test_color_limit_must_be_finite_positive_and_cover_display_values(scores, max_score):
    with pytest.raises(ValueError, match="max_score"):
        interaction_heatmap_plot(scores, max_score=max_score)


@pytest.mark.parametrize(
    "regions", [np.array([0, 1, 2]), np.array([[0.0, 1.0, 2.0]]), np.array([[0, 2, 3]])]
)
def test_invalid_region_maps_raise(scores, regions):
    with pytest.raises(ValueError, match="regions"):
        region_attribution_plot(scores, regions)


def test_missing_order_and_bad_feature_labels_are_reported(scores):
    with pytest.raises(ValueError, match="number of feature names"):
        interaction_heatmap_plot(scores, feature_names=["a", "b"])
    with pytest.raises(ValueError, match="No order-2"):
        interaction_heatmap_plot(scores.get_n_order(1))
    with pytest.raises(ValueError, match="No order-1"):
        region_attribution_plot(scores.get_n_order(2), np.array([[0, 1, 2]]))


def test_nonfinite_recorded_values_raise():
    scores = InteractionValues({(0, 1): np.nan}, index="SII", n_players=2, min_order=2, max_order=2)
    with pytest.raises(ValueError, match="values to plot must be finite"):
        interaction_heatmap_plot(scores)


def test_all_zero_interactions_have_neutral_fill(scores):
    zero_scores = InteractionValues(
        {(1, 2): 0.0}, index="SII", n_players=scores.n_players, min_order=2, max_order=2
    )
    fig, ax = plt.subplots()
    try:
        interaction_heatmap_plot(zero_scores, ax=ax)
        image = ax.images[0]
        assert image.cmap(image.norm(0.0)) == to_rgba("white")
        np.testing.assert_array_equal(fig.axes[1].get_yticks(), [0.0])
    finally:
        plt.close(fig)


def test_show_delegates_to_pyplot(scores, monkeypatch):
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    fig, ax = plt.subplots()
    try:
        assert interaction_heatmap_plot(scores, ax=ax, colorbar=False, show=True) is None
        assert called == [True]
    finally:
        plt.close(fig)
