"""Verify plots and exports for region games with few players."""

from __future__ import annotations

import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from experiments.oxford_pets_uncertainty.reporting import (
    explanation_plot_limits,
    export_explanations,
    plot_epistemic_comparison,
    plot_region_explanation,
)

from shapiq import ExactComputer
from shapiq_games.probly import ImageProblyGame, ProblyPredictor


def _result_for(regions):
    predictor = ProblyPredictor(
        lambda images: images,
        output="brightness",
        quantifier=lambda images: {"brightness": images.mean(axis=(1, 2, 3)) / 255},
    )
    image = np.full((*regions.shape, 3), 255, dtype=np.uint8)
    game = ImageProblyGame(image, predictor, regions=regions, fill="black", normalize=False)
    coalitions = np.asarray(list(product((False, True), repeat=game.n_players)))
    values = game(coalitions)
    computer = ExactComputer(game)
    exact = {"SV": computer(index="SV", order=1)}
    if game.n_players > 1:
        exact["k-SII"] = computer(index="k-SII", order=2)
    result = {
        "target": 0,
        "true_class_index": 0,
        "coalitions": coalitions,
        "values": dict.fromkeys(("P", "TU", "AU", "EU"), values),
        "exact": dict.fromkeys(("P", "TU", "AU", "EU"), exact),
        "games": {},
    }
    return game, result


def test_single_region_shows_only_shapley_values_and_exports_region_names(tmp_path):
    game, result = _result_for(np.zeros((4, 4), dtype=int))
    results = {("pet", "full", "ensemble"): result}
    limits = explanation_plot_limits(results)
    assert limits.epistemic_pairs is None

    names = ["Background"]
    figure = plot_region_explanation(
        "pet", game, result, ["Breed"], limits, region_mode="trimap", region_names=names
    )
    try:
        figure.canvas.draw()
        assert not figure.axes[5].images
        assert "at least two regions" in figure.axes[5].texts[0].get_text()
    finally:
        plt.close(figure)

    exports = export_explanations(
        tmp_path,
        results,
        {"pet": game},
        ["Breed"],
        mask_fill="black",
        region_mode="trimap",
        region_names={"pet": names},
    )
    assert len(exports) == 4
    assert {record["index"] for record in exports} == {"SV"}
    with np.load(tmp_path / "pet_full_ensemble.npz", allow_pickle=False) as archive:
        assert archive["region_names"].tolist() == names
        assert archive["coalitions"].shape == (2, 1)
        assert json.loads(str(archive["metadata_json"]))["region_mode"] == "trimap"
    parameters = json.loads((tmp_path / str(exports[0]["file"])).read_text())["parameters"]
    assert parameters["region_mode"] == "trimap"
    assert parameters["region_names"] == names


def test_irregular_regions_show_semantic_names_and_exact_zero_pair():
    game, result = _result_for(np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]))
    assert (0, 1) not in result["exact"]["EU"]["k-SII"].dict_values
    cases = {
        (condition, method): result
        for condition in ("full", "scarce")
        for method in ("ensemble", "mc_dropout")
    }
    limits = explanation_plot_limits({("pet", *run): result for run in cases})
    names = ["Pet", "Unclassified"]
    regions = plot_region_explanation(
        "pet", game, result, ["Breed"], limits, region_mode="trimap", region_names=names
    )
    comparison = plot_epistemic_comparison(
        "pet", game, cases, limits, region_mode="trimap", region_names=names
    )
    try:
        regions.canvas.draw()
        comparison.canvas.draw()
        heatmap = regions.axes[5].images[0].get_array()
        assert np.ma.is_masked(heatmap[0, 0])
        assert heatmap[0, 1] == 0
        assert [label.get_text() for label in regions.axes[5].get_xticklabels()] == names
        assert comparison.axes[0].texts[0].get_text() == "Pet +0.75\nUnclassified +0.25"
    finally:
        plt.close(regions)
        plt.close(comparison)
