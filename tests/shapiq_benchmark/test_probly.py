"""Caller-defined probly games work directly with benchmark ground-truth computers."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark import Benchmark
from shapiq_benchmark.computers import BruteForceComputer
from shapiq_games.probly import ImageProblyGame, ProblyGame, ProblyPredictor
from shapiq_games.vision import grid_regions


def _coalition_game():
    def quantify(rows):
        value = rows @ np.array([1.0, 2.0, 3.0]) + 2 * rows[:, 0] * rows[:, 1]
        return {"score": value, "twice": 2 * value}

    predictor = ProblyPredictor(lambda rows: rows, output="score", quantifier=quantify)
    return ProblyGame(3, predictor)


@pytest.mark.parametrize("output, factor", [("score", 1), ("twice", 2)])
def test_benchmark_computes_exact_values_of_caller_defined_coalition_games(
    output, factor, monkeypatch, tmp_path
):
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    game = _coalition_game().for_output(output)
    benchmark = Benchmark(game)
    assert isinstance(benchmark.computer, BruteForceComputer)
    assert benchmark.setup is None
    assert benchmark.key is None

    values = benchmark.exact_values("SV", 1)
    np.testing.assert_allclose([values[(i,)] for i in range(3)], factor * np.array([2, 3, 3]))
    assert values.baseline_value == 0
    interactions = benchmark.exact_values("SII", 2)
    np.testing.assert_allclose(interactions[(0, 1)], factor * 2)
    assert not list(tmp_path.iterdir())  # no named setup or disk cache is needed


def test_benchmark_computes_exact_values_of_caller_defined_image_games():
    predictor = ProblyPredictor(
        lambda images: images,
        output="brightness",
        quantifier=lambda images: {"brightness": images.mean(axis=(1, 2, 3)) / 255},
    )
    game = ImageProblyGame(
        np.full((4, 4, 3), 255, dtype=np.uint8),
        predictor,
        regions=grid_regions(4, 4, 1, 2),
        fill="black",
    )
    benchmark = Benchmark(game)
    assert isinstance(benchmark.computer, BruteForceComputer)
    values = benchmark.exact_values("SV", 1)
    np.testing.assert_allclose([values[(0,)], values[(1,)]], [0.5, 0.5])
    assert values.baseline_value == 0
