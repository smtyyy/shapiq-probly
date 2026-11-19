"""Offline numerical and game-contract tests of the public probly integration."""

from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from math import factorial

import numpy as np
import pytest

from shapiq_games import ImageProblyGame, ProblyGame, ProblyPredictor

pytest.importorskip("probly")

from probly.representation.distribution import (
    create_categorical_distribution,
    create_gaussian_distribution,
)
from probly.representation.sample import create_sample


def _entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = probabilities[positive] * np.log(probabilities[positive])
    return -terms.sum(axis=-1)


def _head_probabilities(x):
    x = np.asarray(x, dtype=float)
    first = 0.1 + 0.15 * x[:, 0] + 0.05 * x[:, 1]
    second = 0.85 - 0.2 * x[:, 0] - 0.1 * x[:, 2]
    return np.stack(
        [np.stack([first, 1 - first], axis=-1), np.stack([second, 1 - second], axis=-1)]
    )


def _sample_model(x, *, sample_axis=0):
    return create_sample(
        [create_categorical_distribution(p) for p in _head_probabilities(x)],
        sample_axis=sample_axis,
    )


def _coalitions(n_players=3):
    return np.array(
        [
            [bool(mask & (1 << player)) for player in range(n_players)]
            for mask in range(2**n_players)
        ]
    )


@pytest.mark.parametrize("sample_axis", [0, -1])
def test_entropy_outputs_match_independent_oracle_with_explicit_sample_axes(sample_axis):
    predictor = ProblyPredictor(lambda x: _sample_model(x, sample_axis=sample_axis), base=2)
    coalitions = _coalitions()
    probabilities = _head_probabilities(coalitions)
    total = _entropy(probabilities.mean(axis=0)) / np.log(2)
    aleatoric = _entropy(probabilities).mean(axis=0) / np.log(2)
    outputs = predictor.evaluate_outputs(coalitions)

    np.testing.assert_allclose(outputs["total"], total, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outputs["aleatoric"], aleatoric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(outputs["epistemic"], total - aleatoric, atol=1e-12)
    np.testing.assert_allclose(predictor(coalitions), total, atol=1e-12)


@pytest.mark.parametrize(
    ("heads", "total", "aleatoric", "epistemic"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], 1.0, 0.0, 1.0),
        ([[0.5, 0.5], [0.5, 0.5]], 1.0, 1.0, 0.0),
        ([[1.0, 0.0], [1.0, 0.0]], 0.0, 0.0, 0.0),
    ],
)
def test_agreeing_and_opposing_heads_have_known_uncertainties(heads, total, aleatoric, epistemic):
    def represent(x):
        return create_sample(
            [create_categorical_distribution(np.tile(p, (len(x), 1))) for p in heads],
            sample_axis=0,
        )

    outputs = ProblyPredictor(represent, base=2).evaluate_outputs(np.ones((2, 1)))
    for name, expected in [("total", total), ("aleatoric", aleatoric), ("epistemic", epistemic)]:
        np.testing.assert_allclose(outputs[name], expected, atol=1e-12)


def test_default_entropy_is_in_nats_and_raw_representations_remain_available():
    predictor = ProblyPredictor(_sample_model)
    x = _coalitions()
    np.testing.assert_allclose(
        predictor(x), _entropy(_head_probabilities(x).mean(axis=0)), atol=1e-12
    )
    representation = predictor.represent(x)
    assert representation.sample_size == 2
    np.testing.assert_allclose(
        representation.sample_mean().probabilities, _head_probabilities(x).mean(axis=0)
    )


def test_first_order_distribution_reports_total_without_fabricating_a_decomposition():
    def represent(x):
        return create_categorical_distribution(np.tile([0.25, 0.75], (len(x), 1)))

    outputs = ProblyPredictor(represent).evaluate_outputs(np.ones((3, 1)))
    assert "aleatoric" not in outputs
    assert "epistemic" not in outputs
    np.testing.assert_allclose(outputs["total"], _entropy([0.25, 0.75]))
    with pytest.raises((ValueError, KeyError)):
        ProblyPredictor(represent, output="epistemic")(np.ones((1, 1)))


def test_probability_uses_one_explicit_fixed_class_across_all_coalitions():
    x = _coalitions()
    predictor = ProblyPredictor(_sample_model, output="probability", class_index=0)
    expected = _head_probabilities(x).mean(axis=0)[:, 0]
    np.testing.assert_allclose(predictor(x), expected)
    assert np.any(expected < 0.5)
    with pytest.raises((ValueError, KeyError)):
        ProblyPredictor(_sample_model, output="probability")(x)
    with pytest.raises((ValueError, IndexError)):
        ProblyPredictor(_sample_model, output="probability", class_index=2)(x)


def test_weighted_entropy_samples_are_rejected_instead_of_using_inconsistent_means():
    def represent(x):
        sample = _sample_model(x)
        return type(sample)(
            array=sample.array, sample_axis=sample.sample_axis, weights=np.array([0.2, 0.8])
        )

    with pytest.raises((ValueError, NotImplementedError)):
        ProblyPredictor(represent)(np.ones((1, 3)))


def test_preprocessing_is_applied_before_representation():
    observed = []

    def represent(x):
        observed.append(x.copy())
        return _sample_model(x)

    x = np.zeros((2, 3))
    predictor = ProblyPredictor(represent, preprocess=lambda data: data + 1)
    np.testing.assert_allclose(predictor(x), ProblyPredictor(_sample_model)(x + 1))
    assert all(np.array_equal(row, np.ones((1, 3))) for row in observed)


def test_custom_regression_variance_quantifier_reuses_probly():
    from probly.quantification import SecondOrderVarianceDecomposition

    def represent(x):
        return create_sample(
            [
                create_gaussian_distribution(x[:, 0], np.full(len(x), 0.25)),
                create_gaussian_distribution(x[:, 0] + 2, np.full(len(x), 0.75)),
            ],
            sample_axis=0,
        )

    def quantify(representation):
        variance = SecondOrderVarianceDecomposition(representation)
        return {name: getattr(variance, name) for name in ("total", "aleatoric", "epistemic")}

    outputs = ProblyPredictor(represent, quantifier=quantify).evaluate_outputs(np.array([[0], [3]]))
    np.testing.assert_allclose(outputs["aleatoric"], 0.5, atol=1e-12)
    np.testing.assert_allclose(outputs["epistemic"], 1.0, atol=1e-12)
    np.testing.assert_allclose(outputs["total"], 1.5, atol=1e-12)


def test_custom_quantification_can_expose_nonadditive_outputs():
    predictor = ProblyPredictor(
        lambda x: x,
        output="width",
        quantifier=lambda x: {"width": x[:, 0] + 2, "another_measure": x[:, 0] + 3},
    )
    x = np.array([[1], [4]])
    np.testing.assert_array_equal(predictor(x), [3, 6])
    np.testing.assert_array_equal(predictor.evaluate_outputs(x)["another_measure"], [4, 7])


@pytest.mark.parametrize(
    "quantifier",
    [
        lambda x: {},
        lambda x: {"total": np.array([np.nan])},
        lambda x: {"total": np.array([np.inf])},
        lambda x: {"total": np.array([[1.0]])},
        lambda x: {"total": np.array([1.0, 2.0])},
        lambda x: {"total": 1.0},
    ],
)
def test_custom_quantifiers_must_return_finite_per_row_vectors(quantifier):
    with pytest.raises((ValueError, TypeError)):
        ProblyPredictor(lambda x: x, quantifier=quantifier)(np.ones((2, 1)))


def test_custom_quantifier_missing_selected_output_fails():
    predictor = ProblyPredictor(
        lambda x: x, quantifier=lambda x: {"width": np.ones(len(x))}, output="total"
    )
    with pytest.raises((ValueError, KeyError)):
        predictor(np.ones((1, 1)))


def test_custom_quantifier_output_names_must_remain_consistent_between_rows():
    predictor = ProblyPredictor(
        lambda x: x,
        quantifier=lambda x: {"total" if x[0, 0] == 0 else "different_output": np.ones(len(x))},
    )
    with pytest.raises(ValueError):
        predictor.evaluate_outputs(np.array([[0], [1]]))


def test_custom_seed_context_fixes_draws_per_row_and_restores_external_numpy_rng():
    @contextmanager
    def context(seed):
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)

    predictor = ProblyPredictor(
        lambda x: x[:, 0] + np.random.uniform(size=len(x)),
        quantifier=lambda values: {"score": values},
        output="score",
        evaluation_context=context,
        random_state=4,
    )
    x = np.array([[1.0], [3.0], [5.0]])
    state = np.random.get_state()
    values = predictor(x)
    np.testing.assert_allclose(values - x[:, 0], np.repeat(values[0] - x[0, 0], 3))
    np.testing.assert_array_equal(predictor(x[::-1])[::-1], values)
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert state[2:] == np.random.get_state()[2:]


class _CountingRepresentation:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return _sample_model(x)


def test_output_views_share_evaluations_and_clear_cache_invalidates_every_view():
    represent = _CountingRepresentation()
    game = ProblyGame(3, ProblyPredictor(represent), cache_size=16)
    epistemic = game.for_output("epistemic")
    coalition = np.array([[1, 0, 1]], dtype=bool)
    total_value = game(coalition)
    after_total = represent.calls
    epistemic_value = epistemic(coalition)
    assert represent.calls == after_total
    assert game.output == "total"
    assert epistemic.output == "epistemic"
    np.testing.assert_allclose(
        total_value,
        ProblyGame(3, ProblyPredictor(_sample_model))(coalition),
    )
    assert epistemic_value.shape == (1,)
    epistemic.clear_cache()
    game(coalition)
    assert represent.calls > after_total


def test_lru_eviction_recomputes_identical_values():
    represent = _CountingRepresentation()
    game = ProblyGame(3, ProblyPredictor(represent), cache_size=1)
    first = np.array([[1, 0, 0]], dtype=bool)
    second = np.array([[0, 1, 0]], dtype=bool)
    expected = game(first)
    game(second)
    before_repeat = represent.calls
    np.testing.assert_array_equal(game(first), expected)
    assert represent.calls > before_repeat


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("image_game", [True, False])
def test_output_views_keep_separate_precomputations_and_zero_baseline_centering(
    normalize, image_game
):
    def quantify(rows):
        score = rows.reshape(len(rows), -1).mean(axis=1)
        return {"zero_baseline": score, "nonzero_baseline": score + 5}

    predictor = ProblyPredictor(lambda rows: rows, output="zero_baseline", quantifier=quantify)
    if image_game:
        game = ImageProblyGame(
            np.ones((2, 2, 3), dtype=np.uint8),
            predictor,
            regions=np.array([[0, 0], [1, 1]]),
            fill="black",
            normalize=normalize,
        )
    else:
        game = ProblyGame(2, predictor, normalize=normalize)
    game.precompute()
    assert game.precomputed
    view = game.for_output("nonzero_baseline")
    assert not view.precomputed
    assert view(view.empty_coalition)[0] == pytest.approx(0 if normalize else 5)
    assert view(view.grand_coalition)[0] == pytest.approx(1 if normalize else 6)
    assert game(game.grand_coalition)[0] == pytest.approx(1)
    assert game.n_values_stored == 4


def _shapley_oracle(values, n_players):
    scores = np.zeros(n_players)
    for player in range(n_players):
        for mask in range(2**n_players):
            if mask & (1 << player):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(n_players - size - 1) / factorial(n_players)
            scores[player] += weight * (values[mask | (1 << player)] - values[mask])
    return scores


def _pair_sii_oracle(values, n_players, pair):
    pair_mask = sum(1 << player for player in pair)
    score = 0.0
    for mask in range(2**n_players):
        if mask & pair_mask:
            continue
        size = mask.bit_count()
        weight = factorial(size) * factorial(n_players - size - 2) / factorial(n_players - 1)
        difference = (
            values[mask | pair_mask]
            - values[mask | (1 << pair[0])]
            - values[mask | (1 << pair[1])]
            + values[mask]
        )
        score += weight * difference
    return score


def test_exact_shapley_and_interactions_match_exhaustive_oracles_and_decompose():
    game = ProblyGame(3, ProblyPredictor(_sample_model, base=2))
    coalitions = _coalitions()
    explanations = {}
    for output in ("total", "aleatoric", "epistemic"):
        view = game.for_output(output)
        values = view(coalitions)
        shapley = view.exact_values(index="SV", order=1)
        scores = np.array([shapley[(player,)] for player in range(3)])
        np.testing.assert_allclose(scores, _shapley_oracle(values, 3), atol=1e-12)
        assert scores.sum() == pytest.approx(values[-1] - values[0], abs=1e-12)
        interactions = view.exact_values(index="SII", order=2)
        for pair in combinations(range(3), 2):
            assert interactions[pair] == pytest.approx(_pair_sii_oracle(values, 3, pair), abs=1e-12)
        explanations[output] = (shapley, interactions)

    for position in range(2):
        for interaction in explanations["total"][position].interaction_lookup:
            assert explanations["total"][position][interaction] == pytest.approx(
                explanations["aleatoric"][position][interaction]
                + explanations["epistemic"][position][interaction],
                abs=1e-12,
            )


@pytest.mark.parametrize("fill", ["black", "mean"])
def test_image_game_masks_exact_regions_before_prediction_and_centers_baseline(fill):
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10
    regions = np.array([[0, 0, 1], [0, 2, 2]])
    seen = []

    def represent(images):
        seen.extend(image.copy() for image in images)
        brightness = images.mean(axis=(1, 2, 3)) / 255
        return create_categorical_distribution(np.stack([brightness, 1 - brightness], axis=-1))

    game = ImageProblyGame(
        image,
        ProblyPredictor(represent, output="probability", class_index=0),
        regions=regions,
        fill=fill,
    )
    coalition = np.array([True, False, True])
    expected = image.copy()
    fill_color = np.zeros(3) if fill == "black" else image.reshape(-1, 3).mean(0).round()
    expected[regions == 1] = fill_color.astype(np.uint8)
    np.testing.assert_array_equal(game.masked_image(coalition), expected)
    assert game(game.empty_coalition)[0] == pytest.approx(0.0)
    expected_baseline = fill_color.mean() / 255
    assert game.empty_value == pytest.approx(expected_baseline)
    assert game(coalition.reshape(1, -1))[0] == pytest.approx(
        expected.mean() / 255 - expected_baseline
    )
    assert any(np.array_equal(masked, expected) for masked in seen)


def test_image_output_views_keep_image_masking_and_share_cache():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    regions = np.array([[0, 0], [1, 2]])
    calls = []

    def represent(images):
        calls.append(images.copy())
        return _sample_model(images.mean(axis=(1, 2)) / 255)

    game = ImageProblyGame(image, ProblyPredictor(represent), regions=regions, fill="black")
    other = game.for_output("epistemic")
    coalition = np.array([[1, 0, 1]], dtype=bool)
    game(coalition)
    before_other = len(calls)
    assert isinstance(other, ImageProblyGame)
    assert other(coalition).shape == (1,)
    assert len(calls) == before_other
    np.testing.assert_array_equal(game.masked_image(coalition[0]), other.masked_image(coalition[0]))


def test_image_game_owns_snapshots_of_image_regions_and_fill():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    regions = np.array([[0, 0], [1, 2]])
    fill = np.full_like(image, 128)
    predictor = ProblyPredictor(
        lambda rows: rows,
        output="brightness",
        quantifier=lambda rows: {"brightness": rows.mean(axis=(1, 2, 3))},
    )
    game = ImageProblyGame(image, predictor, regions=regions, fill=fill)
    coalition = np.array([[1, 0, 1]], dtype=bool)
    expected_image = game.masked_image(coalition[0]).copy()
    expected_value = game(coalition)
    image[:] = 255
    regions[:] = 0
    fill[:] = 0
    game.clear_cache()
    np.testing.assert_array_equal(game.masked_image(coalition[0]), expected_image)
    np.testing.assert_array_equal(game(coalition), expected_value)


@pytest.mark.parametrize("regions", [np.zeros((1, 2), dtype=int), np.array([[0, 2], [0, 2]])])
def test_invalid_image_regions_fail_before_model_evaluation(regions):
    represent = _CountingRepresentation()
    with pytest.raises(ValueError):
        ImageProblyGame(
            np.zeros((2, 2, 3), dtype=np.uint8), ProblyPredictor(represent), regions=regions
        )
    assert represent.calls == 0
