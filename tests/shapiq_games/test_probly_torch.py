"""Finite MC Dropout is a deterministic game without leaking model or RNG state."""

from __future__ import annotations

import random

import numpy as np
import pytest

from shapiq_games import ProblyGame, ProblyPredictor, torch_cpu_context

torch = pytest.importorskip("torch")
pytest.importorskip("probly")

from probly.representer import Sampler  # noqa: E402
from probly.transformation import cast  # noqa: E402


def _model_and_predictor(seed=17):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(1).get_state())
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(8, 2),
        )
    model.train()
    model[1].eval()
    sampler = Sampler(cast(model, predictor_type="logit_classifier"), num_samples=12, sample_axis=0)
    predictor = ProblyPredictor(
        sampler,
        output="epistemic",
        preprocess=lambda x: torch.as_tensor(x, dtype=torch.float32),
        evaluation_context=torch_cpu_context(model),
        random_state=seed,
    )
    return model, predictor


def _coalitions():
    return np.array([[bool(mask & (1 << player)) for player in range(3)] for mask in range(8)])


def test_mc_dropout_uses_fixed_draws_across_batches_order_and_fresh_games():
    _, predictor = _model_and_predictor()
    _, fresh = _model_and_predictor()
    x = _coalitions()
    expected = predictor(x)
    chunks = np.concatenate([predictor(x[:2]), predictor(x[2:5]), predictor(x[5:])])
    permutation = np.array([7, 0, 3, 1, 6, 2, 5, 4])
    np.testing.assert_array_equal(chunks, expected)
    np.testing.assert_array_equal(fresh(x[permutation])[np.argsort(permutation)], expected)
    np.testing.assert_array_equal(predictor(x), expected)

    game = ProblyGame(3, predictor, cache_size=1)
    values = game(x)
    game.clear_cache()
    np.testing.assert_array_equal(game(x[::-1])[::-1], values)
    np.testing.assert_array_equal(ProblyGame(3, fresh, cache_size=1)(x), values)
    assert np.any(expected > 0)


def test_mc_dropout_seed_changes_finite_sample_without_changing_model():
    model, predictor = _model_and_predictor(seed=17)
    other_model, other = _model_and_predictor(seed=18)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, other_model.state_dict()[name], rtol=0, atol=0)
    assert np.max(np.abs(predictor(_coalitions()) - other(_coalitions()))) > 1e-7


def test_torch_context_preserves_mixed_training_flags_buffers_and_external_rngs():
    model, predictor = _model_and_predictor()
    flags = [module.training for module in model.modules()]
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    predictor(_coalitions())

    assert [module.training for module in model.modules()] == flags
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    assert random.getstate() == python_state


def test_torch_context_restores_rng_and_module_flags_when_quantification_raises():
    model, predictor = _model_and_predictor()
    flags = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()

    def fail(representation):
        message = "intentional quantification failure"
        raise RuntimeError(message)

    broken = ProblyPredictor(
        predictor.represent,
        quantifier=fail,
        evaluation_context=torch_cpu_context(model),
        random_state=19,
    )
    with pytest.raises(RuntimeError, match="intentional quantification failure"):
        broken(np.ones((1, 3)))

    assert [module.training for module in model.modules()] == flags
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_torch_context_uses_eval_mode_and_temporarily_disables_gradients():
    model, _ = _model_and_predictor()
    flags = [module.training for module in model.modules()]
    with torch.enable_grad(), torch_cpu_context(model)(23):
        assert not any(module.training for module in model.modules())
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    assert [module.training for module in model.modules()] == flags
