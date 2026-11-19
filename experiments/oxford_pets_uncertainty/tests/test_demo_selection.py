"""Exercise the demo rules with small, explicit ensemble probabilities."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("probly")

from experiments.oxford_pets_uncertainty.helpers import Config  # noqa: E402
from experiments.oxford_pets_uncertainty.selection import select_cases  # noqa: E402
from probly.representation.distribution import create_categorical_distribution  # noqa: E402
from probly.representation.sample import create_sample  # noqa: E402


class FixedEnsemble:
    """Use the feature row to look up fixed probabilities for each member."""

    def __init__(self, *members):
        self.members = [np.asarray(member) for member in members]
        self.evaluated = []

    def represent(self, batch):
        indices = batch[:, 0].to(torch.long).numpy()
        self.evaluated.extend(indices.tolist())
        return create_sample(
            [create_categorical_distribution(member[indices]) for member in self.members],
            sample_axis=0,
        )

    __call__ = represent


def _inputs(ids, labels):
    data = {
        "identities": {"test": ids},
        "labels": {"test": torch.tensor(labels)},
        "classes": ["A", "B", "C"],
    }
    features = {"test": torch.arange(len(ids), dtype=torch.float32).reshape(-1, 1)}
    return data, features


def test_demo_uses_mean_class_probabilities_and_breaks_ties_by_image_id():
    ids = ["z-high", "a-high", "z-low", "a-low", "z-error", "a-error"]
    data, features = _inputs(ids, [0, 0, 1, 1, 2, 2])
    first = [(1.0, 0, 0)] * 2 + [(0.25, 0.5, 0.25)] * 2 + [(0, 0.875, 0.125)] * 2
    second = [(0.75, 0.125, 0.125)] * 2 + [(0.375, 0.5, 0.125)] * 2 + [(0, 1.0, 0)] * 2
    ensemble = FixedEnsemble(first, second)

    cases, missing = select_cases(data, features, ensemble, Config(n_images=0), mode="demo")

    assert missing == []
    assert [(case["role"], case["image_id"]) for case in cases] == [
        ("Sicher richtig", "a-high"),
        ("Unsicher richtig", "a-low"),
        ("Sicher falsch", "a-error"),
    ]
    assert [case["index"] for case in cases] == [1, 3, 5]
    assert [case["true_class"] for case in cases] == [0, 1, 2]
    assert [case["target_class"] for case in cases] == [0, 1, 1]
    assert [case["confidence"] for case in cases] == pytest.approx([0.875, 0.5, 0.9375])
    assert ensemble.evaluated == list(range(len(ids)))


@pytest.mark.parametrize(
    ("labels", "expected_ids", "expected_missing"),
    [
        ([0, 0], ["a"], ["Unsicher richtig", "Sicher falsch"]),
        ([0, 1], ["z", "a"], ["Unsicher richtig"]),
        ([1, 1], ["a"], ["Sicher richtig", "Unsicher richtig"]),
    ],
)
def test_demo_reports_missing_roles_and_omits_low_case_without_confidence_contrast(
    labels, expected_ids, expected_missing
):
    data, features = _inputs(["z", "a"], labels)
    ensemble = FixedEnsemble([(0.75, 0.25, 0)] * 2)

    cases, missing = select_cases(data, features, ensemble, Config(), mode="demo")

    assert [case["image_id"] for case in cases] == expected_ids
    assert missing == expected_missing


def test_seeded_mode_evaluates_only_the_selected_images_in_seed_order():
    data, features = _inputs(["d", "c", "b", "a"], [1, 0, 0, 2])
    ensemble = FixedEnsemble(
        [(0.625, 0.375, 0), (0.25, 0.75, 0), (0.875, 0.125, 0), (0.125, 0, 0.875)]
    )

    cases, missing = select_cases(data, features, ensemble, Config(n_images=2), mode="seeded")

    assert missing == []
    assert [case["role"] for case in cases] == [None, None]
    assert [case["image_id"] for case in cases] == ["a", "c"]
    assert [case["confidence"] for case in cases] == pytest.approx([0.875, 0.75])
    assert ensemble.evaluated == [3, 1]


def test_seeded_mode_reports_invalid_image_count_before_inference():
    data, features = _inputs(["a", "b", "c"], [0, 1, 2])
    ensemble = FixedEnsemble([(1.0, 0, 0)] * 3)

    with pytest.raises(ValueError, match="n_images must be between 1 and 3"):
        select_cases(data, features, ensemble, Config(n_images=0), mode="seeded")

    assert ensemble.evaluated == []
