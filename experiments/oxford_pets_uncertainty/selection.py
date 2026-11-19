"""Select illustrative Oxford test images from unmasked ensemble predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch
from probly.decider import categorical_from_mean

if TYPE_CHECKING:
    from experiments.oxford_pets_uncertainty.helpers import (
        Config,
        FrozenFeatures,
        PreparedData,
        TensorRepresenter,
    )

CohortMode = Literal["demo", "seeded"]


class SelectedCase(TypedDict):
    """Image identity, true label and fixed reference-ensemble prediction."""

    role: str | None
    index: int
    image_id: str
    target_class: int
    true_class: int
    confidence: float


def select_cases(
    data: PreparedData,
    features: FrozenFeatures,
    reference: TensorRepresenter,
    config: Config,
    *,
    mode: CohortMode,
) -> tuple[list[SelectedCase], list[str]]:
    """Return selected cases in display order and the names of unavailable demo roles.

    The demo ranks the largest ensemble-mean class probability, breaking ties by image ID.
    Seeded mode chooses at most one image per breed using only labels and the configured seed.
    """
    test_ids = data["identities"]["test"]
    test_labels = data["labels"]["test"].numpy()
    test_features = features["test"]
    if mode == "demo":
        indices = range(len(test_ids))
    elif mode == "seeded":
        n_classes = len(data["classes"])
        if not 1 <= config.n_images <= n_classes:
            msg = f"n_images must be between 1 and {n_classes} in seeded mode."
            raise ValueError(msg)
        rng = np.random.default_rng(config.seed)
        indices = [
            int(rng.choice(np.flatnonzero(test_labels == breed)))
            for breed in rng.permutation(n_classes)[: config.n_images]
        ]
        test_features = test_features[indices]
    else:
        msg = f"Unknown Oxford Pets cohort mode {mode!r}. Choose 'demo' or 'seeded'."
        raise ValueError(msg)

    cases: list[SelectedCase] = []
    with torch.no_grad():
        for batch in test_features.split(256):
            distribution = categorical_from_mean(reference.represent(batch.double()))
            probabilities = np.asarray(distribution.probabilities)
            if (
                probabilities.shape != (len(batch), len(data["classes"]))
                or not np.isfinite(probabilities).all()
            ):
                msg = "Expected finite ensemble probabilities for every test image and class."
                raise ValueError(msg)
            predictions = probabilities.argmax(axis=1)
            batch_indices = indices[len(cases) : len(cases) + len(batch)]
            for row, (index, target) in enumerate(zip(batch_indices, predictions, strict=True)):
                cases.append(
                    {
                        "role": None,
                        "index": index,
                        "image_id": test_ids[index],
                        "target_class": int(target),
                        "true_class": int(test_labels[index]),
                        "confidence": float(probabilities[row, target]),
                    }
                )
    if mode == "seeded":
        return cases, []

    correct = [case for case in cases if case["target_class"] == case["true_class"]]
    incorrect = [case for case in cases if case["target_class"] != case["true_class"]]
    high_correct = min(
        correct, key=lambda case: (-case["confidence"], case["image_id"]), default=None
    )
    low_correct = min(
        correct, key=lambda case: (case["confidence"], case["image_id"]), default=None
    )
    high_incorrect = min(
        incorrect, key=lambda case: (-case["confidence"], case["image_id"]), default=None
    )
    if (
        low_correct is not None
        and high_correct is not None
        and low_correct["confidence"] == high_correct["confidence"]
    ):
        low_correct = None

    selected: list[SelectedCase] = []
    missing: list[str] = []
    for role, case in (
        ("Sicher richtig", high_correct),
        ("Unsicher richtig", low_correct),
        ("Sicher falsch", high_incorrect),
    ):
        if case is None:
            missing.append(role)
        else:
            case["role"] = role
            selected.append(case)
    return selected, missing
