"""Explain predictive uncertainty with probly
==========================================

Which parts of a handwritten digit support the prediction of a 6, and which make
two classifiers disagree? This offline example trains two small classifiers,
explains their predicted probability and uncertainty over four image regions,
and shows where the explanations differ. It needs the optional ``shapiq[probly]``
extra.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from probly.representation.distribution import create_categorical_distribution
from probly.representation.sample import create_sample
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import shapiq
from shapiq_games.probly import ImageProblyGame, ProblyPredictor
from shapiq_games.vision import grid_regions

# %%
# .. _probly-image-example:
#
# Explaining a Handwritten Digit
# ------------------------------
#
# We fit two logistic regression classifiers on separate halves of the training
# data. For this illustration, we picked one held-out image on which their most
# likely digits differ.
#
# Data: Alpaydin and Kaynak (1998), `Optical Recognition of Handwritten Digits
# <https://doi.org/10.24432/C50P49>`_, UCI Machine Learning Repository,
# `CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_.
# We use the 8-by-8 grayscale images bundled with scikit-learn and copy the grayscale
# channel to RGB. We display the digit as dark strokes on a light background.
# For the model, absent regions are set to zero, the value of a blank pixel.

digits = load_digits()
train, _ = train_test_split(
    np.arange(len(digits.target)), test_size=0.2, random_state=0, stratify=digits.target
)
training_halves = train_test_split(
    train, test_size=0.5, random_state=0, stratify=digits.target[train]
)
models = [
    LogisticRegression(max_iter=500).fit(digits.data[indices] / 16.0, digits.target[indices])
    for indices in training_halves
]

example_index = 420
target = int(digits.target[example_index])
image = np.repeat(digits.images[example_index, ..., None], 3, axis=-1).astype(np.uint8)
regions = grid_regions(*image.shape[:2], rows=2, cols=2)


def digit_features(images: np.ndarray) -> np.ndarray:
    """Flatten the first channel of each masked image and scale it to [0, 1]."""
    return images[..., 0].reshape(len(images), -1) / 16.0


print(f"Recorded digit: {target}")
for number, model in enumerate(models, start=1):
    probabilities = model.predict_proba(digit_features(image[None]))[0]
    top_class = probabilities.argmax()
    print(f"Model {number} favors {model.classes_[top_class]} ({probabilities[top_class]:.1%})")

# %%
# Choose What to Explain
# ----------------------
#
# Each model returns probabilities for the same ten digits, ordered from 0 to 9.
# ``represent`` keeps both distributions as a probly sample. The
# :class:`~shapiq_games.ProblyPredictor` turns that sample into scalar outputs.
# The probability output takes the mean probability of the fixed target digit,
# 6, for every mask. Total uncertainty is the entropy of the mean prediction.
# Aleatoric uncertainty is the mean of the two entropies, and the difference is
# epistemic uncertainty, which measures disagreement between these two models.


def represent(features: np.ndarray) -> object:
    """Return one categorical distribution per classifier for every input row."""
    members = [create_categorical_distribution(model.predict_proba(features)) for model in models]
    return create_sample(members, sample_axis=0)


predictor = ProblyPredictor(represent, preprocess=digit_features, class_index=target)
game = ImageProblyGame(image, predictor, regions=regions, fill="black", normalize=False)
views = {name: game.for_output(name) for name in ("probability", "total", "aleatoric", "epistemic")}

# The first coalition shows no quadrants, and the last shows all four.
coalitions = shapiq.get_explicit_subsets(game.n_players, list(range(game.n_players + 1)))
scores = {name: view(coalitions) for name, view in views.items()}
np.testing.assert_allclose(scores["total"], scores["aleatoric"] + scores["epistemic"])

print(f"{'Output':<24} {'Blank image':>12} {'Full image':>12}")
for name, values in scores.items():
    label = f"P({target})" if name == "probability" else f"{name.title()} (nats)"
    print(f"{label:<24} {values[0]:>12.3f} {values[-1]:>12.3f}")

# %%
# The full image reduces total uncertainty, even as the classifiers disagree
# more. The decrease in their mean entropy is larger than the increase in
# disagreement.
#
# Explain Regions and Their Interactions
# --------------------------------------
#
# :class:`~shapiq.game_theory.ExactComputer` can explain any of the four views.
# Here we compare Shapley values for probability and epistemic uncertainty
# and compute pairwise k-SII interactions for epistemic uncertainty.

probability_computer = shapiq.ExactComputer(views["probability"])
epistemic_computer = shapiq.ExactComputer(views["epistemic"])
probability_sv = probability_computer(index="SV", order=1)
epistemic_sv = epistemic_computer(index="SV", order=1)
epistemic_pairs = epistemic_computer(index="k-SII", order=2)

fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), layout="constrained")
original, probability_ax, epistemic_ax, pairs_ax = axes.flat

original.imshow(image[..., 0], cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
original.set(title=f"Handwritten {target}", xticks=[], yticks=[])
original.axhline(3.5, color="0.5", linewidth=0.7, linestyle="--")
original.axvline(3.5, color="0.5", linewidth=0.7, linestyle="--")
for region, (x, y) in enumerate(((0, 0), (7, 0), (0, 7), (7, 7))):
    original.text(
        x,
        y,
        str(region),
        ha="center",
        va="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )

shapiq.region_attribution_plot(
    probability_sv, regions, ax=probability_ax, colorbar_label="Shapley contribution (Δp)"
)
probability_ax.set_title(f"Probability of {target}")
shapiq.region_attribution_plot(
    epistemic_sv, regions, ax=epistemic_ax, colorbar_label="Shapley contribution (nats)"
)
epistemic_ax.set_title("Epistemic uncertainty")
shapiq.interaction_heatmap_plot(
    epistemic_pairs, ax=pairs_ax, colorbar_label="Pairwise k-SII (nats)"
)
pairs_ax.set(title="Epistemic interactions", xlabel="Region")
plt.show()

# %%
# The upper right quadrant (region 1) lowers the probability of a 6 but raises
# disagreement on average. The lower left (region 2) raises both. Their positive
# interaction indicates that revealing them together raises disagreement more
# than their separate effects would suggest, averaged over the other regions.
# These effects are relative to replacing missing regions with blank pixels.
# The gray diagonal has no pairwise values.
#
# To explain the mean member entropy, create an ``ExactComputer`` for
# ``views["aleatoric"]``. See :ref:`games-probly` for tabular games, MC dropout,
# and custom quantifiers.
