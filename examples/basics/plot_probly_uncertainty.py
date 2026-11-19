"""Explain predictive uncertainty with probly
==========================================

A model's prediction and its uncertainty answer different questions. This small,
offline example explains the total, aleatoric, and epistemic entropy of two
classifiers with the same game. It needs the optional ``shapiq[probly]`` extra.

The classifiers below stand in for an already trained probly ensemble. Real
models can supply a probly ``Representer`` instead of the small callable.
Probly owns the representation and entropy calculations; shapiq owns the game
and the Shapley calculations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from probly.representation.distribution import create_categorical_distribution
from probly.representation.sample import create_sample

import shapiq
from shapiq_games.probly import ProblyGame, ProblyPredictor


def represent(coalitions: np.ndarray) -> object:
    """Return a sample of two categorical predictions for each coalition."""
    first = 0.5 + 0.35 * coalitions[:, 0] - 0.2 * coalitions[:, 1]
    second = 0.5 - 0.25 * coalitions[:, 0] + 0.15 * coalitions[:, 1]
    members = [
        create_categorical_distribution(np.column_stack([probability, 1 - probability]))
        for probability in (first, second)
    ]
    return create_sample(members, sample_axis=0)


# %%
# One representation supplies several scalar games. Entropy uses the entire
# class distribution and is measured in nats.
predictor = ProblyPredictor(represent)
total = ProblyGame(2, predictor)
aleatoric = total.for_output("aleatoric")
epistemic = total.for_output("epistemic")

coalitions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
np.testing.assert_allclose(total(coalitions), aleatoric(coalitions) + epistemic(coalitions))

# %%
# Compute exact Shapley values with the standard shapiq game interface.
explanations = {
    name: shapiq.ExactComputer(game)("SV", 1)
    for name, game in {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}.items()
}
np.testing.assert_allclose(
    explanations["total"].values,
    explanations["aleatoric"].values + explanations["epistemic"].values,
    atol=1e-12,
)

fig, ax = plt.subplots(figsize=(6, 3))
positions = np.arange(2)
for offset, (name, explanation) in enumerate(explanations.items()):
    ax.bar(positions + (offset - 1) * 0.25, explanation.get_n_order_values(1), 0.25, label=name)
ax.set(xticks=positions, xticklabels=["Feature 1", "Feature 2"], ylabel="Shapley value (nats)")
ax.axhline(0, color="black", linewidth=0.5)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# A custom quantifier can return other named scalar quantities (for example an
# explicit probly variance decomposition or conformal set size). This extension
# point does not require changing the Game, the approximator, or its result type.
# Explaining uncertainty is distinct from estimating uncertainty of an explanation.
# The preserved ``predictor.represent`` interface leaves member/sample information
# accessible for future explanation-per-member analyses.
