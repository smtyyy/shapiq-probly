"""
Regions and pairwise interactions
==================================

Show first-order contributions at their spatial locations and read all pairwise
interaction scores for a single explanation. The same pairwise plot works with
any named features.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shapiq import ExactComputer, interaction_heatmap_plot, region_attribution_plot


def game(coalitions: np.ndarray) -> np.ndarray:
    """Assign signed main effects and pairwise contributions to three players."""
    first, second, third = coalitions.astype(float).T
    return first - 2 * second + 0.5 * third + first * third - second * third


computer = ExactComputer(game, n_players=3)
regions = np.array([[0, 0, 1], [0, 2, 1], [2, 2, 1]])

# %%
# Spatial contributions
# ---------------------
# The integers in the region map identify players, including for irregular regions.

region_attribution_plot(computer(index="SV", order=1), regions)

# %%
# Pairwise interactions
# ---------------------
# The matrix shows each pair in both directions. Gray marks the diagonal and
# any pair that is absent from the explanation.

interaction_heatmap_plot(
    computer(index="k-SII", order=2), feature_names=["Region 0", "Region 1", "Region 2"]
)
plt.show()
