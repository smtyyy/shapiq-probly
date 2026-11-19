"""Scalar cooperative games built from probly prediction representations.

The adapter keeps representation construction separate from scalar quantification. Games explain
uncertainty, such as predictive entropy; they do not estimate uncertainty of Shapley values.
"""

from ._context import torch_cpu_context
from ._games import ImageProblyGame, ProblyGame
from ._predictor import ProblyPredictor

__all__ = ["ImageProblyGame", "ProblyGame", "ProblyPredictor", "torch_cpu_context"]
