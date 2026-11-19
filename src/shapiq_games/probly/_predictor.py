"""Convert representations into scalar predictions without discarding the representation API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from lazy_dispatch import lazydispatch
from shapiq_games._optional import require

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from shapiq.typing import FloatVector


@lazydispatch
def _as_numpy(value: object) -> np.ndarray:
    """Convert an output to a detached floating NumPy array."""
    return np.asarray(value, dtype=float)


@_as_numpy.register("torch.Tensor")
def _torch_as_numpy(value: object) -> np.ndarray:
    """Convert a Torch output without importing Torch for NumPy users."""
    return cast("Any", value).detach().cpu().double().numpy()


def _output_name(output: str) -> str:
    """Validate a scalar output's name."""
    if not isinstance(output, str) or not output.strip():
        msg = "output must be a nonempty string."
        raise ValueError(msg)
    return output


def _builtin_outputs(
    representation: object, *, class_index: int | None, base: float | None
) -> dict[str, object]:
    """Use probly's categorical entropy decomposition and mean-distribution decider."""
    require("probly", purpose="categorical uncertainty quantification", extra="probly")
    from probly.decider import categorical_from_mean
    from probly.quantification import decompose
    from probly.representation.distribution import (
        CategoricalDistribution,
        CategoricalDistributionSample,
    )

    is_sample = isinstance(representation, CategoricalDistributionSample)
    if not is_sample and not isinstance(representation, CategoricalDistribution):
        msg = (
            "The built-in quantifier requires a categorical distribution or an unweighted "
            "sample of categorical distributions. Supply a custom quantifier for other "
            "representations."
        )
        raise TypeError(msg)
    if is_sample:
        sample = cast("Any", representation)
        if sample.is_weighted:
            msg = (
                "Weighted categorical entropy is unsupported by the selected probly revision. "
                "Supply an explicit custom quantifier with the required weight semantics."
            )
            raise ValueError(msg)
        if sample.sample_size == 0:
            msg = "A categorical distribution sample must contain at least one member."
            raise ValueError(msg)

    distribution = categorical_from_mean(cast("Any", representation))
    probabilities = _as_numpy(distribution.probabilities)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] == 0:
        msg = (
            "The mean categorical distribution must have shape (1, n_classes), got "
            f"{probabilities.shape}. Keep the input row and class axes explicit."
        )
        raise ValueError(msg)
    if not np.isfinite(probabilities).all():
        msg = "Categorical probabilities must be finite."
        raise ValueError(msg)
    decomposition = decompose(cast("Any", representation), base=base)
    outputs: dict[str, object] = {"total": decomposition["total"]}
    if is_sample:
        outputs["aleatoric"] = decomposition["aleatoric"]
        outputs["epistemic"] = decomposition["epistemic"]
    if class_index is not None:
        if class_index >= probabilities.shape[1]:
            msg = f"class_index={class_index} is out of range for {probabilities.shape[1]} classes."
            raise ValueError(msg)
        outputs["probability"] = probabilities[:, class_index]
    return outputs


class ProblyPredictor:
    """Adapt a representation predictor to scalar shapiq model outputs.

    The built-in quantifier explains categorical entropy in nats (``base=None``): total entropy
    of the mean predictive distribution, mean member entropy (aleatoric), and their difference
    (epistemic). First-order categorical distributions provide only total entropy. A fixed
    ``class_index`` additionally exposes its mean ``"probability"``. Other representations and
    measures can be used with an explicit custom quantifier.

    Scalar evaluation always processes singleton rows with the same seed. This prevents caller
    batching and order from selecting different numerical kernels or random draws. A stochastic
    predictor must supply an evaluation context that restores its RNG and model state. The
    supplied predictor and preprocessing must otherwise be deterministic and unchanged while
    games or caches use them. Model sampling, missing-feature imputation, and Shapley approximation
    are separate sources of randomness.

    :meth:`represent` retains the original representation and its member/sample identities for
    future analysis. Explaining entropy is different from computing a distribution of Shapley
    values across model realizations; this adapter implements the former.

    Attributes:
        representer: The supplied representation-producing callable.
        class_index: The fixed class whose probability is exposed, if any.
        base: The entropy logarithm base; ``None`` means nats.
        random_state: The seed passed to each evaluation context.

    Examples:
        A custom quantifier can use any representation without importing probly:

        >>> predictor = ProblyPredictor(
        ...     lambda rows: rows,
        ...     output="score",
        ...     quantifier=lambda rows: {"score": rows.sum(axis=1)},
        ... )
        >>> predictor(np.array([[1., 2.], [3., 4.]]))
        array([3., 7.])
    """

    def __init__(
        self,
        representer: Callable[..., object],
        *,
        output: str = "total",
        preprocess: Callable[[np.ndarray], object] | None = None,
        quantifier: Callable[..., Mapping[str, object]] | None = None,
        class_index: int | None = None,
        base: float | None = None,
        evaluation_context: Callable[[int], AbstractContextManager[object]] | None = None,
        random_state: int = 0,
    ) -> None:
        """Configure representation construction and scalar quantification.

        Args:
            representer: A callable, such as a probly ``Representer``, returning a representation
                for a batch. Model transformations, fitting and sampler construction belong to
                the caller.
            output: The selected output name. Defaults to ``"total"``.
            preprocess: Optional deterministic conversion from a NumPy input batch to the
                representer's input, for example image feature extraction or tensor conversion.
            quantifier: Optional callable mapping a representation to a nonempty mapping of
                named output vectors. Each vector must have exactly one finite scalar per input
                row. Its names must remain the same across evaluations. It owns its own measure,
                logarithm base, sample weighting and representation semantics.
            class_index: A nonnegative fixed class index for built-in mean probability output.
                No class is automatically selected.
            base: The built-in entropy base, positive and different from one; ``None`` means nats.
            evaluation_context: Optional factory accepting a seed and returning a fresh context
                manager around preprocessing, representation construction and quantification.
                It must restore RNG and model state even if evaluation fails.
            random_state: The nonnegative seed reused for every singleton evaluation.

        Raises:
            TypeError: If the representation producer or optional callbacks are not callable.
            ValueError: If the output name, class index, logarithm base or seed is invalid.
        """
        if not callable(representer):
            msg = "representer must be callable."
            raise TypeError(msg)
        for name, callback in (
            ("preprocess", preprocess),
            ("quantifier", quantifier),
            ("evaluation_context", evaluation_context),
        ):
            if callback is not None and not callable(callback):
                msg = f"{name} must be callable or None."
                raise TypeError(msg)
        if class_index is not None and (
            isinstance(class_index, bool) or not isinstance(class_index, int) or class_index < 0
        ):
            msg = "class_index must be a nonnegative integer or None."
            raise ValueError(msg)
        if base is not None and (not np.isfinite(base) or base <= 0 or base == 1):
            msg = "base must be finite, positive and different from one, or None."
            raise ValueError(msg)
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            msg = "random_state must be a nonnegative integer."
            raise ValueError(msg)
        self.representer = representer
        self.class_index = class_index
        self.base = base
        self.random_state = random_state
        self._output = _output_name(output)
        self._preprocess = preprocess
        self._quantifier = quantifier
        self._evaluation_context = evaluation_context
        self._output_names: tuple[str, ...] | None = None

    @property
    def output(self) -> str:
        """The scalar output selected by :meth:`__call__`."""
        return self._output

    def _context(self) -> AbstractContextManager[object]:
        if self._evaluation_context is None:
            return nullcontext()
        return self._evaluation_context(self.random_state)

    def _represent(self, batch: np.ndarray) -> object:
        inputs = batch if self._preprocess is None else self._preprocess(batch)
        return self.representer(inputs)

    def represent(self, batch: np.ndarray) -> object:
        """Build the original representation for an input batch without scalar reduction.

        This lower-level operation forwards the complete batch. Unlike scalar evaluation, its
        batching and sample identity are controlled by the caller. The configured context wraps
        preprocessing and representation construction.

        Args:
            batch: The input array with rows on axis zero.

        Returns:
            The original representation, retaining sample axes and weights.
        """
        batch = _input_batch(batch)
        with self._context():
            return self._represent(batch)

    def _quantify(self, representation: object) -> dict[str, FloatVector]:
        outputs = (
            _builtin_outputs(representation, class_index=self.class_index, base=self.base)
            if self._quantifier is None
            else self._quantifier(representation)
        )
        if not isinstance(outputs, Mapping) or not outputs:
            msg = "quantifier must return a nonempty mapping of output names to vectors."
            raise ValueError(msg)
        for name in outputs:
            _output_name(name)
        names = tuple(sorted(outputs))
        if self._output_names is not None and names != self._output_names:
            msg = "The quantifier's output names must remain the same across evaluations."
            raise ValueError(msg)
        result = {}
        for name in names:
            values = _as_numpy(outputs[name])
            if values.shape != (1,):
                msg = f"Output {name!r} must have shape (1,), got {values.shape}."
                raise ValueError(msg)
            if not np.isfinite(values).all():
                msg = f"Output {name!r} must contain only finite values."
                raise ValueError(msg)
            result[name] = np.array(values, dtype=float, copy=True)
        self._output_names = names
        return result

    def evaluate_outputs(self, batch: np.ndarray) -> dict[str, FloatVector]:
        """Evaluate all configured outputs, independently for every input row.

        Args:
            batch: The input array with rows on axis zero.

        Returns:
            A mapping of output names to finite floating vectors of shape ``(n_rows,)``.

        Raises:
            ValueError: If the quantifier's output names, shapes or values are invalid, or an
                empty batch is supplied before any output names have been established.
        """
        batch = _input_batch(batch)
        if batch.shape[0] == 0:
            if self._output_names is None:
                msg = "Evaluate a nonempty batch before requesting empty output vectors."
                raise ValueError(msg)
            return {name: np.empty(0, dtype=float) for name in self._output_names}
        collected: dict[str, list[float]] = {}
        for index in range(batch.shape[0]):
            with self._context():
                outputs = self._quantify(self._represent(batch[index : index + 1]))
            for name, values in outputs.items():
                collected.setdefault(name, []).append(float(values[0]))
        return {name: np.array(values, dtype=float) for name, values in collected.items()}

    def __call__(self, batch: np.ndarray) -> FloatVector:
        """Return the selected scalar output for every input row."""
        return _select_output(self.evaluate_outputs(batch), self.output)


def _input_batch(batch: np.ndarray) -> np.ndarray:
    """Check the explicit input row axis."""
    batch = np.asarray(batch)
    if batch.ndim == 0:
        msg = "An input batch must have a row axis."
        raise ValueError(msg)
    return batch


def _select_output[T](outputs: Mapping[str, T], output: str) -> T:
    """Select an existing output with an informative error."""
    if output not in outputs:
        msg = f"Output {output!r} is unavailable. Available outputs: {', '.join(sorted(outputs))}."
        raise ValueError(msg)
    return outputs[output]
