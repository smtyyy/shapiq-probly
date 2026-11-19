"""Scalar game views sharing representation inference and a bounded coalition cache."""

from __future__ import annotations

from collections import OrderedDict
from copy import copy
from typing import TYPE_CHECKING, Self

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games.vision._fill import fill_image, filled_images
from shapiq_games.vision._preprocess import as_rgb_array
from shapiq_games.vision._region_game import RegionGame
from shapiq_games.vision._regions import check_regions

from ._predictor import ProblyPredictor, _output_name, _select_output

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Fill


def _cache_capacity(cache_size: int) -> int:
    """Validate the maximum number of scalar coalition records."""
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
        msg = "cache_size must be a nonnegative integer."
        raise ValueError(msg)
    return cache_size


class _CoalitionEvaluator:
    """Share scalar outputs, never representations or model copies, across game views."""

    def __init__(
        self,
        predictor: ProblyPredictor,
        inputs: Callable[[CoalitionMatrix], np.ndarray],
        *,
        n_players: int,
        cache_size: int,
    ) -> None:
        self.predictor = predictor
        self.inputs = inputs
        self.n_players = n_players
        self.cache_size = _cache_capacity(cache_size)
        self.cache: OrderedDict[bytes, dict[str, float]] = OrderedDict()

    def values(self, coalitions: CoalitionMatrix, output: str) -> GameValues:
        coalitions = as_bool_coalitions(coalitions)
        if coalitions.shape[1] != self.n_players:
            msg = f"Expected {self.n_players} coalition columns, got {coalitions.shape[1]}."
            raise ValueError(msg)
        values = np.empty(coalitions.shape[0], dtype=float)
        for index, coalition in enumerate(coalitions):
            key = np.packbits(coalition).tobytes()
            outputs = self.cache.get(key)
            if outputs is None:
                predictions = self.predictor.evaluate_outputs(self.inputs(coalition[None]))
                outputs = {name: float(vector[0]) for name, vector in predictions.items()}
                if self.cache_size:
                    self.cache[key] = outputs
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            elif self.cache_size:
                self.cache.move_to_end(key)
            values[index] = _select_output(outputs, output)
        return values


class ProblyGame(Game):
    """A scalar cooperative game of a probly representation predictor.

    The predictor receives boolean coalitions. Its preprocessing can map them to masked inputs,
    cached features, or any other caller-defined representation inputs. Output views share the
    predictor and a bounded cache containing scalar results for all configured outputs. They do
    not copy models or retain representations. Predictor configuration and model state must remain
    unchanged for the lifetime of the games.

    Attributes:
        predictor: The scalar prediction adapter.
        empty_value: The selected empty-coalition output before centering.
        original_model_output: The selected grand-coalition output before centering.

    Examples:
        >>> predictor = ProblyPredictor(
        ...     lambda coalitions: coalitions,
        ...     output="sum",
        ...     quantifier=lambda rows: {"sum": rows.sum(1), "twice": 2 * rows.sum(1)},
        ... )
        >>> game = ProblyGame(3, predictor)
        >>> game(np.array([[1, 0, 1]], dtype=bool))
        array([2.])
        >>> game.for_output("twice")(np.array([[1, 0, 1]], dtype=bool))
        array([4.])
    """

    def __init__(
        self,
        n_players: int,
        predictor: ProblyPredictor,
        *,
        output: str | None = None,
        normalize: bool = True,
        cache_size: int = 4096,
        verbose: bool = False,
    ) -> None:
        """Build a game whose representation input is its coalition matrix.

        Args:
            n_players: The positive number of players.
            predictor: A configured :class:`ProblyPredictor`. Its ``preprocess`` may look up
                cached features or construct model inputs from coalitions.
            output: The selected output, or ``None`` to use ``predictor.output``.
            normalize: Whether to subtract the empty-coalition output.
            cache_size: Maximum number of cached coalitions across all output views. Zero
                disables memoization. Only scalar outputs are retained.
            verbose: Whether to show progress while evaluating the game.

        Raises:
            TypeError: If ``predictor`` is not a :class:`ProblyPredictor`.
            ValueError: If the player count, cache capacity or selected output is invalid.
        """
        if isinstance(n_players, bool) or not isinstance(n_players, int) or n_players <= 0:
            msg = "n_players must be a positive integer."
            raise ValueError(msg)
        if not isinstance(predictor, ProblyPredictor):
            msg = "predictor must be a ProblyPredictor."
            raise TypeError(msg)
        self.predictor = predictor
        self._output = _output_name(predictor.output if output is None else output)
        self._center = normalize
        self._evaluator = _CoalitionEvaluator(
            predictor, np.asarray, n_players=n_players, cache_size=cache_size
        )
        self._initialize(n_players, normalize=normalize, verbose=verbose)

    def _initialize(self, n_players: int, *, normalize: bool, verbose: bool) -> None:
        self.empty_value = float(
            self._evaluator.values(np.zeros((1, n_players), dtype=bool), self.output)[0]
        )
        self.original_model_output = float(
            self._evaluator.values(np.ones((1, n_players), dtype=bool), self.output)[0]
        )
        Game.__init__(
            self,
            n_players,
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    @property
    def output(self) -> str:
        """The selected scalar output of this game view."""
        return self._output

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the selected raw output for every coalition."""
        return self._evaluator.values(coalitions, self.output)

    def for_output(self, name: str) -> Self:
        """Create a scalar game view sharing the predictor and coalition cache.

        Args:
            name: An output name exposed by the predictor's quantifier.

        Returns:
            A game with its own normalization and precomputed scalar table. Its centering choice
            matches this game, including when the current baseline happens to be zero.

        Raises:
            ValueError: If the output is not available.
        """
        view = copy(self)
        view._output = _output_name(name)  # noqa: SLF001 - another instance of this class
        view._initialize(  # noqa: SLF001 - another instance of this class
            self.n_players, normalize=self._center, verbose=self.verbose
        )
        return view

    def clear_cache(self) -> None:
        """Clear scalar memoization shared by all views; keep precomputed game tables intact."""
        self._evaluator.cache.clear()


class ImageProblyGame(RegionGame):
    """A probly scalar output explained over regions of an RGB image.

    Masking, normalization and image attribution helpers follow the existing shapiq region games.
    The caller supplies the image regions and a prepared predictor. Dataset/model downloads,
    training, resizing, feature extraction and tensor conversion remain caller-controlled.

    Attributes:
        predictor: The scalar prediction adapter.
        image: The explained RGB image.
        regions: The player assigned to every pixel.

    Examples:
        >>> from shapiq_games.vision import grid_regions
        >>> image = np.full((4, 4, 3), 255, dtype=np.uint8)
        >>> predictor = ProblyPredictor(
        ...     lambda images: images,
        ...     output="brightness",
        ...     quantifier=lambda images: {"brightness": images.mean(axis=(1, 2, 3)) / 255},
        ... )
        >>> game = ImageProblyGame(
        ...     image, predictor, regions=grid_regions(4, 4, 2, 2), fill="black"
        ... )
        >>> game(np.array([[1, 0, 0, 1]], dtype=bool))
        array([0.5])
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        predictor: ProblyPredictor,
        *,
        regions: np.ndarray,
        output: str | None = None,
        fill: Fill | np.ndarray = "mean",
        normalize: bool = True,
        cache_size: int = 4096,
        verbose: bool = False,
    ) -> None:
        """Build a game that replaces absent image regions with a fixed fill.

        Args:
            image: An RGB image array or image path. Float images in ``[0, 1]`` are scaled.
            predictor: A configured :class:`ProblyPredictor` receiving RGB image batches. Its
                preprocessing may perform feature extraction or conversion to model tensors.
            regions: Integer player labels of shape ``(height, width)``, using every label
                ``0, ..., n_players - 1``. Use ``vision.grid_regions`` for a regular grid.
            output: The selected output, or ``None`` to use ``predictor.output``.
            fill: The fixed replacement: ``"mean"``, ``"gray"``, ``"black"``, ``"blur"``, or
                another image of the same shape.
            normalize: Whether to subtract the empty-image output.
            cache_size: Maximum number of scalar coalition records shared by output views;
                zero disables memoization.
            verbose: Whether to show progress while evaluating the game.

        Raises:
            TypeError: If ``predictor`` is not a :class:`ProblyPredictor`.
            ValueError: If image regions, fill, cache capacity or output is invalid.
        """
        if not isinstance(predictor, ProblyPredictor):
            msg = "predictor must be a ProblyPredictor."
            raise TypeError(msg)
        self.image = as_rgb_array(image).copy()
        if self.image.shape[0] == 0 or self.image.shape[1] == 0:
            msg = "The image must have nonempty height and width."
            raise ValueError(msg)
        self.regions = check_regions(regions, self.image).copy()
        self._fill = fill_image(self.image, fill).copy()
        self.predictor = predictor
        self._output = _output_name(predictor.output if output is None else output)
        self._center = normalize
        # Own immutable snapshots, so caller-side array mutation cannot change cached values.
        self.image.flags.writeable = False
        self.regions.flags.writeable = False
        self._fill.flags.writeable = False
        image_snapshot, regions_snapshot, fill_snapshot = self.image, self.regions, self._fill

        def inputs(coalitions: CoalitionMatrix) -> np.ndarray:
            return filled_images(image_snapshot, regions_snapshot, fill_snapshot, coalitions)

        self._evaluator = _CoalitionEvaluator(
            predictor,
            inputs,
            n_players=int(self.regions.max()) + 1,
            cache_size=cache_size,
        )
        super().__init__(normalize=normalize, verbose=verbose)

    @property
    def output(self) -> str:
        """The selected scalar output of this game view."""
        return self._output

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        return self._evaluator.values(coalitions, self.output)

    def for_output(self, name: str) -> Self:
        """Create another scalar image game sharing masking, predictor and coalition cache.

        Args:
            name: An output name exposed by the predictor's quantifier.

        Returns:
            An image game with its own baseline and precomputed scalar table, using the same
            centering choice as this view.

        Raises:
            ValueError: If the output is not available.
        """
        view = copy(self)
        view._output = _output_name(name)  # noqa: SLF001 - another instance of this class
        RegionGame.__init__(view, normalize=self._center, verbose=self.verbose)
        return view

    def clear_cache(self) -> None:
        """Clear scalar memoization shared by all views; keep precomputed game tables intact."""
        self._evaluator.cache.clear()
