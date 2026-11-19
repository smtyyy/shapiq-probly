"""Oxford Pets figures and exports for the evaluated uncertainty games."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import fill
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np

from shapiq import InteractionValues
from shapiq.plot import interaction_heatmap_plot, region_attribution_plot

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from shapiq_games.probly import ImageProblyGame, ProblyGame
    from shapiq_games.typing import Fill


class ExplanationResult(TypedDict):
    """The evaluated games and exact explanations for one image and method."""

    target: int
    true_class_index: int
    coalitions: np.ndarray
    values: dict[str, np.ndarray]
    exact: dict[str, dict[str, InteractionValues]]
    games: dict[str, ProblyGame]


@dataclass(frozen=True)
class ExplanationPlotLimits:
    """Symmetric color limits for the four groups of scores in the Oxford figures."""

    probability: float | None
    total_and_aleatoric: float | None
    epistemic: float | None
    epistemic_pairs: float | None


def _max_abs_score(values: Iterable[np.ndarray]) -> float | None:
    """Leave an all-zero scale to the plotting function."""
    return max((float(np.max(np.abs(array))) for array in values), default=0.0) or None


def explanation_plot_limits(
    results: Mapping[tuple[str, str, str], ExplanationResult],
) -> ExplanationPlotLimits:
    """Use one scale per group across all images and compare EU across all conditions."""
    overview_runs = [
        result
        for (_, condition, method), result in results.items()
        if (condition, method) == ("full", "ensemble")
    ]
    return ExplanationPlotLimits(
        probability=_max_abs_score(
            result["exact"]["P"]["SV"].get_n_order_values(1) for result in overview_runs
        ),
        total_and_aleatoric=_max_abs_score(
            result["exact"][short]["SV"].get_n_order_values(1)
            for result in overview_runs
            for short in ("TU", "AU")
        ),
        epistemic=_max_abs_score(
            result["exact"]["EU"]["SV"].get_n_order_values(1) for result in results.values()
        ),
        epistemic_pairs=_max_abs_score(
            result["exact"]["EU"]["k-SII"].get_n_order_values(2) for result in overview_runs
        ),
    )


def plot_test_overview(
    images: Sequence[tuple[np.ndarray, str, str]],
    *,
    captions: Mapping[str, str] | None = None,
) -> Iterator[Figure]:
    """Show cropped test images with IDs and optional case captions, twelve per page."""
    row_height = 4.25 if captions else 3.1
    caption_width = 28 if captions else 24
    for start in range(0, len(images), 12):
        page = images[start : start + 12]
        columns = min(len(page), 4)
        rows = (len(page) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.3 * columns, row_height * rows + (0.85 if len(images) > 12 else 0.6)),
            squeeze=False,
            layout="constrained",
        )
        title = "Test crops · 224 by 224 px"
        if len(images) > 12:
            if len(page) == 1:
                title += f"\nImage {start + 1} of {len(images)}"
            else:
                title += f"\nImages {start + 1} to {start + len(page)} of {len(images)}"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, (image, class_name, image_id) in zip(axes.flat, page, strict=False):
            caption = captions.get(image_id, class_name) if captions else class_name
            wrapped = "\n".join(fill(line, width=caption_width) for line in caption.splitlines())
            ax.imshow(image)
            ax.set_title(f"{wrapped}\n{image_id}", fontsize=11)
        yield fig


def _region_centers(regions: np.ndarray, n_players: int) -> Iterator[tuple[int, float, float]]:
    """Find the centers of the rectangular regions in the Oxford raster."""
    for player in range(n_players):
        ys, xs = np.where(regions == player)
        yield player, float(xs.mean()), float(ys.mean())


def _show_region_reference(ax: Axes, image_game: ImageProblyGame) -> None:
    """Outline and label each image region."""
    ax.imshow(image_game.image)
    labels = image_game.regions
    for row_boundary in np.flatnonzero(np.any(np.diff(labels, axis=0), axis=1)):
        ax.axhline(float(row_boundary) + 0.5, color="white", linewidth=0.8)
    for column_boundary in np.flatnonzero(np.any(np.diff(labels, axis=1), axis=0)):
        ax.axvline(float(column_boundary) + 0.5, color="white", linewidth=0.8)
    for player, x, y in _region_centers(labels, image_game.n_players):
        ax.text(
            x,
            y,
            str(player),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none"},
        )
    ax.axis("off")


def _annotate_epistemic(ax: Axes, explanation: InteractionValues, regions: np.ndarray) -> None:
    """Keep small EU effects legible when a common scale makes them very pale."""
    for player, x, y in _region_centers(regions, explanation.n_players):
        value = explanation[(player,)]
        ax.text(
            x,
            y,
            f"{value:+.2g}" if value else "0",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.87, "edgecolor": "none", "pad": 1.6},
        )


def _reference_to_visible(result: ExplanationResult, short: str) -> str:
    """Format the score of the masked reference and the fully visible image."""
    baseline = result["exact"][short]["SV"].baseline_value
    return f"{baseline:.4g} → {result['values'][short][-1]:.4g}"


def _horizontal_colorbar(
    fig: Figure, axes: Sequence[Axes], label: str, limit: float | None
) -> None:
    """Give narrow guides fewer ticks while always showing zero."""
    ticks = np.linspace(-limit, limit, 3 if len(axes) == 1 else 5) if limit else [0.0]
    fig.colorbar(
        axes[0].images[-1],
        ax=list(axes),
        location="bottom",
        shrink=0.82,
        label=label,
        ticks=ticks,
        format="%.2g",
    )


def plot_region_explanation(
    image_id: str,
    image_game: ImageProblyGame,
    result: ExplanationResult,
    class_names: Sequence[str],
    limits: ExplanationPlotLimits,
    *,
    case_label: str | None = None,
) -> Figure:
    """Draw region labels, signed Shapley contributions, and pairwise EU interactions."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.1), layout="constrained")
    image_title = f"{case_label} · {image_id}" if case_label else image_id
    fig.suptitle(
        f"{image_title} · Full training / ensemble\nScores shown from masked reference to fully visible",
        fontsize=13,
        fontweight="bold",
    )
    _show_region_reference(axes[0, 0], image_game)
    target_name = fill(class_names[result["target"]], width=26)
    axes[0, 0].set_title(f"Region IDs · fixed P target\n{target_name}", fontsize=11)
    panels = (
        ("P", axes[0, 1], limits.probability),
        ("EU", axes[0, 2], limits.epistemic),
        ("TU", axes[1, 0], limits.total_and_aleatoric),
        ("AU", axes[1, 1], limits.total_and_aleatoric),
    )
    for short, ax, limit in panels:
        sv = result["exact"][short]["SV"]
        region_attribution_plot(sv, image_game.regions, ax=ax, max_score=limit, colorbar=False)
        unit = "p" if short == "P" else "nats"
        ax.set_title(f"{short} · SV\n{_reference_to_visible(result, short)} {unit}", fontsize=11)
        if short == "EU":
            _annotate_epistemic(ax, sv, image_game.regions)

    pairs = axes[1, 2]
    interaction_heatmap_plot(
        result["exact"]["EU"]["k-SII"],
        ax=pairs,
        max_score=limits.epistemic_pairs,
        colorbar=False,
    )
    pairs.set_title("EU · pairwise k-SII\nGray cells lack a pair value", fontsize=11)
    pairs.set(xlabel="Region", ylabel="Region")
    guides = (
        ([axes[0, 1]], "P Shapley contribution [Δp]", limits.probability),
        ([axes[0, 2]], "EU Shapley contribution [nats]", limits.epistemic),
        (
            [axes[1, 0], axes[1, 1]],
            "TU / AU Shapley contribution [nats]",
            limits.total_and_aleatoric,
        ),
        ([pairs], "EU k-SII interaction [nats]", limits.epistemic_pairs),
    )
    for guide_axes, label, limit in guides:
        _horizontal_colorbar(fig, guide_axes, label, limit)
    return fig


def plot_epistemic_comparison(
    image_id: str,
    image_game: ImageProblyGame,
    runs: Mapping[tuple[str, str], ExplanationResult],
    limits: ExplanationPlotLimits,
    *,
    case_label: str | None = None,
) -> Figure:
    """Compare all four EU runs with the same scale used for the cohort."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.3), layout="constrained")
    image_title = f"{case_label} · {image_id}" if case_label else image_id
    fig.suptitle(
        f"{image_title} · Epistemic uncertainty\nScores shown from masked reference to fully visible",
        fontsize=13,
        fontweight="bold",
    )
    panels = (
        ("full", "ensemble", "Full training · Ensemble"),
        ("full", "mc_dropout", "Full training · MC Dropout"),
        ("scarce", "ensemble", "Reduced training · Ensemble"),
        ("scarce", "mc_dropout", "Reduced training · MC Dropout"),
    )
    for ax, (condition, method, title) in zip(axes.flat, panels, strict=True):
        result = runs[condition, method]
        sv = result["exact"]["EU"]["SV"]
        region_attribution_plot(
            sv, image_game.regions, ax=ax, max_score=limits.epistemic, colorbar=False
        )
        _annotate_epistemic(ax, sv, image_game.regions)
        ax.set_title(
            f"{title}\n{_reference_to_visible(result, 'EU')} nats",
            fontsize=11,
        )
    _horizontal_colorbar(fig, list(axes.flat), "EU Shapley contribution [nats]", limits.epistemic)
    fig.supxlabel("Local explanation relative to the masked reference", fontsize=10)
    return fig


def export_explanations(
    output_dir: Path,
    results: Mapping[tuple[str, str, str], ExplanationResult],
    image_games: Mapping[str, ImageProblyGame],
    class_names: Sequence[str],
    *,
    mask_fill: Fill,
) -> list[dict[str, object]]:
    """Save the coalition archives and interaction JSON files and verify their round trips."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_explanations: list[dict[str, object]] = []
    for (image_id, condition, method), result in results.items():
        stem = f"{image_id}_{condition}_{method}"
        metadata = {
            "image_id": image_id,
            "condition": condition,
            "method": method,
            "target_index": result["target"],
            "target_class": class_names[result["target"]],
            "true_class_index": result["true_class_index"],
            "true_class": class_names[result["true_class_index"]],
            "n_players": image_games[image_id].n_players,
            "normalize": False,
            "fill": mask_fill,
            "entropy_base": "e",
            "units": {"P": "probability", "TU": "nats", "AU": "nats", "EU": "nats"},
        }
        arrays: dict[str, np.ndarray] = {
            "coalitions": result["coalitions"],
            "regions": image_games[image_id].regions,
            "class_names": np.asarray(class_names),
            "target_index": np.asarray(result["target"], dtype=np.int64),
            "true_class_index": np.asarray(result["true_class_index"], dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(metadata, allow_nan=False)),
        }
        arrays.update({f"values_{name}": values for name, values in result["values"].items()})
        for name, explanations in result["exact"].items():
            for index, explanation in explanations.items():
                arrays[f"{name}_{index}"] = explanation.values
                arrays[f"{name}_{index}_baseline"] = np.asarray(explanation.baseline_value)
                json_path = output_dir / f"{stem}_{name}_{index}.json"
                explanation.to_json_file(
                    json_path,
                    desc=f"{image_id}: {name} {index}; {condition}, {method}",
                    image_id=image_id,
                    condition=condition,
                    method=method,
                    output=name,
                    unit="probability" if name == "P" else "nats",
                    target_index=result["target"],
                )
                restored = InteractionValues.from_json_file(json_path)
                np.testing.assert_equal(restored.dict_values, explanation.dict_values)
                np.testing.assert_equal(restored.baseline_value, explanation.baseline_value)
                np.testing.assert_equal(
                    (restored.index, restored.min_order, restored.max_order, restored.n_players),
                    (
                        explanation.index,
                        explanation.min_order,
                        explanation.max_order,
                        explanation.n_players,
                    ),
                )
                exported_explanations.append(
                    {"file": json_path.name, **metadata, "output": name, "index": index}
                )
        archive_path = output_dir / f"{stem}.npz"
        np.savez_compressed(archive_path, allow_pickle=False, **arrays)
        with np.load(archive_path, allow_pickle=False) as restored_arrays:
            np.testing.assert_array_equal(restored_arrays["coalitions"], result["coalitions"])
            np.testing.assert_array_equal(restored_arrays["regions"], image_games[image_id].regions)
            np.testing.assert_equal(json.loads(str(restored_arrays["metadata_json"])), metadata)
            for name, values in result["values"].items():
                np.testing.assert_array_equal(restored_arrays[f"values_{name}"], values)
    return exported_explanations


def export_summary(
    output_dir: Path,
    report: Mapping[str, object],
    *,
    expected_entries: int,
) -> None:
    """Save the summary and verify the cohort and the number of rows after reading it."""
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(dict(report), indent=2, allow_nan=False), encoding="utf-8")
    restored_report = json.loads(summary_path.read_text(encoding="utf-8"))
    np.testing.assert_equal(restored_report["cohort_ids"], report["cohort_ids"])
    np.testing.assert_equal(len(restored_report["summary"]), expected_entries)
