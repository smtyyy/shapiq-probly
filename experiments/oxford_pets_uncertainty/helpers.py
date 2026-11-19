"""Typed data, hardware preparation, frozen features, and head training for Oxford Pets.

Uncertainty quantification and cooperative games belong to probly and shapiq. This module only
prepares the experiment's data and fitted models; it does not run an experiment on import.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import warnings
from dataclasses import dataclass
from importlib.metadata import distribution, version
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypedDict,
    cast as type_cast,
)

import numpy as np
import torch
from experiments._shared.torch_cache import load_or_compute
from numpy.typing import NDArray
from PIL import Image
from probly.representer import Sampler, representer
from probly.transformation import cast, ensemble, subensemble
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import (
    InterpolationMode,
    functional as vision_functions,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from probly.representation import Representation

    from shapiq.typing import CoalitionMatrix
    from shapiq_games.probly import ImageProblyGame

Split = Literal["trainval", "test"]
Condition = Literal["full", "scarce"]
Method = Literal["ensemble", "mc_dropout"]
RGBImage = NDArray[np.uint8]
SPLITS: tuple[Split, Split] = ("trainval", "test")


@dataclass(frozen=True)
class Config:
    """Research settings; scientific and runtime cache identities are kept separate."""

    seed: int = 0
    n_images: int = 3
    head_seeds: tuple[int, ...] = (0, 1, 2)
    scarce_per_class: int = 20
    subset_seed: int = 10
    mc_samples: int = 64
    feature_batch_size: int = 4
    feature_device: str = "auto"
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5


@dataclass(frozen=True)
class RuntimeInfo:
    """The requested backend, verified backend, and actual precision of each stage."""

    requested_device: str
    device: str
    available_backends: dict[str, bool]
    platform: str
    thread_count: int
    feature_batch_size: int
    encoder_dtype: Literal["float32"]
    training_device: Literal["cpu"]
    training_dtype: Literal["float32"]
    head_inference_device: Literal["cpu"]
    head_inference_dtype: Literal["float64"]
    fallback_reason: str | None


class PreparedData(TypedDict):
    """Public Oxford datasets and fixed, disjoint experimental selections."""

    datasets: dict[Split, OxfordIIITPet]
    identities: dict[Split, list[str]]
    labels: dict[Split, torch.Tensor]
    classes: list[str]
    train: torch.Tensor
    validation: torch.Tensor


class FrozenFeatures(TypedDict):
    """One shared frozen feature tensor per split and its input provenance."""

    trainval: torch.Tensor
    test: torch.Tensor
    provenance: dict[Split, dict[str, object]]


class HeadCheckpoint(TypedDict):
    """A selected head state and the loss trace that selected it."""

    state: dict[str, torch.Tensor]
    best_epoch: int
    validation_loss: list[float]


class TrainingSettings(TypedDict):
    """Settings that change initialization, optimization, or checkpoint selection."""

    architecture: str
    initialization_seed: int
    head_seeds: list[int]
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    dtype: Literal["float32"]


class TrainingIdentity(TypedDict):
    """Scientific identity independent of plotting, MC sampling, and encoder placement."""

    schema: int
    config: TrainingSettings
    train_ids: list[str]
    validation_ids: list[str]
    features: str
    versions: dict[str, str]


class ConditionRecord(TypedDict):
    """The selected heads together with their scientific cache identity."""

    manifest: TrainingIdentity
    heads: list[HeadCheckpoint]


class TensorRepresenter(Protocol):
    """The narrow tensor-to-representation interface used by both public probly samplers."""

    def __call__(self, batch: torch.Tensor) -> Representation: ...

    def represent(self, batch: torch.Tensor) -> Representation: ...


class FittedCondition(TypedDict):
    """A composed model and equivalent lightweight predictors for cached features."""

    subensemble: nn.ModuleList
    heads: nn.ModuleList
    mc_head: nn.Sequential
    record: ConditionRecord
    representers: dict[Method, TensorRepresenter]


class CacheContents(TypedDict, total=False):
    """Tensor and checkpoint fields allowed in the experiment's weights-only caches."""

    features: torch.Tensor
    coalitions: torch.Tensor
    heads: list[HeadCheckpoint]


class CachePayload(CacheContents):
    """A cache payload with required identity metadata."""

    manifest: Mapping[str, object]


class ExternalFeatures(TypedDict):
    """The original Oxford feature format, accepted only after explicit validation."""

    manifest: dict[str, object]
    labels: torch.Tensor
    features: torch.Tensor


class _ResizePIL(Protocol):
    def __call__(
        self,
        image: Image.Image,
        size: list[int],
        *,
        interpolation: InterpolationMode,
        antialias: bool,
    ) -> Image.Image: ...


class _CenterCropPIL(Protocol):
    def __call__(self, image: Image.Image, output_size: list[int]) -> Image.Image: ...


class _TorchSubensembleFactory(Protocol):
    def __call__(
        self,
        base: nn.Module,
        *,
        num_heads: int,
        head: nn.Sequential,
        reset_params: bool,
        predictor_type: Literal["logit_classifier"],
    ) -> nn.ModuleList: ...


class _TorchEnsembleFactory(Protocol):
    def __call__(
        self,
        base: nn.Sequential,
        *,
        num_members: int,
        reset_params: bool,
        predictor_type: Literal["logit_classifier"],
    ) -> nn.ModuleList: ...


class _TorchCastFactory(Protocol):
    def __call__(
        self, base: nn.Sequential, *, predictor_type: Literal["logit_classifier"]
    ) -> nn.Sequential: ...


# These documented runtime signatures are narrower than third-party annotations: torchvision's
# functional transforms accept PIL images, and probly's decorator adds predictor_type while its
# Torch dispatch returns real ModuleLists. Keep those bridges at the external API boundary.
_resize_pil = type_cast("_ResizePIL", vision_functions.resize)
_center_crop_pil = type_cast("_CenterCropPIL", vision_functions.center_crop)
_torch_subensemble = type_cast("_TorchSubensembleFactory", subensemble)
_torch_ensemble = type_cast("_TorchEnsembleFactory", ensemble)
_torch_cast = type_cast("_TorchCastFactory", cast)


def available_feature_backends() -> dict[str, bool]:
    """Report locally available feature-extraction backends without benchmarking them."""
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }


def select_feature_device(requested: str) -> str:
    """Resolve auto as CUDA, MPS, then CPU; reject unavailable explicit selections."""
    available = available_feature_backends()
    if requested == "auto":
        return "cuda:0" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if requested in ("cpu", "mps"):
        if not available[requested]:
            msg = f"Requested feature device {requested!r} is unavailable on this machine."
            raise ValueError(msg)
        return requested
    if re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", requested):
        index = int(requested.partition(":")[2] or "0")
        if not available["cuda"] or index >= torch.cuda.device_count():
            msg = f"Requested feature device {requested!r} is unavailable on this machine."
            raise ValueError(msg)
        return f"cuda:{index}"
    msg = "feature_device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'."
    raise ValueError(msg)


def _check_encoder_ready(encoder: nn.Module, device: str, batch_size: int) -> None:
    """Run the configured float32 VGG batch once, checking backend readiness and capacity."""
    encoder.to(device)
    batch = torch.zeros((batch_size, 3, 224, 224), dtype=torch.float32, device=device)
    with torch.no_grad(), torch.backends.cudnn.flags(benchmark=False, deterministic=True):
        output = encoder(batch)
    if (
        not isinstance(output, torch.Tensor)
        or output.shape != (batch_size, 4096)
        or not torch.isfinite(output).all()
    ):
        msg = "VGG readiness must produce finite float32 features with shape (batch_size, 4096)."
        raise ValueError(msg)
    if output.dtype != torch.float32:
        msg = "The feature encoder must produce float32 output."
        raise ValueError(msg)


def _known_backend_failure(error: RuntimeError | NotImplementedError, device: str) -> bool:
    """Recognize unsupported operations, backend initialization, and backend memory failures."""
    if isinstance(error, (NotImplementedError, torch.OutOfMemoryError)):
        return True
    message = str(error).lower()
    backend = device.partition(":")[0]
    if backend == "cuda" and any(
        phrase in message for phrase in ("nvidia driver", "no kernel image")
    ):
        return True
    if backend not in message:
        return False
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "not implemented",
            "not currently implemented",
            "could not run",
            "initialization",
            "initialisation",
            "not compiled",
            "not linked",
            "driver",
            "no kernel image",
            "not available",
        )
    )


def prepare_encoder(
    encoder: nn.Module, config: Config, *, selected_device: str | None = None
) -> tuple[nn.Module, RuntimeInfo]:
    """Verify a backend once; auto mode may fall back to CPU on known readiness failures."""
    if config.feature_batch_size <= 0:
        msg = "feature_batch_size must be positive."
        raise ValueError(msg)
    device = (
        select_feature_device(config.feature_device) if selected_device is None else selected_device
    )
    encoder.requires_grad_(requires_grad=False).float().eval()
    fallback_reason = None
    try:
        _check_encoder_ready(encoder, device, config.feature_batch_size)
    except (RuntimeError, NotImplementedError) as error:
        if device == "cpu" or not _known_backend_failure(error, device):
            raise
        if config.feature_device != "auto":
            msg = f"Requested feature device {device!r} failed VGG readiness: {error}"
            raise RuntimeError(msg) from error
        fallback_reason = f"{device}: {type(error).__name__}: {error}"
        warnings.warn(
            f"Automatic feature device failed readiness; using CPU. {fallback_reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        device = "cpu"
        _check_encoder_ready(encoder, device, config.feature_batch_size)
    runtime = RuntimeInfo(
        requested_device=config.feature_device,
        device=device,
        available_backends=available_feature_backends(),
        platform=platform.platform(),
        thread_count=torch.get_num_threads(),
        feature_batch_size=config.feature_batch_size,
        encoder_dtype="float32",
        training_device="cpu",
        training_dtype="float32",
        head_inference_device="cpu",
        head_inference_dtype="float64",
        fallback_reason=fallback_reason,
    )
    return encoder, runtime


def digest(value: object) -> str:
    """Hash reproducible metadata or a tensor without a pickle's incidental ordering."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        content = f"{array.dtype}:{array.shape}:".encode() + array.tobytes()
    else:
        content = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def package_versions() -> dict[str, str]:
    """Record package versions and the pinned probly source when available."""
    packages = ("shapiq", "probly", "torch", "torchvision", "numpy", "Pillow", "scikit-learn")
    result = {name: version(name) for name in packages}
    direct_url = distribution("probly").read_text("direct_url.json")
    if direct_url:
        info = type_cast("dict[str, object]", json.loads(direct_url))
        vcs_info = info.get("vcs_info")
        source = (
            type_cast("Mapping[str, object]", vcs_info).get("commit_id")
            if isinstance(vcs_info, dict)
            else info.get("url")
        )
        if isinstance(source, str):
            result["probly_source"] = source
    helper_path = Path(__file__)
    source_files = (
        helper_path,
        helper_path.with_name("reporting.py"),
        helper_path.with_name("selection.py"),
        helper_path.parents[1] / "_shared" / "torch_cache.py",
    )
    result["experiment_source"] = digest(
        {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    )
    return result


def cached(
    path: Path, manifest: Mapping[str, object], compute: Callable[[], CacheContents]
) -> CachePayload:
    """Reuse a matching weights-only-readable payload; consumers validate its tensor fields."""
    payload, hit = load_or_compute(path, manifest, compute)
    if hit:
        print(f"Cache: {path.name}")
    return type_cast("CachePayload", payload)


def prepare_data(
    topic: Path, config: Config, *, data_root: Path | None = None, download: bool = True
) -> PreparedData:
    """Load the official Oxford splits and fix the training and validation sets."""
    data_root = topic / "data" if data_root is None else data_root
    datasets: dict[Split, OxfordIIITPet] = {
        split: OxfordIIITPet(data_root, split=split, target_types="category", download=download)
        for split in SPLITS
    }
    identities: dict[Split, list[str]] = {}
    labels: dict[Split, torch.Tensor] = {}
    for split, dataset in datasets.items():
        annotation = data_root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
        rows = [
            line.split()
            for line in annotation.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        identities[split] = [row[0] for row in rows]
        labels[split] = torch.tensor([int(row[1]) - 1 for row in rows], dtype=torch.long)
        if len(dataset) != len(rows):
            msg = "Dataset and annotation lengths differ."
            raise ValueError(msg)
    if set(identities["trainval"]) & set(identities["test"]):
        msg = "Official training and test identities overlap."
        raise ValueError(msg)
    classes = list(datasets["trainval"].classes)
    if classes != list(datasets["test"].classes) or len(classes) != 37:
        msg = "Expected the same 37 Oxford Pet classes in both splits."
        raise ValueError(msg)
    train, validation = train_test_split(
        np.arange(len(datasets["trainval"])),
        test_size=0.2,
        random_state=config.seed,
        stratify=labels["trainval"].numpy(),
    )
    return {
        "datasets": datasets,
        "identities": identities,
        "labels": labels,
        "classes": classes,
        "train": torch.as_tensor(train),
        "validation": torch.as_tensor(validation),
    }


def load_encoder(
    topic: Path, config: Config, *, weights_root: Path | None = None
) -> tuple[nn.Module, RuntimeInfo]:
    """Load fixed VGG weights and verify the selected feature-extraction backend."""
    selected_device = select_feature_device(config.feature_device)
    weights_root = topic / "models" / "torch_hub" if weights_root is None else weights_root
    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(str(weights_root))
    try:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    finally:
        torch.hub.set_dir(original_hub_dir)
    model.classifier[-1] = nn.Identity()
    return prepare_encoder(model, config, selected_device=selected_device)


def crop_image(image: Image.Image) -> RGBImage:
    """Apply the VGG weights' PIL spatial transform once, before coalition masking."""
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    resized = _resize_pil(
        image.convert("RGB"),
        transform.resize_size,
        interpolation=transform.interpolation,
        antialias=transform.antialias,
    )
    return np.asarray(_center_crop_pil(resized, transform.crop_size), dtype=np.uint8).copy()


def encode_crops(
    encoder: nn.Module, crops: Iterable[RGBImage], batch_size: int = 4
) -> torch.Tensor:
    """Encode cropped RGB images once, using scoped deterministic cuDNN convolution settings."""
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    parameter = next(encoder.parameters())
    outputs: list[torch.Tensor] = []
    pending: list[RGBImage] = []

    def encode_batch(images: list[RGBImage]) -> torch.Tensor:
        tensor = torch.stack(
            [vision_functions.to_tensor(Image.fromarray(image)) for image in images]
        )
        tensor = vision_functions.normalize(tensor, transform.mean, transform.std)
        with torch.no_grad(), torch.backends.cudnn.flags(benchmark=False, deterministic=True):
            features = encoder(tensor.to(device=parameter.device, dtype=parameter.dtype))
        if not isinstance(features, torch.Tensor):
            msg = "The VGG encoder must return a tensor."
            raise TypeError(msg)
        return features.detach().cpu().float()

    for crop in crops:
        pending.append(crop)
        if len(pending) == batch_size:
            outputs.append(encode_batch(pending))
            pending = []
    if pending:
        outputs.append(encode_batch(pending))
    features = torch.cat(outputs)
    if features.ndim != 2 or features.shape[1] != 4096 or not torch.isfinite(features).all():
        msg = "Expected finite VGG features with shape (n, 4096)."
        raise ValueError(msg)
    return features


def _feature_runtime(encoder: nn.Module, config: Config) -> dict[str, object]:
    """Record actual encoder placement and evaluation settings in feature cache identities."""
    parameter = next(encoder.parameters())
    return {
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "batch_size": config.feature_batch_size,
        "threads": torch.get_num_threads(),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def split_features(
    topic: Path,
    data: PreparedData,
    encoder: nn.Module,
    config: Config,
    *,
    external_features: Mapping[Split, Path] | None = None,
) -> FrozenFeatures:
    """Cache shared frozen features, or validate explicitly supplied original feature files."""
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    outputs: dict[Split, torch.Tensor] = {}
    provenance: dict[Split, dict[str, object]] = {}
    for split, dataset in data["datasets"].items():
        manifest: dict[str, object] = {
            "schema": 2,
            "split": split,
            "ids": data["identities"][split],
            "weights": str(VGG16_Weights.IMAGENET1K_V1),
            "preprocessing": repr(transform),
            "runtime": _feature_runtime(encoder, config),
            "versions": {name: version(name) for name in ("torch", "torchvision", "Pillow")},
        }
        if external_features and split in external_features:
            payload = type_cast(
                "ExternalFeatures",
                torch.load(external_features[split], map_location="cpu", weights_only=True),
            )
            source = payload["manifest"]
            preprocessing = source.get("preprocessing")
            if (
                source.get("image_ids") != manifest["ids"]
                or source.get("classes") != data["classes"]
                or source.get("weights") != manifest["weights"]
                or not isinstance(preprocessing, dict)
                or type_cast("Mapping[str, object]", preprocessing).get("description")
                != manifest["preprocessing"]
            ):
                msg = f"Explicit external {split} features describe another experiment."
                raise ValueError(msg)
            if not torch.equal(payload["labels"], data["labels"][split]):
                msg = "External feature labels differ from the official split."
                raise ValueError(msg)
            features = payload["features"]
            provenance[split] = {"origin": "explicit external features", **source}
            print(f"Explicit read-only features: {split}")
        else:
            cached_payload = cached(
                topic / "outputs" / f"{split}_features.pt",
                manifest,
                lambda dataset=dataset: {
                    "features": encode_crops(
                        encoder,
                        (crop_image(dataset[i][0]) for i in range(len(dataset))),
                        config.feature_batch_size,
                    )
                },
            )
            features = cached_payload["features"]
            provenance[split] = {"origin": "notebook extraction", **manifest}
        if (
            features.shape != (len(dataset), 4096)
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            msg = f"Expected finite float32 {split} features with shape ({len(dataset)}, 4096)."
            raise ValueError(msg)
        outputs[split] = features
        provenance[split]["feature_digest"] = digest(features)
    return {"trainval": outputs["trainval"], "test": outputs["test"], "provenance": provenance}


def new_head(seed: int, config: Config) -> nn.Sequential:
    """Initialize a small head without disturbing the caller's CPU random stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(nn.Dropout(config.dropout), nn.Linear(4096, 37))


def train_head(
    head: nn.Sequential,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    validation: torch.Tensor,
    seed: int,
    config: Config,
) -> HeadCheckpoint:
    """Fit one independently initialized probly head and select its best CPU checkpoint."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head.cpu().float()
        loader = DataLoader(
            TensorDataset(features[indices], labels[indices]),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        history: list[float] = []
        for epoch in range(1, config.max_epochs + 1):
            head.train()
            for batch, target in loader:
                optimizer.zero_grad(set_to_none=True)
                criterion(head(batch), target).backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                loss = float(criterion(head(features[validation]), labels[validation]))
            if not np.isfinite(loss):
                msg = "Training produced a non-finite validation loss."
                raise ValueError(msg)
            history.append(loss)
            if loss < best_loss:
                best_loss, best_epoch, stale = loss, epoch, 0
                best_state = {
                    name: value.detach().clone() for name, value in head.state_dict().items()
                }
            else:
                stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        msg = "Training produced no valid checkpoint."
        raise RuntimeError(msg)
    return {"state": best_state, "best_epoch": best_epoch, "validation_loss": history}


def training_identity(
    config: Config, train_ids: list[str], validation_ids: list[str], feature_digest: str
) -> TrainingIdentity:
    """Separate head training from feature placement, display choices, and MC sample count."""
    versions = package_versions()
    return {
        "schema": 2,
        "config": {
            "architecture": "dropout-linear-4096x37-v1",
            "initialization_seed": config.seed,
            "head_seeds": list(config.head_seeds),
            "dropout": config.dropout,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dtype": "float32",
        },
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "features": feature_digest,
        "versions": {
            name: versions[name]
            for name in ("torch", "probly", "probly_source")
            if name in versions
        },
    }


def _member_head(member: nn.Module) -> nn.Sequential:
    """Check the Torch subensemble dispatch's documented backbone/head composition."""
    if not isinstance(member, nn.Sequential) or not isinstance(member[1], nn.Sequential):
        msg = "The probly Torch subensemble must contain Sequential(backbone, head) members."
        raise TypeError(msg)
    return member[1]


def fitted_conditions(
    topic: Path,
    data: PreparedData,
    features: FrozenFeatures,
    encoder: nn.Module,
    config: Config,
) -> dict[Condition, FittedCondition]:
    """Train genuine subensemble heads and reuse equivalent predictors for cached features."""
    rng = np.random.default_rng(config.subset_seed)
    labels = data["labels"]["trainval"]
    scarce: list[int] = []
    for breed in range(37):
        candidates = data["train"][labels[data["train"]] == breed].numpy()
        if config.scarce_per_class > len(candidates):
            msg = "scarce_per_class exceeds the training examples of one breed."
            raise ValueError(msg)
        scarce.extend(rng.choice(candidates, size=config.scarce_per_class, replace=False).tolist())
    results: dict[Condition, FittedCondition] = {}
    selections: tuple[tuple[Condition, torch.Tensor], ...] = (
        ("full", data["train"]),
        ("scarce", torch.tensor(sorted(scarce))),
    )
    feature_digest = digest(features["trainval"])
    for name, indices in selections:
        encoder_device = next(encoder.parameters()).device
        encoder.cpu()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.seed)
                model = _torch_subensemble(
                    encoder,
                    num_heads=len(config.head_seeds),
                    head=new_head(config.seed, config),
                    reset_params=True,
                    predictor_type="logit_classifier",
                )
        finally:
            encoder.to(encoder_device)
        manifest = training_identity(
            config,
            [data["identities"]["trainval"][i] for i in indices.tolist()],
            [data["identities"]["trainval"][i] for i in data["validation"].tolist()],
            feature_digest,
        )
        cached_record = cached(
            topic / "models" / f"{name}_heads.pt",
            manifest,
            lambda indices=indices, model=model: {
                "heads": [
                    train_head(
                        _member_head(member),
                        features["trainval"],
                        labels,
                        indices,
                        data["validation"],
                        seed,
                        config,
                    )
                    for member, seed in zip(model, config.head_seeds, strict=True)
                ]
            },
        )
        record: ConditionRecord = {"manifest": manifest, "heads": cached_record["heads"]}
        heads = _torch_ensemble(
            new_head(config.head_seeds[0], config),
            num_members=len(config.head_seeds),
            reset_params=False,
            predictor_type="logit_classifier",
        )
        for member, head, fitted in zip(model, heads, record["heads"], strict=True):
            _member_head(member).load_state_dict(fitted["state"])
            _member_head(member).to(encoder_device)
            head.load_state_dict(fitted["state"])
        model.eval()
        heads.double().cpu().eval()
        mc_head = _torch_cast(
            new_head(config.head_seeds[0], config), predictor_type="logit_classifier"
        )
        mc_head.load_state_dict(record["heads"][0]["state"])
        mc_head.double().cpu().eval()
        results[name] = {
            "subensemble": model,
            "heads": heads,
            "mc_head": mc_head,
            "record": record,
            "representers": {
                "ensemble": type_cast("TensorRepresenter", representer(heads)),
                "mc_dropout": type_cast(
                    "TensorRepresenter", Sampler(mc_head, num_samples=config.mc_samples)
                ),
            },
        }
    return results


def cached_coalition_features(
    topic: Path,
    image_game: ImageProblyGame,
    image_id: str,
    encoder: nn.Module,
    config: Config,
) -> tuple[CoalitionMatrix, torch.Tensor, Callable[[CoalitionMatrix], torch.Tensor]]:
    """Share a single encoded coalition table across models, uncertainty outputs, and plots."""
    n = image_game.n_players
    coalitions = np.array([list(map(bool, row)) for row in np.ndindex(*(2,) * n)], dtype=bool)
    manifest: dict[str, object] = {
        "schema": 3,
        "image_id": image_id,
        "image": digest(torch.from_numpy(image_game.image.copy())),
        "regions": digest(torch.from_numpy(image_game.regions.copy())),
        "masked_reference": digest(torch.from_numpy(image_game.masked_image(coalitions[0]))),
        "runtime": _feature_runtime(encoder, config),
        "versions": {name: version(name) for name in ("torch", "torchvision", "Pillow", "shapiq")},
        "weights": str(VGG16_Weights.IMAGENET1K_V1),
    }
    payload = cached(
        topic / "outputs" / f"{image_id}_coalition_features.pt",
        manifest,
        lambda: {
            "coalitions": torch.from_numpy(coalitions),
            "features": encode_crops(
                encoder,
                (image_game.masked_image(row) for row in coalitions),
                config.feature_batch_size,
            ),
        },
    )
    np.testing.assert_array_equal(payload["coalitions"].numpy(), coalitions)
    table = payload["features"]
    if table.shape != (2**n, 4096) or not torch.isfinite(table).all():
        msg = "Invalid coalition feature cache."
        raise ValueError(msg)
    binary_weights = 1 << np.arange(n - 1, -1, -1)

    def inputs(query: CoalitionMatrix) -> torch.Tensor:
        indices = np.asarray(query, dtype=np.int64) @ binary_weights
        return table[indices].double()

    return coalitions, table, inputs
