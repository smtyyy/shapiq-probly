"""Offline checks for the notebook's model, hardware and cache orchestration.

These tests use tiny encoders and local feature tensors. They neither execute the notebook nor
download Oxford data or VGG weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("probly")

from experiments.oxford_pets_uncertainty import helpers  # noqa: E402

from shapiq_games.probly import ImageProblyGame, ProblyPredictor  # noqa: E402


class TinyEncoder(torch.nn.Module):
    """Produce VGG-shaped features with one scalar parameter and no convolution weights."""

    def __init__(self, width=4096):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.width = width
        self.seen = []

    def forward(self, images):
        self.seen.append((tuple(images.shape), images.dtype, images.device.type, self.training))
        return images.mean(dim=(1, 2, 3))[:, None].expand(-1, self.width) + self.weight


def _availability(monkeypatch, *, cuda=False, mps=False, mps_built=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if cuda else 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: mps_built)


@pytest.mark.parametrize(
    ("cuda", "mps", "mps_built", "expected"),
    [
        (True, True, True, "cuda:0"),
        (False, True, True, "mps"),
        (False, False, True, "cpu"),
        (False, False, False, "cpu"),
    ],
)
def test_auto_selects_a_usable_backend_in_priority_order(
    monkeypatch, cuda, mps, mps_built, expected
):
    _availability(monkeypatch, cuda=cuda, mps=mps, mps_built=mps_built)
    assert helpers.select_feature_device("auto") == expected
    assert helpers.available_feature_backends()["mps"] is mps


@pytest.mark.parametrize("requested", ["cpu", "mps", "cuda:1"])
def test_explicit_device_selection_overrides_the_automatic_preference(monkeypatch, requested):
    _availability(monkeypatch, cuda=True, mps=True)
    assert helpers.select_feature_device(requested) == requested


@pytest.mark.parametrize("requested", ["mps", "cuda", "cuda:0"])
def test_explicit_unavailable_devices_raise_instead_of_silently_selecting_cpu(
    monkeypatch, requested
):
    _availability(monkeypatch, cuda=False, mps=False, mps_built=True)
    with pytest.raises(ValueError):
        helpers.select_feature_device(requested)


def test_cpu_readiness_checks_a_real_forward_and_prepares_float32_frozen_features(monkeypatch):
    _availability(monkeypatch)
    encoder = TinyEncoder()
    encoder.train()
    config = helpers.Config(feature_device="auto", feature_batch_size=2)
    model, runtime = helpers.prepare_encoder(encoder, config)

    assert model is encoder
    assert encoder.seen == [((2, 3, 224, 224), torch.float32, "cpu", False)]
    assert not encoder.training
    assert encoder.weight.dtype == torch.float32
    assert not encoder.weight.requires_grad
    assert runtime.device == "cpu"
    assert runtime.fallback_reason is None
    assert "float32" in runtime.encoder_dtype
    assert runtime.training_device == runtime.head_inference_device == "cpu"
    assert "float64" in runtime.head_inference_dtype
    assert runtime.feature_batch_size == 2
    features = helpers.encode_crops(
        encoder, [np.full((224, 224, 3), value, dtype=np.uint8) for value in (0, 128, 255)], 2
    )
    assert features.shape == (3, 4096)
    assert features.dtype == torch.float32
    assert features.device.type == "cpu"
    assert torch.isfinite(features).all()
    # Provenance must survive the notebook's JSON result export.
    provenance = json.loads(json.dumps(asdict(runtime), allow_nan=False))
    assert provenance["device"] == "cpu"
    assert provenance["available_backends"]["mps"] is False


@pytest.mark.parametrize(
    "failure",
    [
        NotImplementedError("test operator is not implemented for the MPS device"),
        RuntimeError("MPS backend out of memory"),
        RuntimeError("PyTorch is not linked with support for mps devices"),
    ],
)
def test_auto_readiness_failure_falls_back_and_records_the_actual_cpu_execution(
    monkeypatch, failure
):
    _availability(monkeypatch, mps=True)
    attempted = []

    def probe(encoder, device, batch_size):
        attempted.append(device)
        if device == "mps":
            raise failure

    monkeypatch.setattr(helpers, "_check_encoder_ready", probe)
    with pytest.warns(RuntimeWarning):
        _, runtime = helpers.prepare_encoder(TinyEncoder(), helpers.Config(feature_device="auto"))

    assert attempted == ["mps", "cpu"]
    assert runtime.requested_device == "auto"
    assert runtime.device == "cpu"
    assert runtime.available_backends["mps"] is True
    assert str(failure) in runtime.fallback_reason
    provenance = json.loads(json.dumps(asdict(runtime)))
    assert provenance["device"] == "cpu"
    assert provenance["fallback_reason"]


def test_explicit_backend_readiness_failure_is_reported_without_cpu_fallback(monkeypatch):
    _availability(monkeypatch, mps=True)
    attempted = []
    failure = NotImplementedError("test operator is not implemented for the MPS device")

    def probe(encoder, device, batch_size):
        attempted.append(device)
        raise failure

    monkeypatch.setattr(helpers, "_check_encoder_ready", probe)
    with pytest.raises(RuntimeError) as caught:
        helpers.prepare_encoder(TinyEncoder(), helpers.Config(feature_device="mps"))
    assert attempted == ["mps"]
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "failure",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("no kernel image is available for execution on the device"),
        RuntimeError("Found no NVIDIA driver on your system"),
    ],
)
def test_auto_cuda_backend_failures_fall_back_to_cpu_without_probing_other_accelerators(
    monkeypatch, failure
):
    _availability(monkeypatch, cuda=True, mps=True)
    attempted = []

    def probe(encoder, device, batch_size):
        attempted.append(device)
        if device == "cuda:0":
            raise failure

    monkeypatch.setattr(helpers, "_check_encoder_ready", probe)
    with pytest.warns(RuntimeWarning):
        _, runtime = helpers.prepare_encoder(TinyEncoder(), helpers.Config(feature_device="auto"))
    assert attempted == ["cuda:0", "cpu"]
    assert runtime.device == "cpu"
    assert str(failure) in runtime.fallback_reason


def test_cpu_readiness_failures_propagate_without_a_retry(monkeypatch):
    _availability(monkeypatch)
    attempted = []
    failure = RuntimeError("CPU allocator ran out of memory")

    def probe(encoder, device, batch_size):
        attempted.append(device)
        raise failure

    monkeypatch.setattr(helpers, "_check_encoder_ready", probe)
    with pytest.raises(RuntimeError) as caught:
        helpers.prepare_encoder(TinyEncoder(), helpers.Config(feature_device="auto"))
    assert caught.value is failure
    assert attempted == ["cpu"]


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("feature vectors have the wrong shape"),
        RuntimeError("matrix sizes do not match"),
        RuntimeError("MPS feature values are non-finite"),
    ],
)
def test_auto_selection_does_not_hide_unrelated_model_errors(monkeypatch, failure):
    _availability(monkeypatch, mps=True)
    attempted = []

    def probe(encoder, device, batch_size):
        attempted.append(device)
        raise failure

    monkeypatch.setattr(helpers, "_check_encoder_ready", probe)
    with pytest.raises(type(failure)) as caught:
        helpers.prepare_encoder(TinyEncoder(), helpers.Config(feature_device="auto"))
    assert caught.value is failure
    assert attempted == ["mps"]


def test_real_encoder_shape_errors_are_not_reclassified_as_backend_failures(monkeypatch):
    _availability(monkeypatch)
    with pytest.raises(ValueError):
        helpers.prepare_encoder(TinyEncoder(width=3), helpers.Config(feature_device="auto"))


@pytest.mark.parametrize("fail", [False, True])
def test_load_encoder_restores_torch_hub_path(monkeypatch, tmp_path, fail):
    original_path = torch.hub.get_dir()
    weights_root = tmp_path / "weights"
    model = TinyEncoder()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(1, 1))

    def load_model(*, weights):
        assert weights == helpers.VGG16_Weights.IMAGENET1K_V1
        assert torch.hub.get_dir() == str(weights_root)
        if fail:
            msg = "unavailable weights"
            raise OSError(msg)
        return model

    monkeypatch.setattr(helpers, "vgg16", load_model)
    config = helpers.Config(feature_device="cpu")
    if fail:
        with pytest.raises(OSError, match="unavailable weights"):
            helpers.load_encoder(tmp_path, config, weights_root=weights_root)
    else:
        encoder, _ = helpers.load_encoder(tmp_path, config, weights_root=weights_root)
        assert encoder is model
        assert isinstance(encoder.classifier[-1], torch.nn.Identity)

    assert torch.hub.get_dir() == original_path


def test_external_features_require_float32(tmp_path):
    labels = torch.tensor([0])
    splits = ("trainval", "test")
    data = {
        "datasets": {split: [None] for split in splits},
        "identities": {split: [f"{split}_pet"] for split in splits},
        "classes": ["pet"],
        "labels": dict.fromkeys(splits, labels),
    }
    weights = helpers.VGG16_Weights.IMAGENET1K_V1
    manifest = {
        "classes": ["pet"],
        "weights": str(weights),
        "preprocessing": {"description": repr(weights.transforms())},
    }
    sources = {split: tmp_path / f"{split}.pt" for split in splits}
    payload = {
        "manifest": {**manifest, "image_ids": ["trainval_pet"]},
        "labels": labels,
        "features": torch.zeros((1, 4096), dtype=torch.float64),
    }
    torch.save(payload, sources["trainval"])
    encoder = TinyEncoder().float()
    config = helpers.Config(feature_device="cpu")

    with pytest.raises(ValueError, match="finite float32 trainval features"):
        helpers.split_features(tmp_path, data, encoder, config, external_features=sources)

    payload["features"] = payload["features"].float()
    torch.save(payload, sources["trainval"])
    torch.save({**payload, "manifest": {**manifest, "image_ids": ["test_pet"]}}, sources["test"])
    features = helpers.split_features(tmp_path, data, encoder, config, external_features=sources)
    torch.testing.assert_close(features["trainval"], payload["features"])


def test_coalition_cache_reuses_identical_fill_and_reencodes_changed_fill(monkeypatch, tmp_path):
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    regions = np.zeros((2, 2), dtype=int)
    predictor = ProblyPredictor(
        lambda rows: rows,
        output="brightness",
        quantifier=lambda rows: {"brightness": rows.mean(axis=(1, 2, 3))},
    )
    encoder = TinyEncoder().float()
    config = helpers.Config(feature_device="cpu")
    encoded: list[tuple[float, ...]] = []

    def encode(_encoder, crops, _batch_size):
        brightness = tuple(float(crop.mean()) for crop in crops)
        encoded.append(brightness)
        return torch.tensor(brightness, dtype=torch.float32)[:, None].repeat(1, 4096)

    monkeypatch.setattr(helpers, "encode_crops", encode)
    for fill, expected in (("black", 0.0), ("black", 0.0), ("mean", 255.0)):
        game = ImageProblyGame(image, predictor, regions=regions, fill=fill)
        _, table, _ = helpers.cached_coalition_features(tmp_path, game, "pet", encoder, config)
        assert table[:, 0].tolist() == [expected, 255.0]

    assert encoded == [(0.0, 255.0), (255.0, 255.0)]


def test_runtime_changes_reuse_fitted_heads_but_scientific_changes_retrain(monkeypatch, tmp_path):
    config = helpers.Config(
        head_seeds=(0,), scarce_per_class=1, mc_samples=2, max_epochs=1, feature_device="cpu"
    )
    labels = torch.cat([torch.arange(37), torch.arange(37)])
    data = {
        "labels": {"trainval": labels},
        "train": torch.arange(37),
        "validation": torch.arange(37, 74),
        "identities": {"trainval": [f"breed_{i % 37}_image_{i}" for i in range(74)]},
    }
    features = {"trainval": torch.zeros((74, 4096), dtype=torch.float32)}
    fitted = []

    def fit(head, features, labels, indices, validation, seed, config):
        assert all(parameter.dtype == torch.float32 for parameter in head.parameters())
        fitted.append((tuple(indices.tolist()), helpers.digest(features)))
        return {
            "state": {name: value.detach().clone() for name, value in head.state_dict().items()},
            "best_epoch": 1,
            "validation_loss": [1.0],
        }

    monkeypatch.setattr(helpers, "train_head", fit)
    versions = {**helpers.package_versions(), "probly_source": "training-source-revision-a"}
    monkeypatch.setattr(helpers, "package_versions", lambda: versions)
    encoder = TinyEncoder().float()
    conditions = helpers.fitted_conditions(tmp_path, data, features, encoder, config)
    assert len(fitted) == 2
    for condition in conditions.values():
        for name in ("heads", "mc_head"):
            assert all(
                parameter.dtype == torch.float64 for parameter in condition[name].parameters()
            )
            assert all(parameter.device.type == "cpu" for parameter in condition[name].parameters())

    runtime_change = replace(
        config, feature_device="auto", feature_batch_size=2, n_images=7, mc_samples=3
    )
    # A runtime helper revision belongs in provenance, not the trained model's identity.
    versions = {**versions, "experiment_source": "a different hardware-only helper revision"}
    helpers.fitted_conditions(tmp_path, data, features, encoder, runtime_change)
    assert len(fitted) == 2

    versions = {**versions, "probly_source": "training-source-revision-b"}
    helpers.fitted_conditions(tmp_path, data, features, encoder, runtime_change)
    assert len(fitted) == 4

    changed_features = {"trainval": features["trainval"].clone()}
    changed_features["trainval"][0, 0] = 1.0
    helpers.fitted_conditions(tmp_path, data, changed_features, encoder, runtime_change)
    assert len(fitted) == 6

    changed_science = replace(runtime_change, dropout=0.25)
    helpers.fitted_conditions(tmp_path, data, changed_features, encoder, changed_science)
    assert len(fitted) == 8
