"""Verify that Oxford Pets can reuse existing artifacts and retry failed cache writes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments._shared.torch_cache import load_or_compute  # noqa: E402


def test_reuses_existing_flat_payload(tmp_path):
    path = tmp_path / "trainval_features.pt"
    manifest = {"schema": 2, "ids": ["sample"]}
    features = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    torch.save({"manifest": manifest, "features": features}, path)

    def compute():
        pytest.fail("A matching cache should be reused")

    loaded, hit = load_or_compute(path, manifest, compute)

    assert hit
    assert loaded["manifest"] == manifest
    assert loaded["features"].device.type == "cpu"
    torch.testing.assert_close(loaded["features"], features)


def test_replaces_cache_on_manifest_change(tmp_path):
    path = tmp_path / "outputs" / "test_features.pt"
    features = torch.tensor([[3.0, 4.0]])
    _, first_hit = load_or_compute(
        path, {"schema": 2}, lambda: {"features": torch.zeros_like(features)}
    )
    _, changed_hit = load_or_compute(path, {"schema": 3}, lambda: {"features": features})

    assert not first_hit
    assert not changed_hit
    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert stored["manifest"] == {"schema": 3}
    torch.testing.assert_close(stored["features"], features)


def test_failed_write_preserves_previous_cache(tmp_path, monkeypatch):
    path = tmp_path / "test_features.pt"
    previous = torch.tensor([2.0])
    torch.save({"manifest": {"schema": 2}, "features": previous}, path)

    def fail_after_partial_write(_payload, handle):
        handle.write(b"incomplete")
        message = "simulated short write"
        raise OSError(message)

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated short write"):
        load_or_compute(path, {"schema": 3}, lambda: {"features": torch.zeros_like(previous)})

    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert stored["manifest"] == {"schema": 2}
    torch.testing.assert_close(stored["features"], previous)
    assert list(tmp_path.iterdir()) == [path]
