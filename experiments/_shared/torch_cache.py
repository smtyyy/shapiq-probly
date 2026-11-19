"""Persist computed Torch artifacts using experiment-defined cache manifests."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import mkstemp
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def load_or_compute(
    path: Path,
    manifest: Mapping[str, object],
    compute: Callable[[], Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    """Return an artifact payload and whether it was loaded from the cache.

    Args:
        path: Destination for the experiment's local artifact.
        manifest: Identity of the computation whose result can be reused.
        compute: Produces fields readable by Torch's weights-only loader.
            The cache supplies the reserved `manifest` field.

    Returns:
        The flat payload and a flag indicating that the stored manifest matched.
    """
    identity = dict(manifest)
    try:
        payload = cast("dict[str, object]", torch.load(path, map_location="cpu", weights_only=True))
    except FileNotFoundError:
        pass
    else:
        if payload.get("manifest") == identity:
            return payload, True

    payload = {**compute(), "manifest": identity}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload, False
