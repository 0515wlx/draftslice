"""Versioned artifact bundles.

An artifact bundle is a manifest plus immutable arrays. The manifest carries the
model shape, partition metadata, screener metadata, and calibration record. The
bundle id is a content hash of the manifest (without the id) and every array.

The writer is intentionally small and local: it is a build-time contract, not a
serving database. A runtime refuses a bundle whose manifest hash or array hash
does not match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "artifact/1"


class ArtifactError(ValueError):
    """Raised when a bundle is malformed or fails a hash check."""


def _to_torch(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ArtifactError("torch and numpy are required for artifact arrays") from exc
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, (list, tuple)):
        return torch.tensor(value)
    raise ArtifactError(f"unsupported array type: {type(value).__name__}")


def _hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def array_hash(value: Any) -> str:

    tensor = _to_torch(value)
    return _hash_bytes(tensor.numpy().tobytes())


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_id", None)
    payload.pop("arrays", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _hash_bytes(encoded)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    model: dict[str, Any]
    partition: dict[str, Any] = field(default_factory=dict)
    screener: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    arrays: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_by: str = "draftslice-build"
    git_commit: str = ""
    schema_version: str = ARTIFACT_SCHEMA
    bundle_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA:
            raise ArtifactError(f"unsupported artifact schema: {self.schema_version}")
        if not self.model:
            raise ArtifactError("model metadata is required")
        if "vocab_size" not in self.model or "hidden_size" not in self.model:
            raise ArtifactError("model metadata needs vocab_size and hidden_size")

    def computed_id(self) -> str:
        return _manifest_hash({**asdict(self), "arrays": self.arrays})

    def verify(self, arrays: dict[str, Any]) -> None:
        if self.bundle_id != self.computed_id():
            raise ArtifactError("manifest content hash does not match bundle_id")
        for name, meta in self.arrays.items():
            if name not in arrays:
                raise ArtifactError(f"manifest references missing array {name!r}")
            actual = array_hash(arrays[name])
            if actual != meta.get("hash"):
                raise ArtifactError(f"array {name!r} hash mismatch")


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    manifest: ArtifactManifest
    arrays: dict[str, Any] = field(default_factory=dict)

    def verify(self) -> None:
        self.manifest.verify(self.arrays)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        import torch

        tensors = {name: _to_torch(value) for name, value in self.arrays.items()}
        torch.save(tensors, path / "arrays.pt")
        manifest = asdict(self.manifest)
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ArtifactBundle:
        path = Path(path)
        manifest_path = path / "manifest.json"
        arrays_path = path / "arrays.pt"
        if not manifest_path.exists() or not arrays_path.exists():
            raise ArtifactError("bundle needs manifest.json and arrays.pt")
        import torch

        manifest = ArtifactManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        arrays = torch.load(arrays_path, map_location="cpu", weights_only=True)
        bundle = cls(manifest=manifest, arrays=dict(arrays))
        bundle.verify()
        return bundle


def build_bundle(
    *,
    model: dict[str, Any],
    arrays: dict[str, Any],
    partition: dict[str, Any] | None = None,
    screener: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    created_by: str = "draftslice-build",
    git_commit: str = "",
) -> ArtifactBundle:
    """Hash arrays and return a verified bundle."""

    array_meta = {}
    normalized = {}
    for name, value in arrays.items():
        tensor = _to_torch(value)
        normalized[name] = tensor
        array_meta[name] = {
            "hash": array_hash(tensor),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }
    manifest = ArtifactManifest(
        model=dict(model),
        partition=dict(partition or {}),
        screener=screener,
        calibration=calibration,
        arrays=array_meta,
        created_by=created_by,
        git_commit=git_commit,
    )
    bundle_id = manifest.computed_id()
    manifest = ArtifactManifest(
        model=manifest.model,
        partition=manifest.partition,
        screener=manifest.screener,
        calibration=manifest.calibration,
        arrays=manifest.arrays,
        created_by=manifest.created_by,
        git_commit=manifest.git_commit,
        schema_version=manifest.schema_version,
        bundle_id=bundle_id,
    )
    bundle = ArtifactBundle(manifest=manifest, arrays=normalized)
    bundle.verify()
    return bundle


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactBundle",
    "ArtifactError",
    "ArtifactManifest",
    "array_hash",
    "build_bundle",
]
