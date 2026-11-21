"""Serialize screeners to and from artifact bundles.

The bundle format is intentionally data-only: arrays plus a typed
``manifest.screener`` record. Loaders reconstruct the same policy dataclasses
used by the compiler and the offline baseline runner, so an artifact cannot
silently change the candidate algorithm between training and serving.

Supported kinds: ``full``, ``static``, ``group``, ``router``, ``lowrank``.
The oracle policies are upper bounds and are not deployable, so they are
rejected by the writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..artifact.bundle import ArtifactBundle, ArtifactError, build_bundle
from ..plan.types import TokenSet
from .base import Screener
from .policies import (
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    RouterPolicy,
    StaticPolicy,
)

ARTIFACT_SCREENER_SCHEMA = "screener/1"


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ArtifactError("torch is required for screener artifacts") from exc
    return torch


def _group_arrays(groups: tuple[tuple[int, ...], ...]) -> dict[str, Any]:
    torch = _require_torch()
    flat: list[int] = []
    offsets = [0]
    for group in groups:
        if not group:
            raise ArtifactError("groups must not be empty")
        flat.extend(group)
        offsets.append(len(flat))
    return {
        "group_tokens": torch.tensor(flat, dtype=torch.int64),
        "group_offsets": torch.tensor(offsets, dtype=torch.int64),
    }


def _groups_from_arrays(arrays: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    torch = _require_torch()
    tokens = arrays.get("group_tokens")
    offsets = arrays.get("group_offsets")
    if tokens is None or offsets is None:
        raise ArtifactError("group artifacts need group_tokens and group_offsets")
    tokens = torch.as_tensor(tokens).reshape(-1).tolist()
    offsets = torch.as_tensor(offsets).reshape(-1).tolist()
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ArtifactError("group_offsets do not describe group_tokens")
    groups = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        if stop <= start:
            raise ArtifactError("groups must not be empty")
        groups.append(tuple(int(token) for token in tokens[start:stop]))
    return tuple(groups)


def screener_to_arrays(
    screener: Screener,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(arrays, screener_manifest)`` for a supported screener."""

    torch = _require_torch()
    if isinstance(screener, FullPolicy):
        return {}, {
            "schema": ARTIFACT_SCREENER_SCHEMA,
            "kind": "full",
            "name": screener.name,
            "vocab_size": screener.vocab_size,
        }
    if isinstance(screener, StaticPolicy):
        return {"rows": torch.tensor(screener.rows, dtype=torch.int64)}, {
            "schema": ARTIFACT_SCREENER_SCHEMA,
            "kind": "static",
            "name": screener.name,
        }
    if isinstance(screener, GroupPolicy):
        arrays = _group_arrays(screener.groups)
        arrays["centroids"] = screener.centroids
        return arrays, {
            "schema": ARTIFACT_SCREENER_SCHEMA,
            "kind": "group",
            "name": screener.name,
            "top_c": screener.top_c,
        }
    if isinstance(screener, RouterPolicy):
        arrays = _group_arrays(screener.groups)
        arrays["router_w1"] = screener.w1
        arrays["router_w2"] = screener.w2
        return arrays, {
            "schema": ARTIFACT_SCREENER_SCHEMA,
            "kind": "router",
            "name": screener.name,
            "top_c": screener.top_c,
        }
    if isinstance(screener, LowRankPolicy):
        return {"vr": screener.vr, "a": screener.a}, {
            "schema": ARTIFACT_SCREENER_SCHEMA,
            "kind": "lowrank",
            "name": screener.name,
            "top_m": screener.top_m,
        }
    raise ArtifactError(
        f"screener {type(screener).__name__} cannot be serialized to an artifact"
    )


def build_screener_bundle(
    screener: Screener,
    *,
    vocab_size: int,
    hidden_size: int,
    calibration: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    created_by: str = "draftslice-screen",
    git_commit: str = "",
) -> ArtifactBundle:
    """Wrap a screener in a verified artifact bundle."""

    if vocab_size <= 0 or hidden_size <= 0:
        raise ArtifactError("vocab_size and hidden_size must be positive")
    arrays, manifest = screener_to_arrays(screener)
    model_meta = dict(model or {})
    model_meta.setdefault("vocab_size", int(vocab_size))
    model_meta.setdefault("hidden_size", int(hidden_size))
    return build_bundle(
        model=model_meta,
        arrays=arrays,
        screener=manifest,
        calibration=calibration,
        created_by=created_by,
        git_commit=git_commit,
    )


def screener_from_bundle(bundle: ArtifactBundle, *, device: Any = None) -> Screener:
    """Reconstruct a screener from a loaded and verified bundle."""

    torch = _require_torch()
    meta = bundle.manifest.screener
    if not isinstance(meta, dict):
        raise ArtifactError("bundle has no screener manifest")
    if meta.get("schema") not in (None, ARTIFACT_SCREENER_SCHEMA):
        raise ArtifactError(f"unsupported screener schema: {meta.get('schema')}")
    kind = meta.get("kind")
    arrays = bundle.arrays

    def tensor(name: str) -> Any:
        if name not in arrays:
            raise ArtifactError(f"screener artifact is missing array {name!r}")
        value = torch.as_tensor(arrays[name])
        return value if device is None else value.to(device)

    if kind == "full":
        vocab_size = int(meta.get("vocab_size") or bundle.manifest.model["vocab_size"])
        return FullPolicy(name=str(meta.get("name", "full")), vocab_size=vocab_size)
    if kind == "static":
        return StaticPolicy(
            rows=tuple(int(row) for row in tensor("rows").reshape(-1).tolist()),
            name=str(meta.get("name", "static")),
        )
    if kind == "group":
        return GroupPolicy(
            groups=_groups_from_arrays(arrays),
            centroids=tensor("centroids"),
            top_c=int(meta["top_c"]),
            name=str(meta.get("name", "group_dot")),
        )
    if kind == "router":
        return RouterPolicy(
            groups=_groups_from_arrays(arrays),
            w1=tensor("router_w1"),
            w2=tensor("router_w2"),
            top_c=int(meta["top_c"]),
            name=str(meta.get("name", "mlp_router")),
        )
    if kind == "lowrank":
        return LowRankPolicy(
            vr=tensor("vr"),
            a=tensor("a"),
            top_m=int(meta["top_m"]),
            name=str(meta.get("name", "lowrank")),
        )
    raise ArtifactError(f"unsupported screener kind: {kind!r}")


def load_screener_bundle(
    path: str | Path,
    *,
    device: Any = None,
) -> tuple[Screener, ArtifactBundle]:
    """Load a verified bundle and return ``(screener, bundle)``."""

    bundle = ArtifactBundle.load(path)
    return screener_from_bundle(bundle, device=device), bundle


def screener_calibration(bundle: ArtifactBundle) -> dict[str, Any]:
    """Return the calibration metadata (empty when the bundle has none)."""

    return dict(bundle.manifest.calibration or {})


def static_rows_from_tokens(tokens: Any) -> TokenSet:
    """Convenience helper for building a static screener from a token array."""

    return TokenSet(int(token) for token in tokens)


__all__ = [
    "ARTIFACT_SCREENER_SCHEMA",
    "build_screener_bundle",
    "load_screener_bundle",
    "screener_calibration",
    "screener_from_bundle",
    "screener_to_arrays",
    "static_rows_from_tokens",
]
