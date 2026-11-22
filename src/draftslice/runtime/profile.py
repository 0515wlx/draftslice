"""Profitability table (``profiles/1``).

The table is the production switch: DraftSlice executes only for an *exact*
enabled cell. There is no "nearest cell" fallback — a miss, a disabled cell, or
an engine mismatch returns the built-in path with a stable reason code
(``docs/production-architecture.md`` D2).

The artifact is hashed so a table cannot be edited between the gate run and the
deployment: :meth:`ProfileTable.load` raises :class:`ProfileHashError` on a
mismatch instead of silently serving a different table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reasons import FallbackReason

SCHEMA = "profiles/1"

ENGINES = frozenset({"vllm", "sglang"})
METHODS = frozenset(
    {
        "builtin",
        "mtp",
        "eagle3",
        "dflash",
        "draftslice_mtp",
        "draftslice_eagle3",
        "draftslice_dflash",
    }
)
ACTIONS = frozenset({"builtin", "strict_slice", "mask", "mask_allpos"})
QUANTS = frozenset({"bf16", "fp16", "fp32", "fp8", "int8", "nvfp4", "auto"})
GRAPHS = frozenset({"eager", "step0", "full_cuda_graph"})

BATCH_BUCKETS: tuple[str, ...] = ("1", "2-8", "9-32", "33-64", ">64")
VALID_BUCKETS: tuple[str, ...] = ("0-64", "65-512", "513-4096", ">4096")


class ProfileError(ValueError):
    """Base class for profile table failures."""


class ProfileParseError(ProfileError):
    """The artifact is not a valid ``profiles/1`` document."""


class ProfileValidationError(ProfileError):
    """A cell or table field violates the schema."""


class ProfileHashError(ProfileError):
    """The artifact hash does not match its content."""


def batch_bucket(batch: int) -> str:
    """Bucket a concrete decode batch size for the profile table."""

    if batch < 1:
        raise ProfileValidationError("batch must be >= 1")
    if batch == 1:
        return "1"
    if batch <= 8:
        return "2-8"
    if batch <= 32:
        return "9-32"
    if batch <= 64:
        return "33-64"
    return ">64"


def valid_bucket(count: int) -> str:
    """Bucket the exact valid-set size of a step."""

    if count < 0:
        raise ProfileValidationError("valid-set size must be >= 0")
    if count <= 64:
        return "0-64"
    if count <= 512:
        return "65-512"
    if count <= 4096:
        return "513-4096"
    return ">4096"


def _normalize_bucket(value: Any, allowed: tuple[str, ...], *, name: str) -> str:
    if isinstance(value, bool):
        raise ProfileValidationError(f"{name} must be a bucket label or count")
    if isinstance(value, int):
        return batch_bucket(value) if name == "batch_bucket" else valid_bucket(value)
    text = str(value).strip()
    if text not in allowed:
        raise ProfileValidationError(
            f"invalid {name} {text!r}; expected one of {', '.join(allowed)}"
        )
    return text


@dataclass(frozen=True, slots=True)
class ProfitCell:
    """One measured ``(model, engine, workload, batch, k, ...)`` operating point."""

    model: str
    engine: str
    method: str
    workload: str
    batch_bucket: str
    k: int
    tp: int
    quant: str
    graph: str
    valid_bucket: str
    action: str
    expected_gain: float
    measured_p99_ratio: float
    manifest_id: str
    enabled: bool

    def __post_init__(self) -> None:
        for name in ("model", "workload", "manifest_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProfileValidationError(f"{name} must be a non-empty string")
        if self.engine not in ENGINES:
            raise ProfileValidationError(f"invalid engine: {self.engine!r}")
        if self.method not in METHODS:
            raise ProfileValidationError(f"invalid method: {self.method!r}")
        if self.action not in ACTIONS:
            raise ProfileValidationError(f"invalid action: {self.action!r}")
        if self.quant not in QUANTS:
            raise ProfileValidationError(f"invalid quant: {self.quant!r}")
        if self.graph not in GRAPHS:
            raise ProfileValidationError(f"invalid graph: {self.graph!r}")
        if self.k < 0:
            raise ProfileValidationError("k must be >= 0")
        if self.tp < 1:
            raise ProfileValidationError("tp must be >= 1")
        if self.expected_gain <= 0:
            raise ProfileValidationError("expected_gain must be positive")
        if self.measured_p99_ratio <= 0:
            raise ProfileValidationError("measured_p99_ratio must be positive")
        if not isinstance(self.enabled, bool):
            raise ProfileValidationError("enabled must be a bool")
        if not isinstance(self.batch_bucket, str) or not isinstance(
            self.valid_bucket, str
        ):
            raise ProfileValidationError("buckets must be labels, not counts")
        _normalize_bucket(self.batch_bucket, BATCH_BUCKETS, name="batch_bucket")
        _normalize_bucket(self.valid_bucket, VALID_BUCKETS, name="valid_bucket")

    @property
    def cell_id(self) -> str:
        """Stable identity used by traces and gates."""

        return "|".join(
            (
                self.model,
                self.engine,
                self.method,
                self.workload,
                self.batch_bucket,
                f"k{self.k}",
                f"tp{self.tp}",
                self.quant,
                self.graph,
                self.valid_bucket,
            )
        )

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.engine,
            self.method,
            self.workload,
            self.batch_bucket,
            int(self.k),
            int(self.tp),
            self.quant,
            self.graph,
            self.valid_bucket,
        )

    @property
    def is_profitable(self) -> bool:
        """Whether the cell may execute DraftSlice when matched."""

        return self.enabled and self.action != "builtin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "engine": self.engine,
            "method": self.method,
            "workload": self.workload,
            "batch_bucket": self.batch_bucket,
            "k": self.k,
            "tp": self.tp,
            "quant": self.quant,
            "graph": self.graph,
            "valid_bucket": self.valid_bucket,
            "action": self.action,
            "expected_gain": self.expected_gain,
            "measured_p99_ratio": self.measured_p99_ratio,
            "manifest_id": self.manifest_id,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProfitCell:
        try:
            return cls(
                model=str(data["model"]),
                engine=str(data["engine"]),
                method=str(data["method"]),
                workload=str(data["workload"]),
                batch_bucket=_normalize_bucket(
                    data["batch_bucket"], BATCH_BUCKETS, name="batch_bucket"
                ),
                k=int(data["k"]),
                tp=int(data["tp"]),
                quant=str(data["quant"]),
                graph=str(data["graph"]),
                valid_bucket=_normalize_bucket(
                    data["valid_bucket"], VALID_BUCKETS, name="valid_bucket"
                ),
                action=str(data["action"]),
                expected_gain=float(data["expected_gain"]),
                measured_p99_ratio=float(data["measured_p99_ratio"]),
                manifest_id=str(data["manifest_id"]),
                enabled=bool(data["enabled"]),
            )
        except KeyError as exc:
            raise ProfileParseError(f"missing cell field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ProfileError):
                raise
            raise ProfileParseError(f"invalid cell field: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProfileMatch:
    """Result of an exact-cell lookup."""

    hit: bool
    reason: FallbackReason
    cell: ProfitCell | None = None
    table_id: str = ""

    @property
    def cell_id(self) -> str | None:
        return None if self.cell is None else self.cell.cell_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "reason": self.reason.value,
            "cell_id": self.cell_id,
            "table_id": self.table_id,
        }


@dataclass(frozen=True, slots=True)
class ProfileTable:
    """A hashed collection of profit cells for one deployment."""

    cells: tuple[ProfitCell, ...] = ()
    table_id: str = "empty"
    engine: str = ""
    model: str = ""
    created: str = ""
    notes: str = ""
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.table_id:
            raise ProfileValidationError("table_id must not be empty")
        if self.engine and self.engine not in ENGINES:
            raise ProfileValidationError(f"invalid table engine: {self.engine!r}")
        seen: set[tuple[Any, ...]] = set()
        for cell in self.cells:
            if cell.key in seen:
                raise ProfileValidationError(
                    f"duplicate cell for key {cell.cell_id!r}"
                )
            seen.add(cell.key)
            if self.engine and cell.engine != self.engine:
                raise ProfileValidationError(
                    f"cell {cell.cell_id!r} engine differs from table engine"
                )
            if self.model and cell.model != self.model:
                raise ProfileValidationError(
                    f"cell {cell.cell_id!r} model differs from table model"
                )

    # -- construction -----------------------------------------------------
    @classmethod
    def empty(cls, *, table_id: str = "empty") -> ProfileTable:
        """A valid table with no cells: every step uses the built-in path."""

        return cls(cells=(), table_id=table_id, created=_now())

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, require_hash: bool = True
    ) -> ProfileTable:
        if not isinstance(data, Mapping):
            raise ProfileParseError("profile table must be a JSON object")
        if data.get("schema") != SCHEMA:
            raise ProfileParseError(
                f"unsupported profile schema {data.get('schema')!r}; "
                f"expected {SCHEMA!r}"
            )
        raw_cells = data.get("cells", [])
        if not isinstance(raw_cells, list):
            raise ProfileParseError("cells must be a list")
        cells = tuple(ProfitCell.from_dict(cell) for cell in raw_cells)
        table = cls(
            cells=cells,
            table_id=str(data.get("table_id") or "empty"),
            engine=str(data.get("engine") or ""),
            model=str(data.get("model") or ""),
            created=str(data.get("created") or ""),
            notes=str(data.get("notes") or ""),
        )
        stored = data.get("hash")
        if stored is None:
            if require_hash:
                raise ProfileParseError("profile table is missing its hash")
            return table
        table = cls(
            cells=table.cells,
            table_id=table.table_id,
            engine=table.engine,
            model=table.model,
            created=table.created,
            notes=table.notes,
            hash=str(stored),
        )
        table.verify()
        return table

    @classmethod
    def load(cls, path: str | Path, *, require_hash: bool = True) -> ProfileTable:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileParseError(f"cannot read profile table {source}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProfileParseError(f"invalid profile JSON {source}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ProfileParseError("profile table must be a JSON object")
        return cls.from_dict(data, require_hash=require_hash)

    # -- serialization ----------------------------------------------------
    def canonical_dict(self) -> dict[str, Any]:
        """Document without the hash, in a deterministic order."""

        return {
            "schema": SCHEMA,
            "table_id": self.table_id,
            "engine": self.engine,
            "model": self.model,
            "created": self.created,
            "notes": self.notes,
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def digest(self) -> str:
        """``sha256:<hex>`` over the canonical document."""

        payload = json.dumps(
            self.canonical_dict(),
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def with_hash(self) -> ProfileTable:
        return ProfileTable(
            cells=self.cells,
            table_id=self.table_id,
            engine=self.engine,
            model=self.model,
            created=self.created,
            notes=self.notes,
            hash=self.digest(),
        )

    def verify(self) -> None:
        """Raise :class:`ProfileHashError` unless the stored hash is current."""

        expected = self.digest()
        if self.hash != expected:
            raise ProfileHashError(
                f"profile table hash mismatch: stored {self.hash!r}, "
                f"computed {expected!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        document = self.canonical_dict()
        document["hash"] = self.hash or self.digest()
        return document

    def save(self, path: str | Path) -> str:
        """Write the hashed artifact and return its digest."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self.to_dict()
        target.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return document["hash"]

    # -- queries ----------------------------------------------------------
    @property
    def enabled_cells(self) -> tuple[ProfitCell, ...]:
        return tuple(cell for cell in self.cells if cell.is_profitable)

    def cell(self, cell_id: str) -> ProfitCell | None:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def lookup(
        self,
        *,
        model: str,
        engine: str,
        method: str,
        workload: str,
        batch: int | str,
        k: int,
        tp: int,
        quant: str,
        graph: str,
        valid: int | str,
    ) -> ProfileMatch:
        """Exact-cell lookup; never crosses a bucket boundary."""

        try:
            batch_label = _normalize_bucket(batch, BATCH_BUCKETS, name="batch_bucket")
            valid_label = _normalize_bucket(valid, VALID_BUCKETS, name="valid_bucket")
        except ProfileValidationError as exc:
            raise ProfileParseError(str(exc)) from exc
        if self.engine and engine != self.engine:
            return ProfileMatch(
                hit=False, reason=FallbackReason.ENGINE_MISMATCH, table_id=self.table_id
            )
        if self.model and model != self.model:
            return ProfileMatch(
                hit=False, reason=FallbackReason.NO_CELL, table_id=self.table_id
            )
        if not self.cells:
            return ProfileMatch(
                hit=False,
                reason=FallbackReason.PROFILES_UNAVAILABLE,
                table_id=self.table_id,
            )
        key = (
            model,
            engine,
            method,
            workload,
            batch_label,
            int(k),
            int(tp),
            quant,
            graph,
            valid_label,
        )
        for cell in self.cells:
            if cell.key != key:
                continue
            if not cell.is_profitable:
                return ProfileMatch(
                    hit=False,
                    reason=FallbackReason.PROFILE_DISABLED,
                    cell=cell,
                    table_id=self.table_id,
                )
            return ProfileMatch(
                hit=True, reason=FallbackReason.NO_CELL, cell=cell, table_id=self.table_id
            )
        return ProfileMatch(
            hit=False, reason=FallbackReason.NO_CELL, table_id=self.table_id
        )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def build_table(
    cells: Iterable[ProfitCell],
    *,
    table_id: str,
    engine: str = "",
    model: str = "",
    notes: str = "",
) -> ProfileTable:
    """Build and hash a table; convenience for gate/manifest tooling."""

    table = ProfileTable(
        cells=tuple(cells),
        table_id=table_id,
        engine=engine,
        model=model,
        created=_now(),
        notes=notes,
    )
    return table.with_hash()


def load_table(path: str | Path, *, require_hash: bool = True) -> ProfileTable:
    return ProfileTable.load(path, require_hash=require_hash)


__all__ = [
    "ACTIONS",
    "BATCH_BUCKETS",
    "ENGINES",
    "GRAPHS",
    "METHODS",
    "ProfileError",
    "ProfileHashError",
    "ProfileMatch",
    "ProfileParseError",
    "ProfileTable",
    "ProfileValidationError",
    "ProfitCell",
    "QUANTS",
    "SCHEMA",
    "VALID_BUCKETS",
    "batch_bucket",
    "build_table",
    "load_table",
    "valid_bucket",
]
