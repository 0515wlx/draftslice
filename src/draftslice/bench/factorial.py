"""Factorial screener evaluation over multiple cases and candidate sources.

A factorial run answers "which screener wins on which workload at what cost".
Each (screener, case) cell reports the B1 metrics plus the exact top-k coverage
event used by the conformal calibration, the mean candidate count, and the
candidate bytes for an iso-resource comparison.

The runner is engine-free. Cases come from cached hidden states, so a factorial
matrix can be reproduced from a manifest without running a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..plan.types import TokenSet
from ..screen.base import Screener
from .baseline_runner import BaselineRunnerError, HeadQualityCase, run_policy

FACTORIAL_SCHEMA = "factorial/1"


@dataclass(frozen=True, slots=True)
class FactorialCase:
    """One hidden-state workload slice."""

    name: str
    hidden: Any
    truth: tuple[tuple[int, ...], ...]
    valid_sets: tuple[TokenSet | None, ...] | None = None
    domains: tuple[str, ...] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise BaselineRunnerError("case name must not be empty")
        batch = int(self.hidden.shape[0])
        if len(self.truth) != batch:
            raise BaselineRunnerError(
                f"case {self.name!r}: truth has {len(self.truth)} rows for batch={batch}"
            )
        object.__setattr__(self, "truth", tuple(tuple(row) for row in self.truth))
        if self.valid_sets is not None and len(self.valid_sets) != batch:
            raise BaselineRunnerError(
                f"case {self.name!r}: valid_sets length does not match the batch"
            )
        if self.domains is not None and len(self.domains) != batch:
            raise BaselineRunnerError(
                f"case {self.name!r}: domains length does not match the batch"
            )

    def as_head_case(self) -> HeadQualityCase:
        return HeadQualityCase(
            hidden=self.hidden,
            truth=self.truth,
            domains=self.domains,
            valid_sets=self.valid_sets,
            name=self.name,
            meta=dict(self.meta),
        )


def _coverage(candidates: Sequence[TokenSet], truth: tuple[tuple[int, ...], ...]) -> float:
    if not truth:
        return 0.0
    covered = sum(
        1
        for candidate, row in zip(candidates, truth, strict=True)
        if set(row).issubset(candidate.ids)
    )
    return covered / len(truth)


def cell_metrics(
    screener: Screener,
    case: FactorialCase,
    *,
    k: int = 5,
    intersect_valid: bool = False,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Evaluate one screener on one case."""

    if dtype_bytes <= 0:
        raise BaselineRunnerError("dtype_bytes must be positive")
    candidates = screener.candidates(case.hidden)
    if len(candidates) != int(case.hidden.shape[0]):
        raise BaselineRunnerError(
            f"screener {screener.name!r} returned {len(candidates)} rows for batch="
            f"{int(case.hidden.shape[0])}"
        )
    for candidate in candidates:
        if not isinstance(candidate, TokenSet):
            raise BaselineRunnerError("screener must return TokenSet rows")
    if intersect_valid and case.valid_sets is not None:
        candidates = [
            candidate if valid is None else candidate.intersection(valid)
            for candidate, valid in zip(candidates, case.valid_sets, strict=True)
        ]
    result = run_policy(
        screener,
        case.as_head_case(),
        k=k,
        intersect_valid=intersect_valid,
    )
    sizes = [len(candidate) for candidate in candidates]
    hidden_size = int(case.hidden.shape[1])
    return {
        "screener": screener.name,
        "case": case.name,
        "n": len(sizes),
        "metrics": dict(result.metrics),
        "coverage": _coverage(candidates, case.truth),
        "mean_candidates": sum(sizes) / len(sizes),
        "max_candidates": max(sizes),
        "candidate_bytes": sum(sizes) * hidden_size * dtype_bytes,
        "valid_intersected": bool(intersect_valid),
    }


def run_factorial(
    screeners: Iterable[Screener],
    cases: Iterable[FactorialCase],
    *,
    k: int = 5,
    intersect_valid: bool = False,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Run every screener on every case and return a serializable matrix."""

    screeners = list(screeners)
    cases = list(cases)
    if not screeners:
        raise BaselineRunnerError("no screeners supplied")
    if not cases:
        raise BaselineRunnerError("no cases supplied")
    names = [screener.name for screener in screeners]
    if len(set(names)) != len(names):
        raise BaselineRunnerError("screener names must be unique")
    case_names = [case.name for case in cases]
    if len(set(case_names)) != len(case_names):
        raise BaselineRunnerError("case names must be unique")

    cells = []
    for case in cases:
        for screener in screeners:
            cells.append(
                cell_metrics(
                    screener,
                    case,
                    k=k,
                    intersect_valid=intersect_valid,
                    dtype_bytes=dtype_bytes,
                )
            )
    return {
        "schema_version": FACTORIAL_SCHEMA,
        "k": k,
        "dtype_bytes": dtype_bytes,
        "screeners": names,
        "cases": case_names,
        "cells": cells,
    }


def write_factorial(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_cases(path: str | Path) -> list[FactorialCase]:
    """Load cases from a torch file produced by a data builder.

    The file may contain ``{"cases": [...]}`` or a list of case mappings. Each
    case needs ``name``, ``hidden`` (``[B, D]`` tensor), and ``truth`` (``[B, k]``
    or nested list). Optional keys are ``valid_sets``, ``domains``, and ``meta``.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BaselineRunnerError("torch is required to load factorial cases") from exc
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        raw_cases = payload.get("cases")
    else:
        raw_cases = payload
    if not isinstance(raw_cases, list):
        raise BaselineRunnerError("factorial input needs a 'cases' list")
    cases = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise BaselineRunnerError("each case must be an object")
        hidden = raw.get("hidden")
        truth = raw.get("truth")
        if hidden is None or truth is None:
            raise BaselineRunnerError("each case needs 'hidden' and 'truth'")
        if hasattr(truth, "tolist"):
            truth = truth.tolist()
        valid = raw.get("valid_sets")
        if valid is not None:
            valid = tuple(
                None if row is None else TokenSet(int(token) for token in row)
                for row in valid
            )
        cases.append(
            FactorialCase(
                name=str(raw.get("name", f"case-{len(cases)}")),
                hidden=hidden,
                truth=tuple(tuple(int(token) for token in row) for row in truth),
                valid_sets=valid,
                domains=None if raw.get("domains") is None else tuple(raw["domains"]),
                meta=dict(raw.get("meta", {})),
            )
        )
    if not cases:
        raise BaselineRunnerError("factorial input has no cases")
    return cases


__all__ = [
    "FACTORIAL_SCHEMA",
    "FactorialCase",
    "cell_metrics",
    "load_cases",
    "run_factorial",
    "write_factorial",
]
