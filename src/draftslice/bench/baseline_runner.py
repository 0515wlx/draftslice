"""Run a fair candidate-policy matrix over one hidden-state/oracle case."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..plan.types import TokenSet
from .baselines import CandidatePolicy
from .harness import BenchmarkResult, CandidateRecord, evaluate

BASELINE_SCHEMA = "baseline_matrix/1"


class BaselineRunnerError(ValueError):
    """Raised when a baseline case or policy is malformed."""


@dataclass(frozen=True, slots=True)
class HeadQualityCase:
    """Hidden states, oracle top-k, and optional strict valid sets."""

    hidden: Any
    truth: tuple[tuple[int, ...], ...]
    domains: tuple[str, ...] | None = None
    valid_sets: tuple[TokenSet | None, ...] | None = None
    name: str = "case"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        batch = int(self.hidden.shape[0])
        if not self.truth:
            raise BaselineRunnerError("truth must not be empty")
        if len(self.truth) != batch:
            raise BaselineRunnerError(f"truth has {len(self.truth)} rows for batch={batch}")
        object.__setattr__(self, "truth", tuple(tuple(row) for row in self.truth))
        if self.domains is not None and len(self.domains) != batch:
            raise BaselineRunnerError("domains length does not match the batch")
        if self.valid_sets is not None and len(self.valid_sets) != batch:
            raise BaselineRunnerError("valid_sets length does not match the batch")


def _intersect(candidates: TokenSet, valid: TokenSet | None) -> TokenSet:
    return candidates if valid is None else candidates.intersection(valid)


def run_policy(
    policy: CandidatePolicy,
    case: HeadQualityCase,
    *,
    k: int = 5,
    intersect_valid: bool = False,
) -> BenchmarkResult:
    """Evaluate one policy with the shared B1 metrics."""

    candidates = policy.candidates(case.hidden)
    if len(candidates) != len(case.truth):
        raise BaselineRunnerError(
            f"policy {policy.name!r} returned {len(candidates)} rows, expected {len(case.truth)}"
        )
    records = []
    for index, (candidate, truth) in enumerate(zip(candidates, case.truth, strict=True)):
        valid = None if case.valid_sets is None else case.valid_sets[index]
        if intersect_valid:
            candidate = _intersect(candidate, valid)
        records.append(
            CandidateRecord(
                candidates=tuple(candidate.ids),
                truth=tuple(truth),
                valid=None if valid is None else tuple(valid.ids),
                domain="default" if case.domains is None else case.domains[index],
                meta={"policy": policy.name},
            )
        )
    return evaluate(records, name=policy.name, k=k)


def run_matrix(
    policies: Iterable[CandidatePolicy],
    case: HeadQualityCase,
    *,
    k: int = 5,
    intersect_valid: bool = False,
) -> dict[str, BenchmarkResult]:
    results = {}
    for policy in policies:
        if policy.name in results:
            raise BaselineRunnerError(f"duplicate policy name: {policy.name}")
        results[policy.name] = run_policy(
            policy,
            case,
            k=k,
            intersect_valid=intersect_valid,
        )
    if not results:
        raise BaselineRunnerError("no policies supplied")
    return results


def write_matrix(
    path: str | Path,
    results: dict[str, BenchmarkResult],
    *,
    case_name: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    value = {
        "schema_version": BASELINE_SCHEMA,
        "case": case_name,
        "metadata": metadata or {},
        "policies": {name: result.to_dict() for name, result in sorted(results.items())},
    }
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BASELINE_SCHEMA",
    "BaselineRunnerError",
    "HeadQualityCase",
    "run_matrix",
    "run_policy",
    "write_matrix",
]
