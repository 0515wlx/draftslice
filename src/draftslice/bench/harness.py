"""B1 offline accuracy harness.

The harness consumes one record per query: the candidate set, the true top-k in
the measurement universe, and optionally the strict valid set. Model execution is
out of scope; a runner that has hidden states and a head produces these records.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import (
    MetricsError,
    all_k,
    order_accuracy,
    prefix_coverage,
    set_recall,
    top1,
)

BENCH_SCHEMA = "bench/1"


class HarnessError(ValueError):
    """Raised when benchmark input is malformed."""


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One query's candidate set and oracle top-k."""

    candidates: tuple[int, ...]
    truth: tuple[int, ...]
    valid: tuple[int, ...] | None = None
    domain: str = "default"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.truth:
            raise HarnessError("truth must contain at least the top-1 token")
        for name, values in (
            ("candidates", self.candidates),
            ("truth", self.truth),
            ("valid", self.valid),
        ):
            if values is None:
                continue
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise HarnessError(f"{name} must contain non-negative token ids")
        object.__setattr__(self, "candidates", tuple(sorted(set(self.candidates))))
        if self.valid is not None:
            object.__setattr__(self, "valid", tuple(sorted(set(self.valid))))

    @classmethod
    def from_dict(cls, value: Any) -> CandidateRecord:
        if not isinstance(value, dict):
            raise HarnessError("record must be an object")
        candidates = value.get("candidates")
        truth = value.get("truth")
        if not isinstance(candidates, list) or not isinstance(truth, list):
            raise HarnessError("record needs 'candidates' and 'truth' lists")
        valid = value.get("valid")
        if valid is not None and not isinstance(valid, list):
            raise HarnessError("'valid' must be a list or null")
        meta = value.get("meta", {})
        if not isinstance(meta, dict):
            raise HarnessError("'meta' must be an object")
        return cls(
            candidates=tuple(candidates),
            truth=tuple(truth),
            valid=None if valid is None else tuple(valid),
            domain=str(value.get("domain", "default")),
            meta=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "candidates": list(self.candidates),
            "truth": list(self.truth),
            "domain": self.domain,
            "meta": self.meta,
        }
        if self.valid is not None:
            value["valid"] = list(self.valid)
        return value


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    n: int
    metrics: dict[str, float]
    by_domain: dict[str, dict[str, float]]
    warnings: tuple[str, ...] = ()
    schema_version: str = BENCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "n": self.n,
            "metrics": self.metrics,
            "by_domain": self.by_domain,
            "warnings": list(self.warnings),
        }


def _valid_metrics(records: Sequence[CandidateRecord]) -> dict[str, float]:
    constrained = [record for record in records if record.valid is not None]
    if not constrained:
        return {}
    recalls = []
    precisions = []
    equal = 0
    for record in constrained:
        valid = set(record.valid or ())
        candidates = set(record.candidates)
        recalls.append(1.0 if not valid else len(candidates & valid) / len(valid))
        precisions.append(1.0 if not candidates else len(candidates & valid) / len(candidates))
        equal += int(record.valid == record.candidates)
    return {
        "valid_recall": statistics.fmean(recalls),
        "valid_precision": statistics.fmean(precisions),
        "plan_equals_valid": equal / len(constrained),
    }


def _compute(
    records: Sequence[CandidateRecord],
    *,
    k: int,
    weights: Sequence[float] | None,
) -> dict[str, float]:
    if not records:
        raise HarnessError("cannot evaluate an empty record set")
    candidates = [record.candidates for record in records]
    truth = [record.truth for record in records]
    metrics = {
        "top1": top1(candidates, truth),
        f"set_recall@{k}": set_recall(candidates, truth, k=k),
        f"prefix_coverage@{k}": prefix_coverage(candidates, truth, k),
        f"all@{k}": all_k(candidates, truth, k),
        "OA": order_accuracy(candidates, truth, weights=weights),
        "m_mean": statistics.fmean(len(record.candidates) for record in records),
    }
    metrics.update(_valid_metrics(records))
    return metrics


def evaluate(
    records: Iterable[CandidateRecord],
    *,
    name: str = "accuracy",
    k: int = 5,
    weights: Sequence[float] | None = None,
) -> BenchmarkResult:
    """Compute B1 metrics overall and per domain."""

    items = list(records)
    if not items:
        raise HarnessError("cannot evaluate an empty record set")
    if k <= 0:
        raise HarnessError("k must be positive")
    min_truth = min(len(record.truth) for record in items)
    if min_truth < k:
        raise HarnessError(f"truth rows have at least one row shorter than k={k}")

    metrics = _compute(items, k=k, weights=weights)
    domains: dict[str, list[CandidateRecord]] = {}
    for record in items:
        domains.setdefault(record.domain, []).append(record)
    by_domain = {
        domain: _compute(domain_records, k=k, weights=weights)
        for domain, domain_records in sorted(domains.items())
    }
    warnings = []
    if len(domains) == 1 and "default" in domains:
        warnings.append("no domain labels; per-domain metrics are not meaningful")
    return BenchmarkResult(
        name=name,
        n=len(items),
        metrics=metrics,
        by_domain=by_domain,
        warnings=tuple(warnings),
    )


def load_records(path: str | Path) -> list[CandidateRecord]:
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HarnessError(f"cannot read records: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(CandidateRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, HarnessError, MetricsError) as exc:
            raise HarnessError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise HarnessError(f"{path}: no records")
    return records


def write_records(path: str | Path, records: Iterable[CandidateRecord]) -> None:
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_result(path: str | Path, result: BenchmarkResult) -> None:
    Path(path).write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BENCH_SCHEMA",
    "BenchmarkResult",
    "CandidateRecord",
    "HarnessError",
    "evaluate",
    "load_records",
    "write_records",
    "write_result",
]
