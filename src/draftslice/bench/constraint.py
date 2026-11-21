"""B3 constraint-state replay.

The replay runner consumes recorded grammar states, compiles a strict plan for
each state, optionally checks the plan cache, and verifies that every recorded
valid token is covered. It does not reimplement a grammar backend: real traces
come from xgrammar, an engine adapter, or a recorded vLLM structured-output run.

Input formats:

* JSON object with ``{"states": [...]}``;
* JSON array of state objects;
* JSON Lines, one state object per line.

A state object has ``state_id`` (optional, assigned by position when absent) and
``valid_ids``. Optional ``next_token`` records the transition that produced the
next state.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..cache.plan_cache import LRUPlanCache
from ..constraints.base import ConstraintState
from ..plan.compiler import Budget, PlanCompiler, PlanPolicy
from ..plan.types import TokenSet

CONSTRAINT_SCHEMA = "constraint_bench/1"


class ConstraintBenchError(ValueError):
    """Raised when a replay file or state is malformed."""


@dataclass(frozen=True, slots=True)
class ConstraintRecord:
    state_id: int
    valid_ids: tuple[int, ...]
    next_token: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state_id < 0:
            raise ValueError("state_id must be non-negative")
        if any(token < 0 for token in self.valid_ids):
            raise ValueError("valid ids must be non-negative")

    @property
    def valid_set(self) -> TokenSet:
        return TokenSet(self.valid_ids)


def _record_from_dict(item: dict[str, Any], index: int) -> ConstraintRecord:
    if "valid_ids" not in item:
        raise ConstraintBenchError("state is missing valid_ids")
    raw_ids = item["valid_ids"]
    if not isinstance(raw_ids, (list, tuple)):
        raise ConstraintBenchError("valid_ids must be a list")
    try:
        state_id = int(item.get("state_id", index))
    except (TypeError, ValueError) as exc:
        raise ConstraintBenchError("state_id must be an integer") from exc
    next_token = item.get("next_token")
    if next_token is not None:
        next_token = int(next_token)
    meta = {
        key: value
        for key, value in item.items()
        if key not in ("state_id", "valid_ids", "next_token")
    }
    return ConstraintRecord(
        state_id=state_id,
        valid_ids=tuple(int(token) for token in raw_ids),
        next_token=next_token,
        meta=meta,
    )


def load_constraint_records(path: str | Path) -> list[ConstraintRecord]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConstraintBenchError("replay file is empty")
    if text[0] in "[{":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if value is not None:
            items = value.get("states") if isinstance(value, dict) else value
            if not isinstance(items, list):
                raise ConstraintBenchError("JSON replay must be an array or contain 'states'")
            return [_record_from_dict(item, index) for index, item in enumerate(items)]
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConstraintBenchError(f"line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ConstraintBenchError(f"line {line_number}: state must be an object")
        records.append(_record_from_dict(item, line_number - 1))
    if not records:
        raise ConstraintBenchError("replay file has no states")
    return records


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def run_constraint_replay(
    records: list[ConstraintRecord],
    *,
    vocab_size: int,
    hidden_size: int,
    max_candidates: int | None = None,
    max_bytes: int | None = None,
    use_cache: bool = True,
    cache_bytes: int = 64 * 1024 * 1024,
    policy: PlanPolicy | None = None,
) -> dict[str, Any]:
    """Compile plans for recorded states and verify strict coverage."""

    if not records:
        raise ConstraintBenchError("no records to replay")
    if vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("vocab_size and hidden_size must be positive")

    cache = LRUPlanCache(cache_bytes) if use_cache else None
    compiler = PlanCompiler(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        plan_cache=cache,
    )
    budget = Budget(max_candidates=max_candidates, max_bytes=max_bytes)
    policy = policy or PlanPolicy(mode="strict")

    compile_us: list[float] = []
    fallback = 0
    violations = 0
    valid_sizes: list[int] = []
    executed_sizes: list[int] = []
    details: list[dict[str, Any]] = []

    for record in records:
        state = ConstraintState(state_id=record.state_id, valid_ids=record.valid_set)
        start = time.perf_counter()
        plan = compiler.compile_strict(state, budget=budget, policy=policy)
        elapsed_us = (time.perf_counter() - start) * 1e6
        compile_us.append(elapsed_us)

        covered = plan.candidates.covers(record.valid_set)
        execution_covered = plan.execution.covers(record.valid_set)
        if not covered or not execution_covered:
            violations += 1
        if plan.source == "fallback" or plan.execution.kind == "full":
            fallback += 1
        valid_sizes.append(len(record.valid_set))
        executed_sizes.append(
            vocab_size if plan.execution.kind == "full" else len(plan.execution.rows or ())
        )
        details.append(
            {
                "state_id": record.state_id,
                "valid": len(record.valid_set),
                "candidates": plan.size,
                "executed": executed_sizes[-1],
                "execution": plan.execution.kind,
                "source": plan.source,
                "compile_us": elapsed_us,
                "covered": covered and execution_covered,
                "plan_id": plan.plan_id,
            }
        )

    stats = None if cache is None else cache.stats
    if stats is not None:
        cache_summary = asdict(stats)
        cache_summary["requests"] = stats.requests
        cache_summary["hit_rate"] = stats.hit_rate
    else:
        cache_summary = None
    return {
        "schema_version": CONSTRAINT_SCHEMA,
        "config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "max_candidates": max_candidates,
            "max_bytes": max_bytes,
            "use_cache": use_cache,
            "policy": {
                "mode": policy.mode,
                "layout": policy.layout,
                "prefer_kernel": policy.prefer_kernel,
            },
        },
        "summary": {
            "states": len(records),
            "unique_states": len({record.state_id for record in records}),
            "violations": violations,
            "fallback_rate": fallback / len(records),
            "valid_mean": statistics.fmean(valid_sizes),
            "valid_p50": _percentile([float(value) for value in valid_sizes], 0.5),
            "valid_p90": _percentile([float(value) for value in valid_sizes], 0.9),
            "valid_max": max(valid_sizes),
            "executed_mean": statistics.fmean(executed_sizes),
            "candidate_ratio": statistics.fmean(
                executed / vocab_size for executed in executed_sizes
            ),
            "compile_p50_us": _percentile(compile_us, 0.5),
            "compile_p90_us": _percentile(compile_us, 0.9),
            "compile_max_us": max(compile_us),
            "cache": cache_summary,
        },
        "states": details,
    }


def write_constraint_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CONSTRAINT_SCHEMA",
    "ConstraintBenchError",
    "ConstraintRecord",
    "load_constraint_records",
    "run_constraint_replay",
    "write_constraint_result",
]
