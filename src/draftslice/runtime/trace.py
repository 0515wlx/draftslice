"""Per-step production trace (``trace/1``).

A :class:`StepTrace` is the observability record behind one draft-head decision.
It is data-only and JSON-safe so the same record can go to the in-process
registry, a sampled JSONL sink, or a benchmark artifact. The layout follows
``docs/production-architecture.md`` §7:

* identity  -- engine/model/tp/quant/graph/workload/config hash
* decision  -- profile mode, cell, action, reason, row/head share
* timing    -- plan compile, kernel, and step time
* cache     -- plan and candidate-set cache hits
* outcome   -- accepted tokens, tau, validity, whether a fallback ran

``row_share`` and ``measured_head_share`` are deliberately separate fields: the
row share is the fraction of the vocabulary the slice reads, the head share is
the measured draft-head fraction of step time. Conflating them overstated the
benefit in the early experiments.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .reasons import fallback_reason, reason_code

SCHEMA = "trace/1"

#: Contract modes and the matching execution actions.
MODES = frozenset({"builtin", "full", "strict", "mask", "mask_allpos"})
ACTIONS = frozenset({"builtin", "full", "slice", "mask", "mask_allpos"})
PROFILE_MODES = frozenset({"off", "observe", "enforce"})
GRAPH_MODES = frozenset({"eager", "full_cuda_graph", "step0"})


class TraceError(ValueError):
    """Raised when a trace record is malformed."""


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    """What produced the step; part of every profile/gate comparison."""

    engine: str
    model: str
    tp: int = 1
    quant: str = "bf16"
    graph: str = "eager"
    workload: str = "unknown"
    config_hash: str = ""

    def __post_init__(self) -> None:
        if self.engine not in ("vllm", "sglang", "reference"):
            raise TraceError(f"invalid engine: {self.engine!r}")
        if not self.model:
            raise TraceError("model must not be empty")
        if self.tp < 1:
            raise TraceError("tp must be >= 1")
        if self.graph not in GRAPH_MODES:
            raise TraceError(f"invalid graph mode: {self.graph!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "tp": self.tp,
            "quant": self.quant,
            "graph": self.graph,
            "workload": self.workload,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class TraceDecision:
    """The policy decision and its coverage accounting."""

    mode: str = "builtin"
    action: str = "builtin"
    profile_mode: str = "off"
    source: str = "builtin"
    k: int = 0
    cell_id: str | None = None
    cell_hit: bool = False
    reason: str | None = None
    candidates: int = 0
    rows: int = 0
    row_share: float = 0.0
    measured_head_share: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise TraceError(f"invalid mode: {self.mode!r}")
        if self.action not in ACTIONS:
            raise TraceError(f"invalid action: {self.action!r}")
        if self.profile_mode not in PROFILE_MODES:
            raise TraceError(f"invalid profile_mode: {self.profile_mode!r}")
        if self.k < 0:
            raise TraceError("k must be >= 0")
        if self.candidates < 0 or self.rows < 0:
            raise TraceError("candidates and rows must be >= 0")
        if not (0.0 <= self.row_share <= 1.0):
            raise TraceError("row_share must be in [0, 1]")
        if self.measured_head_share is not None and not (
            0.0 <= self.measured_head_share <= 1.0
        ):
            raise TraceError("measured_head_share must be in [0, 1]")
        if self.reason is not None:
            # Normalize legacy strings so dashboards can group by code.
            object.__setattr__(self, "reason", reason_code(self.reason))

    @property
    def reason_enum(self) -> Any:
        return fallback_reason(self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "action": self.action,
            "profile_mode": self.profile_mode,
            "source": self.source,
            "k": self.k,
            "cell_id": self.cell_id,
            "cell_hit": self.cell_hit,
            "reason": self.reason,
            "candidates": self.candidates,
            "rows": self.rows,
            "row_share": self.row_share,
            "measured_head_share": self.measured_head_share,
        }


@dataclass(frozen=True, slots=True)
class TraceTiming:
    """Microseconds; ``None`` means not measured (never zero-filled)."""

    plan_compile_us: float | None = None
    kernel_us: float | None = None
    step_us: float | None = None

    def __post_init__(self) -> None:
        for name in ("plan_compile_us", "kernel_us", "step_us"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise TraceError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_compile_us": self.plan_compile_us,
            "kernel_us": self.kernel_us,
            "step_us": self.step_us,
        }


@dataclass(frozen=True, slots=True)
class TraceCache:
    plan_hit: bool = False
    candidate_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"plan_hit": self.plan_hit, "candidate_hit": self.candidate_hit}


@dataclass(frozen=True, slots=True)
class TraceOutcome:
    """Verification/acceptance result; unknown until the step completes."""

    accepted: int | None = None
    tau: float | None = None
    validity: bool | None = None
    fallback: bool = False

    def __post_init__(self) -> None:
        if self.accepted is not None and self.accepted < 0:
            raise TraceError("accepted must be >= 0")
        if self.tau is not None and self.tau < 0:
            raise TraceError("tau must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "tau": self.tau,
            "validity": self.validity,
            "fallback": self.fallback,
        }


@dataclass(frozen=True, slots=True)
class StepTrace:
    """One decision record; ``to_jsonl`` is the stable sink format."""

    identity: TraceIdentity
    decision: TraceDecision
    timing: TraceTiming = field(default_factory=TraceTiming)
    cache: TraceCache = field(default_factory=TraceCache)
    outcome: TraceOutcome = field(default_factory=TraceOutcome)
    step: int = 0
    contract: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise TraceError("step must be >= 0")

    @property
    def schema(self) -> str:
        return SCHEMA

    @property
    def is_fallback(self) -> bool:
        return self.decision.action in ("builtin", "full")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "step": self.step,
            "identity": self.identity.to_dict(),
            "decision": self.decision.to_dict(),
            "timing": self.timing.to_dict(),
            "cache": self.cache.to_dict(),
            "outcome": self.outcome.to_dict(),
            "contract": self.contract,
        }

    def to_jsonl(self) -> str:
        """One compact JSON line (stable key order)."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepTrace:
        if data.get("schema") != SCHEMA:
            raise TraceError(
                f"unsupported trace schema {data.get('schema')!r}; expected {SCHEMA!r}"
            )
        try:
            identity = TraceIdentity(**data["identity"])
            decision = TraceDecision(**data["decision"])
            timing = TraceTiming(**data.get("timing", {}))
            cache = TraceCache(**data.get("cache", {}))
            outcome = TraceOutcome(**data.get("outcome", {}))
        except KeyError as exc:
            raise TraceError(f"missing trace field: {exc}") from exc
        except TypeError as exc:
            raise TraceError(f"bad trace field: {exc}") from exc
        return cls(
            identity=identity,
            decision=decision,
            timing=timing,
            cache=cache,
            outcome=outcome,
            step=int(data.get("step", 0)),
            contract=data.get("contract"),
        )

    @classmethod
    def from_jsonl(cls, line: str) -> StepTrace:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceError(f"invalid trace JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise TraceError("trace JSON must be an object")
        return cls.from_dict(data)


class TraceRegistry:
    """Bounded in-process ring of recent traces (always-on observability)."""

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise TraceError("capacity must be positive")
        self.capacity = int(capacity)
        self._records: deque[StepTrace] = deque(maxlen=self.capacity)
        self.recorded = 0

    def add(self, trace: StepTrace) -> None:
        self._records.append(trace)
        self.recorded += 1

    def recent(self, count: int | None = None) -> list[StepTrace]:
        if count is None or count >= len(self._records):
            return list(self._records)
        return list(self._records)[-int(count) :]

    def clear(self) -> None:
        self._records.clear()
        self.recorded = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "recorded": self.recorded,
            "recent": [trace.to_dict() for trace in self._records],
        }

    def jsonl(self) -> Iterable[str]:
        for trace in self._records:
            yield trace.to_jsonl()

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "ACTIONS",
    "GRAPH_MODES",
    "MODES",
    "PROFILE_MODES",
    "SCHEMA",
    "StepTrace",
    "TraceCache",
    "TraceDecision",
    "TraceError",
    "TraceIdentity",
    "TraceOutcome",
    "TraceRegistry",
    "TraceTiming",
]
