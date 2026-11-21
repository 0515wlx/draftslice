"""Core plan types.

A plan is deliberately data-only: it can be serialized, cached, logged, and
passed across the compiler/runtime boundary. The target verifier is never
represented here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

KernelKind = Literal["auto", "slice_gemv", "slice_gemm", "full"]
LayoutKind = Literal["native", "leaf_contiguous", "packed"]
Universe = Literal["vocab", "valid"]


class GuaranteeKind(str, Enum):
    """What a plan promises about its candidate set."""

    FULL = "full"
    EXACT_VALID_SET = "exact_valid_set"
    CALIBRATED = "calibrated"


@dataclass(frozen=True, slots=True, init=False)
class TokenSet:
    """A sorted, deduplicated, immutable set of native token ids.

    ``TokenSet`` normalizes its input so callers do not have to sort or dedupe.
    Membership is O(log n) via a tuple scan for small sets; plans are normally
    consumed by kernels that materialize arrays anyway.
    """

    ids: tuple[int, ...]

    def __init__(self, ids: Iterable[int] = ()) -> None:
        values = tuple(ids)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"token ids must be int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"token ids must be non-negative, got {value}")
        object.__setattr__(self, "ids", tuple(sorted(set(values))))

    @classmethod
    def _from_sorted_unique(cls, ids: Iterable[int]) -> TokenSet:
        """Internal fast path: the caller guarantees sorted, unique, non-negative ids.

        Bitmask decoding produces ascending unique ids directly, so re-running the
        normalization in ``__init__`` is pure overhead on the per-step hot path.
        Never call this with unnormalized input.
        """

        instance = object.__new__(cls)
        object.__setattr__(instance, "ids", tuple(ids))
        return instance

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, token_id: object) -> bool:
        return token_id in self.ids

    def __bool__(self) -> bool:
        return bool(self.ids)

    def __repr__(self) -> str:
        return f"TokenSet(size={len(self.ids)})"

    @property
    def is_empty(self) -> bool:
        return not self.ids

    def to_list(self) -> list[int]:
        return list(self.ids)

    def union(self, other: TokenSet) -> TokenSet:
        return TokenSet(self.ids + other.ids)

    def intersection(self, other: TokenSet) -> TokenSet:
        other_ids = set(other.ids)
        return TokenSet(value for value in self.ids if value in other_ids)

    def difference(self, other: TokenSet) -> TokenSet:
        other_ids = set(other.ids)
        return TokenSet(value for value in self.ids if value not in other_ids)

    def issubset(self, other: TokenSet) -> bool:
        other_ids = set(other.ids)
        return all(value in other_ids for value in self.ids)

    def covers(self, other: TokenSet) -> bool:
        return other.issubset(self)


@dataclass(frozen=True, slots=True)
class Guarantee:
    """The coverage contract attached to a plan."""

    kind: GuaranteeKind
    universe: Universe
    delta: float | None = None
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind is GuaranteeKind.CALIBRATED:
            if self.delta is None or not (0.0 < self.delta < 1.0):
                raise ValueError("calibrated guarantees require 0 < delta < 1")
        elif self.delta is not None:
            raise ValueError("only calibrated guarantees may carry a delta")
        if self.kind is GuaranteeKind.FULL and self.universe != "vocab":
            raise ValueError("full guarantees use the 'vocab' universe")
        if self.kind is GuaranteeKind.EXACT_VALID_SET and self.universe != "valid":
            raise ValueError("exact-valid-set guarantees use the 'valid' universe")

    @classmethod
    def full(cls) -> Guarantee:
        return cls(kind=GuaranteeKind.FULL, universe="vocab")

    @classmethod
    def exact_valid_set(cls) -> Guarantee:
        return cls(kind=GuaranteeKind.EXACT_VALID_SET, universe="valid")

    @classmethod
    def calibrated(
        cls,
        delta: float,
        *,
        universe: Universe = "vocab",
        calibration_id: str | None = None,
    ) -> Guarantee:
        return cls(
            kind=GuaranteeKind.CALIBRATED,
            universe=universe,
            delta=delta,
            calibration_id=calibration_id,
        )


@dataclass(frozen=True, slots=True)
class Execution:
    """Which rows a kernel actually reads.

    ``kind == "full"`` means the executor reads the full head and ``rows`` must
    be ``None``. For slice executions, ``rows`` is the execution row set. It may
    be a superset of the logical candidates.
    """

    kind: Literal["full", "slice_gemv", "slice_gemm"]
    rows: TokenSet | None = None
    layout: LayoutKind = "native"

    def __post_init__(self) -> None:
        if self.kind == "full":
            if self.rows is not None:
                raise ValueError("full execution must not carry rows")
            return
        if self.rows is None:
            raise ValueError("slice execution requires rows")

    def covers(self, candidates: TokenSet) -> bool:
        if self.kind == "full":
            return True
        assert self.rows is not None
        return candidates.issubset(self.rows)


@dataclass(frozen=True, slots=True)
class PlanCost:
    """A hardware-independent estimate plus optional per-profile predictions."""

    static_bytes: int = 0
    dynamic_bytes: int = 0
    macs: int = 0
    alu_ops: int = 0
    launch_count: int = 0
    predicted_us: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("static_bytes", "dynamic_bytes", "macs", "alu_ops", "launch_count"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bytes(self) -> int:
        return self.static_bytes + self.dynamic_bytes

    @classmethod
    def zero(cls) -> PlanCost:
        return cls()

    @classmethod
    def dense(cls, rows: int, hidden_size: int, dtype_bytes: int = 2) -> PlanCost:
        return cls(
            dynamic_bytes=rows * hidden_size * dtype_bytes,
            macs=rows * hidden_size,
            launch_count=1,
        )


@dataclass(frozen=True, slots=True)
class HeadPlan:
    """An executable plan for one grammar state or one screened query."""

    candidates: TokenSet
    execution: Execution
    guarantee: Guarantee
    state_id: int | None = None
    cost: PlanCost = field(default_factory=PlanCost.zero)
    source: str = "manual"
    plan_id: str = ""

    def __post_init__(self) -> None:
        if self.state_id is not None and self.state_id < 0:
            raise ValueError("state_id must be non-negative or None")
        if not self.execution.covers(self.candidates):
            raise ValueError("execution rows do not cover the candidate set")

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.candidates.ids

    @property
    def kernel(self) -> str:
        return self.execution.kind

    @property
    def packed(self) -> bool:
        return self.execution.layout == "packed"

    @property
    def estimated_bytes(self) -> int:
        return self.cost.total_bytes

    @property
    def is_strict(self) -> bool:
        return self.guarantee.kind is GuaranteeKind.EXACT_VALID_SET

    def covers(self, token_ids: Iterable[int]) -> bool:
        return self.candidates.covers(TokenSet(token_ids))

    @classmethod
    def from_slice(
        cls,
        slice: Slice,
        *,
        hidden_size: int = 0,
        dtype_bytes: int = 2,
        source: str = "constraint",
    ) -> HeadPlan:
        """Build a strict plan from the legacy :class:`Slice` type."""

        candidates = TokenSet(slice.token_ids)
        return cls(
            candidates=candidates,
            execution=Execution(kind="slice_gemv", rows=candidates),
            guarantee=Guarantee.exact_valid_set(),
            state_id=slice.state_id,
            cost=PlanCost.dense(len(candidates), hidden_size, dtype_bytes),
            source=source,
        )


@dataclass(frozen=True, slots=True)
class NodePlan:
    """A plan attached to one node of a draft tree."""

    node_id: int
    plan: HeadPlan


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Plans for all nodes in one decode step."""

    nodes: tuple[NodePlan, ...]
    union_rows: TokenSet
    kernel: KernelKind = "auto"
    cost: PlanCost = field(default_factory=PlanCost.zero)
    plan_id: str = ""

    def __post_init__(self) -> None:
        if self.kernel == "full":
            return
        for node in self.nodes:
            if not node.plan.candidates.issubset(self.union_rows):
                raise ValueError("step union does not cover a node's candidates")
        if self.kernel == "auto" and not self.nodes:
            raise ValueError("an empty step plan needs an explicit kernel")

    @property
    def size(self) -> int:
        return len(self.nodes)

    def covers(self, token_ids: Iterable[int]) -> bool:
        if self.kernel == "full":
            return True
        return self.union_rows.covers(TokenSet(token_ids))


@dataclass(frozen=True, slots=True)
class Slice:
    """Legacy exact valid-token set.

    Kept as a small, stable convenience type for callers that only need the
    state id and ids. New code should use :class:`HeadPlan`.
    """

    state_id: int
    token_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.token_ids)


__all__ = [
    "Execution",
    "Guarantee",
    "GuaranteeKind",
    "HeadPlan",
    "KernelKind",
    "LayoutKind",
    "NodePlan",
    "PlanCost",
    "Slice",
    "StepPlan",
    "TokenSet",
    "Universe",
]
