"""Core plan types.

A plan is deliberately data-only: it can be serialized, cached, logged, and
passed across the compiler/runtime boundary. The target verifier is never
represented here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

KernelKind = Literal["auto", "slice_gemv", "slice_gemm", "full"]
LayoutKind = Literal["native", "leaf_contiguous", "packed"]
Universe = Literal["vocab", "valid"]
ContractMode = Literal["strict", "mask", "mask_allpos", "full"]
GuaranteeScope = Literal["mu_proj", "mu_star"]
FallbackRule = Literal["full_head", "built_in_method", "none"]


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
    """The coverage contract attached to a plan.

    ``scope`` is the distribution the guarantee is relative to. DraftSlice only
    ever claims ``mu_proj`` (the engine's locally-projected constrained
    distribution); ``mu_star`` (a grammar-faithful distribution) is deliberately
    not claimed. ``fallback_rule`` is the fail-closed action when a promise
    cannot be kept for a row.
    """

    kind: GuaranteeKind
    universe: Universe
    delta: float | None = None
    calibration_id: str | None = None
    scope: GuaranteeScope = "mu_proj"
    fallback_rule: FallbackRule = "full_head"

    def __post_init__(self) -> None:
        if self.scope not in ("mu_proj", "mu_star"):
            raise ValueError("guarantee scope must be 'mu_proj' or 'mu_star'")
        if self.fallback_rule not in ("full_head", "built_in_method", "none"):
            raise ValueError("invalid fallback rule")
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
class ServingPoint:
    """The server-side operating point chosen for a contract.

    Records *why* a step ran the way it did, so benchmark and operations can
    explain a ``k`` / ``mode`` choice instead of inferring it from output ids.
    """

    k: int = 0
    source: str = "builtin"
    expected_tau: float | None = None
    target_metric: str = "throughput"
    # Fraction of the vocabulary rows a slice reads. This is *not* the measured
    # draft-head share of step time; the two were conflated in the first
    # experiments. ``measured_head_share`` is filled from B4 timing.
    row_share: float | None = None
    measured_head_share: float | None = None

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("k must be non-negative")
        if self.expected_tau is not None and self.expected_tau < 0:
            raise ValueError("expected_tau must be non-negative")
        for name in ("row_share", "measured_head_share"):
            value = getattr(self, name)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def head_share(self) -> float | None:
        """Deprecated alias for the row share; use ``row_share`` in new code."""

        return self.row_share

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "source": self.source,
            "expected_tau": self.expected_tau,
            "target_metric": self.target_metric,
            "row_share": self.row_share,
            "measured_head_share": self.measured_head_share,
        }


@dataclass(frozen=True, slots=True)
class DraftContract:
    """Immutable end-to-end contract for one draft-head decision.

    This is the concrete form of the production design's ``DraftContract``: it
    follows a grammar state through candidate selection, execution, guarantee,
    and the chosen serving point. It is data-only so it can be logged, serialized
    and compared across vLLM/SGLang. The target verifier is never represented.
    """

    grammar_state_id: str | None
    valid_set: TokenSet | None
    candidate_set: TokenSet | None
    mode: ContractMode
    execution: Execution
    guarantee: Guarantee
    serving: ServingPoint
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("strict", "mask", "mask_allpos", "full"):
            raise ValueError(f"invalid contract mode: {self.mode!r}")
        if self.guarantee.scope != "mu_proj":
            raise ValueError("DraftSlice contracts must use the mu_proj scope")
        if self.guarantee.fallback_rule == "none" and self.fallback_reason is not None:
            raise ValueError("a contract with a fallback reason needs a fallback rule")
        if self.mode == "strict":
            if self.guarantee.kind is not GuaranteeKind.EXACT_VALID_SET:
                raise ValueError("strict contracts require an exact-valid-set guarantee")
            if self.candidate_set is None or self.valid_set is None:
                raise ValueError("strict contracts carry both V and A")
            if not self.candidate_set.issubset(self.valid_set):
                raise ValueError("strict contracts require A subset of V")
        elif self.mode in ("mask", "mask_allpos"):
            if self.candidate_set is not None and self.valid_set is not None:
                if not self.candidate_set.issubset(self.valid_set):
                    raise ValueError("mask contracts require A subset of V")
        else:  # full
            if self.execution.kind != "full":
                raise ValueError("full contracts use a full-head execution")
            if self.candidate_set is not None:
                raise ValueError("full contracts do not carry a candidate set")

    @property
    def is_sliced(self) -> bool:
        return self.execution.kind != "full"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view for stats and release manifests."""

        def tokens(value: TokenSet | None) -> dict[str, Any] | None:
            return None if value is None else {"count": len(value), "ids": list(value.ids)}

        return {
            "grammar_state_id": self.grammar_state_id,
            "valid_set": tokens(self.valid_set),
            "candidate_set": tokens(self.candidate_set),
            "mode": self.mode,
            "execution": {
                "kind": self.execution.kind,
                "rows": None
                if self.execution.rows is None
                else len(self.execution.rows),
                "layout": self.execution.layout,
            },
            "guarantee": {
                "kind": self.guarantee.kind.value,
                "universe": self.guarantee.universe,
                "delta": self.guarantee.delta,
                "calibration_id": self.guarantee.calibration_id,
                "scope": self.guarantee.scope,
                "fallback_rule": self.guarantee.fallback_rule,
            },
            "serving": self.serving.to_dict(),
            "fallback_reason": self.fallback_reason,
        }


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
    "ContractMode",
    "DraftContract",
    "Execution",
    "FallbackRule",
    "Guarantee",
    "GuaranteeKind",
    "GuaranteeScope",
    "HeadPlan",
    "KernelKind",
    "LayoutKind",
    "NodePlan",
    "PlanCost",
    "ServingPoint",
    "Slice",
    "StepPlan",
    "TokenSet",
    "Universe",
]
