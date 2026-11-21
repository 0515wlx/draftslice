"""Strict plan compilation.

M1 implements the strict path only: grammar/schema -> valid token set -> head
plan. Calibrated screening is a later milestone and is deliberately rejected here
instead of silently dropping valid tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..constraints.base import ConstraintSession, ConstraintState
from .serialize import compute_plan_id, compute_step_plan_id
from .types import (
    Execution,
    Guarantee,
    HeadPlan,
    LayoutKind,
    NodePlan,
    PlanCost,
    StepPlan,
    TokenSet,
    Universe,
)


class PlanBudgetError(RuntimeError):
    """Raised when a strict plan cannot meet the configured budget."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Resource limits for plan compilation."""

    max_candidates: int | None = None
    max_bytes: int | None = None
    deadline_ms: float | None = None
    allow_full_fallback: bool = True

    def __post_init__(self) -> None:
        if self.max_candidates is not None and self.max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")


@dataclass(frozen=True, slots=True)
class PlanPolicy:
    """Plan-mode and layout preferences."""

    mode: Literal["strict", "full", "calibrated"] = "strict"
    layout: LayoutKind = "native"
    prefer_kernel: Literal["auto", "slice_gemv", "slice_gemm"] = "auto"

    def __post_init__(self) -> None:
        if self.mode not in ("strict", "full", "calibrated"):
            raise ValueError(
                "supported modes are strict, full, and calibrated; calibrated "
                "compilation requires an explicit candidate set and delta"
            )


def _as_state(state: ConstraintState | ConstraintSession) -> ConstraintState:
    if isinstance(state, ConstraintState):
        return state
    return state.current()


class PlanCompiler:
    """Compile constraint states into strict or full head plans."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        dtype_bytes: int = 2,
        plan_cache: Any | None = None,
    ) -> None:
        if vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        if hidden_size < 0:
            raise ValueError("hidden_size must be non-negative")
        if dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype_bytes = dtype_bytes
        self.plan_cache = plan_cache

    def _dense_cost(self, rows: int) -> PlanCost:
        return PlanCost.dense(rows, self.hidden_size, self.dtype_bytes)

    def _full_cost(self) -> PlanCost:
        return self._dense_cost(self.vocab_size)

    def compile_full(self, *, state_id: int | None = None) -> HeadPlan:
        if self.vocab_size <= 0:
            raise ValueError("full plans require a positive vocab_size")
        plan = HeadPlan(
            candidates=TokenSet(range(self.vocab_size)),
            execution=Execution(kind="full"),
            guarantee=Guarantee.full(),
            state_id=state_id,
            cost=self._full_cost(),
            source="full",
        )
        return HeadPlan(
            candidates=plan.candidates,
            execution=plan.execution,
            guarantee=plan.guarantee,
            state_id=plan.state_id,
            cost=plan.cost,
            source=plan.source,
            plan_id=compute_plan_id(plan),
        )

    def compile_strict(
        self,
        state: ConstraintState | ConstraintSession,
        *,
        budget: Budget | None = None,
        policy: PlanPolicy | None = None,
    ) -> HeadPlan:
        budget = budget or Budget()
        policy = policy or PlanPolicy()
        current = _as_state(state)
        cache_key = self._cache_key(current.state_id, budget, policy)
        if cache_key is not None:
            cached = self.plan_cache.get(cache_key)
            if cached is not None:
                return cached

        if policy.mode == "full":
            plan = self.compile_full(state_id=current.state_id)
            if cache_key is not None:
                self.plan_cache.put_keyed(cache_key, plan)
            return plan
        if policy.mode != "strict":
            raise ValueError(f"unsupported plan mode: {policy.mode}")

        candidates = current.valid_ids
        if self.vocab_size > 0:
            for token_id in candidates:
                if token_id >= self.vocab_size:
                    raise ValueError(
                        f"valid token {token_id} is outside vocab_size={self.vocab_size}"
                    )
        slice_cost = self._dense_cost(len(candidates))
        over_budget = (
            budget.max_candidates is not None and len(candidates) > budget.max_candidates
        ) or (budget.max_bytes is not None and slice_cost.dynamic_bytes > budget.max_bytes)

        if over_budget:
            if not budget.allow_full_fallback:
                raise PlanBudgetError(
                    f"strict plan needs {len(candidates)} rows / "
                    f"{slice_cost.dynamic_bytes} bytes, over budget"
                )
            plan = self._with_id(
                HeadPlan(
                    candidates=candidates,
                    execution=Execution(kind="full"),
                    guarantee=Guarantee.exact_valid_set(),
                    state_id=current.state_id,
                    cost=self._full_cost(),
                    source="fallback",
                )
            )
            if cache_key is not None:
                self.plan_cache.put_keyed(cache_key, plan)
            return plan

        kernel = "slice_gemm" if policy.prefer_kernel == "slice_gemm" else "slice_gemv"
        plan = HeadPlan(
            candidates=candidates,
            execution=Execution(
                kind=kernel,
                rows=candidates,
                layout=policy.layout,
            ),
            guarantee=Guarantee.exact_valid_set(),
            state_id=current.state_id,
            cost=slice_cost,
            source="constraint",
        )
        plan = self._with_id(plan)
        if cache_key is not None:
            self.plan_cache.put_keyed(cache_key, plan)
        return plan

    def _cache_key(
        self,
        state_id: int | None,
        budget: Budget,
        policy: PlanPolicy,
    ) -> tuple[Any, ...] | None:
        if self.plan_cache is None or state_id is None:
            return None
        return (
            state_id,
            budget.max_candidates,
            budget.max_bytes,
            budget.deadline_ms,
            budget.allow_full_fallback,
            policy.mode,
            policy.layout,
            policy.prefer_kernel,
            self.vocab_size,
            self.hidden_size,
            self.dtype_bytes,
        )

    def compile_calibrated(
        self,
        candidates: TokenSet,
        *,
        universe: TokenSet | None = None,
        delta: float,
        state_id: int | None = None,
        budget: Budget | None = None,
        policy: PlanPolicy | None = None,
        source: str = "router",
        calibration_id: str | None = None,
    ) -> HeadPlan:
        """Compile a screened candidate set with a calibrated guarantee."""

        budget = budget or Budget()
        policy = policy or PlanPolicy(mode="calibrated")
        if not 0.0 < delta < 1.0:
            raise ValueError("calibrated plans require 0 < delta < 1")
        if universe is not None and not candidates.issubset(universe):
            raise ValueError("calibrated candidates are not a subset of the universe")
        if self.vocab_size > 0 and any(token >= self.vocab_size for token in candidates):
            raise ValueError("candidate token is outside vocab_size")

        slice_cost = self._dense_cost(len(candidates))
        over_budget = (
            budget.max_candidates is not None and len(candidates) > budget.max_candidates
        ) or (budget.max_bytes is not None and slice_cost.dynamic_bytes > budget.max_bytes)
        if over_budget:
            if not budget.allow_full_fallback:
                raise PlanBudgetError(
                    f"calibrated plan needs {len(candidates)} rows / "
                    f"{slice_cost.dynamic_bytes} bytes, over budget"
                )
            full = self.compile_full(state_id=state_id)
            plan = HeadPlan(
                candidates=candidates,
                execution=full.execution,
                guarantee=Guarantee.calibrated(
                    delta,
                    universe="valid" if universe is not None else "vocab",
                    calibration_id=calibration_id,
                ),
                state_id=state_id,
                cost=self._full_cost(),
                source="fallback",
            )
            return self._with_id(plan)

        kernel = "slice_gemm" if policy.prefer_kernel == "slice_gemm" else "slice_gemv"
        plan = HeadPlan(
            candidates=candidates,
            execution=Execution(kind=kernel, rows=candidates, layout=policy.layout),
            guarantee=Guarantee.calibrated(
                delta,
                universe="valid" if universe is not None else "vocab",
                calibration_id=calibration_id,
            ),
            state_id=state_id,
            cost=slice_cost,
            source=source,
        )
        return self._with_id(plan)

    def compile_step(
        self,
        states: Sequence[ConstraintState | ConstraintSession],
        *,
        budget: Budget | None = None,
        policy: PlanPolicy | None = None,
        mode: Literal["strict", "full", "calibrated"] | None = None,
        screener: Any | None = None,
        hidden_states: Any | None = None,
        delta: float | None = None,
        universe: Universe = "valid",
        source: str = "screener",
        calibration_id: str | None = None,
    ) -> StepPlan:
        """Compile plans for all nodes of a draft tree.

        Without ``screener`` this is the strict/full path. With a screener the
        call is an explicit calibrated compilation: ``mode`` must resolve to
        ``calibrated``, ``delta`` is required, and candidates are restricted to
        the per-node valid set when ``universe="valid"``. Strict mode never
        accepts a screener, so valid tokens cannot be dropped silently.
        """

        policy = policy or PlanPolicy()
        if mode is not None and mode != policy.mode:
            policy = PlanPolicy(
                mode=mode,
                layout=policy.layout,
                prefer_kernel=policy.prefer_kernel,
            )

        if screener is None:
            budget = budget or Budget()
            nodes = tuple(
                NodePlan(
                    node_id=index,
                    plan=self.compile_strict(state, budget=budget, policy=policy),
                )
                for index, state in enumerate(states)
            )
            return self._step_from_nodes(nodes)

        if hidden_states is None:
            raise ValueError("a screener requires hidden_states")
        if delta is None:
            raise ValueError("a screener requires an explicit delta")
        if policy.mode != "calibrated":
            raise ValueError(
                "a screener requires calibrated mode; strict/full plans must not "
                "silently drop valid tokens"
            )
        candidates = screener.candidates(hidden_states)
        if len(candidates) != len(states):
            raise ValueError(
                f"screener returned {len(candidates)} rows for {len(states)} states"
            )
        return self.compile_screened(
            states,
            candidates,
            delta=delta,
            universe=universe,
            budget=budget,
            policy=policy,
            source=source,
            calibration_id=calibration_id,
        )

    def compile_screened(
        self,
        states: Sequence[ConstraintState | ConstraintSession],
        candidates: Sequence[TokenSet],
        *,
        delta: float,
        universe: Universe = "valid",
        budget: Budget | None = None,
        policy: PlanPolicy | None = None,
        source: str = "screener",
        calibration_id: str | None = None,
    ) -> StepPlan:
        """Compile calibrated plans from per-node screened candidates.

        ``universe="valid"`` enforces ``A := A ∩ V`` and records ``V`` as the
        coverage universe. ``universe="vocab"`` keeps ``A`` unchanged for
        free-form screening. Empty rows fall back to the full head unless the
        budget forbids fallback.
        """

        if universe not in ("valid", "vocab"):
            raise ValueError("universe must be 'valid' or 'vocab'")
        if len(candidates) != len(states):
            raise ValueError(
                f"got {len(candidates)} candidate rows for {len(states)} states"
            )
        budget = budget or Budget()
        policy = policy or PlanPolicy(mode="calibrated")
        if policy.mode != "calibrated":
            raise ValueError("screened plans require PlanPolicy(mode='calibrated')")

        nodes: list[NodePlan] = []
        for index, (state, candidate) in enumerate(zip(states, candidates, strict=True)):
            current = _as_state(state)
            if not isinstance(candidate, TokenSet):
                raise TypeError("screened candidates must be TokenSet rows")
            if universe == "valid":
                selected = candidate.intersection(current.valid_ids)
                valid_universe: TokenSet | None = current.valid_ids
            else:
                selected = candidate
                valid_universe = None
            if selected.is_empty:
                if not budget.allow_full_fallback:
                    raise PlanBudgetError("screened candidate set is empty")
                plan = self.compile_full(state_id=current.state_id)
            else:
                plan = self.compile_calibrated(
                    selected,
                    universe=valid_universe,
                    delta=delta,
                    state_id=current.state_id,
                    budget=budget,
                    policy=policy,
                    source=source,
                    calibration_id=calibration_id,
                )
            nodes.append(NodePlan(node_id=index, plan=plan))
        return self._step_from_nodes(nodes)

    def _step_from_nodes(self, nodes: Sequence[NodePlan]) -> StepPlan:
        nodes = tuple(nodes)
        if any(node.plan.execution.kind == "full" for node in nodes):
            step = StepPlan(
                nodes=nodes,
                union_rows=TokenSet(),
                kernel="full",
                cost=self._full_cost(),
            )
            return self._with_step_id(step)

        union = TokenSet()
        for node in nodes:
            if node.plan.execution.rows is not None:
                union = union.union(node.plan.execution.rows)
        kernel = "slice_gemm" if len(nodes) > 1 else "slice_gemv"
        step = StepPlan(
            nodes=nodes,
            union_rows=union,
            kernel=kernel,
            cost=self._dense_cost(len(union)),
        )
        return self._with_step_id(step)

    @staticmethod
    def _with_id(plan: HeadPlan) -> HeadPlan:
        return HeadPlan(
            candidates=plan.candidates,
            execution=plan.execution,
            guarantee=plan.guarantee,
            state_id=plan.state_id,
            cost=plan.cost,
            source=plan.source,
            plan_id=compute_plan_id(plan),
        )

    @staticmethod
    def _with_step_id(step: StepPlan) -> StepPlan:
        return StepPlan(
            nodes=step.nodes,
            union_rows=step.union_rows,
            kernel=step.kernel,
            cost=step.cost,
            plan_id=compute_step_plan_id(step),
        )


__all__ = [
    "Budget",
    "PlanBudgetError",
    "PlanCompiler",
    "PlanPolicy",
]
