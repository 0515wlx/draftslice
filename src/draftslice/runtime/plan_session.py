"""Per-step plan sessions for the production executor.

A plan session is the bridge between the engine adapter's per-step inputs
(grammar valid sets, a policy decision) and the executors in
:mod:`draftslice.runtime.executor` / :mod:`draftslice.runtime.union`.

Design rules (``docs/production-architecture.md`` D4):

* a row without an enabled cell is a **full node**, never a guessed slice;
* only exact valid sets become slice nodes;
* a tree step deduplicates all slice rows into one union;
* plan/candidate caches and their hit rates are first-class telemetry;
* any uncertainty (oversize set, empty candidates, missing coverage) falls back
  to a full node with a stable reason.

The session is engine-agnostic and does not import torch.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..cache.candidate_cache import CandidateSetCache
from ..cache.plan_cache import LRUPlanCache
from ..plan.compiler import PlanCompiler
from ..plan.serialize import compute_plan_id, compute_step_plan_id
from ..plan.types import (
    Execution,
    Guarantee,
    HeadPlan,
    NodePlan,
    PlanCost,
    StepPlan,
    TokenSet,
)
from .scheduler import StepContext, StepScheduler


class PlanSessionError(ValueError):
    """A plan session cannot build a strict step plan."""


@dataclass(frozen=True, slots=True)
class PlanSessionConfig:
    """Shape and budget limits for one session."""

    vocab_size: int
    hidden_size: int
    dtype_bytes: int = 2
    max_rows: int = 32_768
    max_cache_bytes: int = 64 * 1024 * 1024
    max_candidate_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise PlanSessionError("vocab_size must be positive")
        if self.hidden_size < 0:
            raise PlanSessionError("hidden_size must be non-negative")
        if self.dtype_bytes <= 0:
            raise PlanSessionError("dtype_bytes must be positive")
        if self.max_rows <= 0:
            raise PlanSessionError("max_rows must be positive")
        if self.max_cache_bytes <= 0 or self.max_candidate_bytes <= 0:
            raise PlanSessionError("cache budgets must be positive")


@dataclass(frozen=True, slots=True)
class PlanSessionResult:
    """One step's plans plus cache/timing telemetry."""

    step_plan: StepPlan
    action: str
    plans: tuple[HeadPlan, ...]
    sliced_rows: int
    full_rows: int
    plan_compile_us: float
    plan_cache_hit: bool = False
    plan_cache_miss: bool = False
    candidate_cache_hit: bool = False
    candidate_cache_miss: bool = False
    scheduler_reason: str = ""
    scheduler_decision: str = ""
    explanation: Mapping[str, Any] = field(default_factory=dict)

    @property
    def node_logits_count(self) -> int:
        return len(self.plans)

    @property
    def scheduled_full(self) -> bool:
        return self.scheduler_decision == "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "plan_id": self.step_plan.plan_id,
            "nodes": len(self.plans),
            "sliced_rows": self.sliced_rows,
            "full_rows": self.full_rows,
            "plan_compile_us": self.plan_compile_us,
            "plan_cache_hit": self.plan_cache_hit,
            "plan_cache_miss": self.plan_cache_miss,
            "candidate_cache_hit": self.candidate_cache_hit,
            "candidate_cache_miss": self.candidate_cache_miss,
            "scheduler_reason": self.scheduler_reason,
            "scheduler_decision": self.scheduler_decision,
        }


class PlanSession:
    """Compile and cache one :class:`StepPlan` per draft step."""

    def __init__(
        self,
        config: PlanSessionConfig,
        *,
        plan_cache: LRUPlanCache | None = None,
        candidate_cache: CandidateSetCache | None = None,
        scheduler: StepScheduler | None = None,
        compiler: PlanCompiler | None = None,
    ) -> None:
        self.config = config
        self.plan_cache = plan_cache or LRUPlanCache(max_bytes=config.max_cache_bytes)
        self.candidate_cache = candidate_cache or CandidateSetCache(
            max_bytes=config.max_candidate_bytes
        )
        self.scheduler = scheduler
        self.compiler = compiler or PlanCompiler(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            dtype_bytes=config.dtype_bytes,
            plan_cache=self.plan_cache,
        )
        self.plan_hits = 0
        self.plan_misses = 0
        self.candidate_hits = 0
        self.candidate_misses = 0
        self.compile_calls = 0
        self.compile_us_total = 0.0

    # -- construction -----------------------------------------------------
    def full_plan(
        self,
        *,
        state_id: int | None = None,
        candidates: TokenSet | None = None,
        source: str = "full",
    ) -> HeadPlan:
        plan = HeadPlan(
            candidates=candidates or TokenSet(),
            execution=Execution(kind="full"),
            guarantee=Guarantee.full(),
            state_id=state_id,
            cost=PlanCost.dense(self.config.vocab_size, self.config.hidden_size),
            source=source,
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

    def full_step(
        self,
        rows: int | Sequence[TokenSet | None],
        *,
        state_ids: Sequence[int | None] | None = None,
    ) -> StepPlan:
        """A step plan in which every node executes the full head."""

        if isinstance(rows, int):
            count = rows
            valid_rows: Sequence[TokenSet | None] = [None] * count
        else:
            valid_rows = rows
            count = len(rows)
        if state_ids is not None and len(state_ids) != count:
            raise PlanSessionError("state_ids length does not match rows")
        nodes = tuple(
            NodePlan(
                node_id=index,
                plan=self.full_plan(
                    state_id=None if state_ids is None else state_ids[index],
                    candidates=valid_rows[index],
                ),
            )
            for index in range(count)
        )
        return self._step_from_nodes(nodes, kernel="full")

    # -- main entry point -------------------------------------------------
    def build(
        self,
        valid_sets: Sequence[TokenSet | None],
        *,
        state_ids: Sequence[int | None] | None = None,
        action: str = "slice",
        row_mask: Sequence[bool] | None = None,
        candidates: Sequence[TokenSet | None] | None = None,
        tree: bool = False,
        context: StepContext | None = None,
        cell: Any | None = None,
    ) -> PlanSessionResult:
        """Build one step plan from per-row grammar valid sets.

        ``action`` comes from :class:`~draftslice.runtime.policy.PolicyDecision`
        (``builtin``/``full``/``slice``/``mask``/``mask_allpos``). Only
        ``slice`` produces slice nodes; everything else is a full node.
        """

        rows = len(valid_sets)
        if state_ids is not None and len(state_ids) != rows:
            raise PlanSessionError("state_ids length does not match valid_sets")
        if row_mask is None:
            row_mask = [True] * rows
        if len(row_mask) != rows:
            raise PlanSessionError("row_mask length does not match valid_sets")
        if candidates is not None and len(candidates) != rows:
            raise PlanSessionError("candidates length does not match valid_sets")

        full_fallback = self.full_step(valid_sets, state_ids=state_ids)
        if action not in ("slice",):
            return self._result(
                full_fallback,
                action="full",
                plans=tuple(node.plan for node in full_fallback.nodes),
                sliced_rows=0,
                full_rows=rows,
                compile_us=0.0,
                reason="action_full",
            )

        t0 = time.perf_counter()
        nodes: list[NodePlan] = []
        any_hit = False
        any_miss = False
        any_candidate_hit = False
        any_candidate_miss = False
        for index, valid in enumerate(valid_sets):
            state_id = None if state_ids is None else state_ids[index]
            if not row_mask[index] or valid is None or valid.is_empty:
                nodes.append(
                    NodePlan(
                        node_id=index,
                        plan=self.full_plan(state_id=state_id, candidates=valid),
                    )
                )
                continue
            selected = self._candidate_set(valid, None if candidates is None else candidates[index])
            if selected is None:
                nodes.append(
                    NodePlan(
                        node_id=index,
                        plan=self.full_plan(state_id=state_id, candidates=valid),
                    )
                )
                continue
            candidate_hit, candidate_miss, selected = self._cached_candidates(valid, selected)
            any_candidate_hit = any_candidate_hit or candidate_hit
            any_candidate_miss = any_candidate_miss or candidate_miss
            if len(selected) > self.config.max_rows or selected.is_empty:
                nodes.append(
                    NodePlan(
                        node_id=index,
                        plan=self.full_plan(state_id=state_id, candidates=valid),
                    )
                )
                continue
            plan, hit, miss = self._slice_plan(state_id, selected)
            any_hit = any_hit or hit
            any_miss = any_miss or miss
            nodes.append(NodePlan(node_id=index, plan=plan))

        kernel = "full" if all(
            node.plan.execution.kind == "full" for node in nodes
        ) else ("slice_gemm" if sum(
            1 for node in nodes if node.plan.execution.kind != "full"
        ) > 1 or tree else "slice_gemv")
        primary = self._step_from_nodes(tuple(nodes), kernel=kernel)
        compile_us = (time.perf_counter() - t0) * 1e6

        scheduler_reason = ""
        scheduler_decision = "slice"
        if self.scheduler is not None:
            schedule = self.scheduler.choose(
                primary,
                full_fallback,
                context=context or StepContext(batch=max(1, rows)),
                cell=cell,
            )
            scheduler_reason = str(schedule.explanation.get("reason", ""))
            scheduler_decision = "full" if (
                schedule.step_plan.kernel == "full"
                or any(node.plan.execution.kind == "full" for node in schedule.step_plan.nodes)
            ) else "slice"
            primary = schedule.step_plan
            compile_us += float(schedule.explanation.get("compile_us", 0.0) or 0.0)

        sliced_rows = sum(
            1 for node in primary.nodes if node.plan.execution.kind != "full"
        )
        return self._result(
            primary,
            action="slice" if sliced_rows else "full",
            plans=tuple(node.plan for node in primary.nodes),
            sliced_rows=sliced_rows,
            full_rows=rows - sliced_rows,
            compile_us=compile_us,
            hit=any_hit,
            miss=any_miss,
            candidate_hit=any_candidate_hit,
            candidate_miss=any_candidate_miss,
            reason=scheduler_reason or "strict_slice",
            scheduler_decision=scheduler_decision,
        )

    # -- internals --------------------------------------------------------
    def _candidate_set(
        self, valid: TokenSet, screened: TokenSet | None
    ) -> TokenSet | None:
        if screened is None:
            return valid
        return valid.intersection(screened)

    def _cached_candidates(
        self, valid: TokenSet, selected: TokenSet
    ) -> tuple[bool, bool, TokenSet]:
        # CandidateSetCache supports opaque hashable keys; use the pair of
        # token tuples directly so different valid/screened combinations cannot
        # collide after grouping.
        key = (valid.ids, selected.ids)
        cached = self.candidate_cache.get_keyed(key)
        if cached is not None:
            self.candidate_hits += 1
            return True, False, cached
        self.candidate_cache.put_keyed(key, selected)
        self.candidate_misses += 1
        return False, True, selected

    def _slice_plan(
        self, state_id: int | None, candidates: TokenSet
    ) -> tuple[HeadPlan, bool, bool]:
        key = (
            "session",
            state_id,
            candidates.ids,
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.dtype_bytes,
            self.config.max_rows,
        )
        cached = self.plan_cache.get(key)
        if cached is not None:
            self.plan_hits += 1
            return cached, True, False
        self.plan_misses += 1
        self.compile_calls += 1
        start = time.perf_counter()
        plan = HeadPlan(
            candidates=candidates,
            execution=Execution(kind="slice_gemv", rows=candidates),
            guarantee=Guarantee.exact_valid_set(),
            state_id=state_id,
            cost=PlanCost.dense(
                len(candidates), self.config.hidden_size, self.config.dtype_bytes
            ),
            source="plan_session",
        )
        plan = HeadPlan(
            candidates=plan.candidates,
            execution=plan.execution,
            guarantee=plan.guarantee,
            state_id=plan.state_id,
            cost=plan.cost,
            source=plan.source,
            plan_id=compute_plan_id(plan),
        )
        self.compile_us_total += (time.perf_counter() - start) * 1e6
        self.plan_cache.put_keyed(key, plan)
        return plan, False, True

    def _step_from_nodes(self, nodes: tuple[NodePlan, ...], *, kernel: str) -> StepPlan:
        if kernel == "full":
            union = TokenSet()
        else:
            ids: list[int] = []
            for node in nodes:
                if node.plan.execution.rows is None:
                    continue
                ids.extend(node.plan.execution.rows.ids)
            union = TokenSet(ids)
            for node in nodes:
                if node.plan.execution.kind == "full":
                    continue
                if not node.plan.candidates.issubset(union):
                    raise PlanSessionError(
                        "step union does not cover a slice node's candidates"
                    )
        step = StepPlan(
            nodes=nodes,
            union_rows=union,
            kernel=kernel,  # type: ignore[arg-type]
            cost=PlanCost.dense(
                len(union) if kernel != "full" else self.config.vocab_size,
                self.config.hidden_size,
                self.config.dtype_bytes,
            ),
        )
        return StepPlan(
            nodes=step.nodes,
            union_rows=step.union_rows,
            kernel=step.kernel,
            cost=step.cost,
            plan_id=compute_step_plan_id(step),
        )

    def _result(
        self,
        step: StepPlan,
        *,
        action: str,
        plans: tuple[HeadPlan, ...],
        sliced_rows: int,
        full_rows: int,
        compile_us: float,
        hit: bool = False,
        miss: bool = False,
        candidate_hit: bool = False,
        candidate_miss: bool = False,
        reason: str,
        scheduler_decision: str = "",
    ) -> PlanSessionResult:
        return PlanSessionResult(
            step_plan=step,
            action=action,
            plans=plans,
            sliced_rows=sliced_rows,
            full_rows=full_rows,
            plan_compile_us=compile_us,
            plan_cache_hit=hit,
            plan_cache_miss=miss,
            candidate_cache_hit=candidate_hit,
            candidate_cache_miss=candidate_miss,
            scheduler_reason=reason,
            scheduler_decision=scheduler_decision,
            explanation={
                "reason": reason,
                "plan_cache_hits": self.plan_hits,
                "plan_cache_misses": self.plan_misses,
                "candidate_cache_hits": self.candidate_hits,
                "candidate_cache_misses": self.candidate_misses,
            },
        )

    def cache_stats(self) -> dict[str, Any]:
        return {
            "plan": asdict(self.plan_cache.stats),
            "candidate": asdict(self.candidate_cache.stats),
        }


def validate_strict_step(step: StepPlan) -> None:
    """Raise if any strict node's execution rows miss its candidates."""

    if step.kernel == "full":
        return
    for node in step.nodes:
        plan = node.plan
        if plan.guarantee.kind.value != "exact_valid_set":
            continue
        if not plan.execution.covers(plan.candidates):
            raise PlanSessionError(
                f"node {node.node_id} execution rows miss its strict candidates"
            )
        if not plan.candidates.issubset(step.union_rows):
            raise PlanSessionError(
                f"node {node.node_id} is not covered by the step union"
            )


__all__ = [
    "PlanSession",
    "PlanSessionConfig",
    "PlanSessionError",
    "PlanSessionResult",
    "validate_strict_step",
]
