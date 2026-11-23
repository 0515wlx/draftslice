"""Cost-aware step scheduling.

The scheduler checks latency and, when an acceptance model is provided, the
minimum expected accepted length. Strict coverage is checked by the plan/cache
layer before a step reaches the scheduler; the scheduler never shrinks a strict
candidate set to meet a deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..cost.model import CostModel
from ..cost.profiles import HardwareProfile
from ..plan.types import GuaranteeKind, PlanCost, StepPlan
from .acceptance import AcceptanceEstimate, AcceptanceModel, TreeShape
from .reasons import FallbackReason


@dataclass(frozen=True, slots=True)
class StepContext:
    batch: int = 1
    profile: str | HardwareProfile = "a100"
    deadline_ms: float | None = None
    fused: bool = False
    unique_rows: int | None = None
    min_expected_tau: float | None = None
    tree: TreeShape | None = None

    def __post_init__(self) -> None:
        if self.batch <= 0:
            raise ValueError("batch must be positive")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")
        if self.unique_rows is not None and self.unique_rows < 0:
            raise ValueError("unique_rows must be non-negative")
        if self.min_expected_tau is not None and self.min_expected_tau < 0:
            raise ValueError("min_expected_tau must be non-negative")

    @property
    def deadline_us(self) -> float | None:
        return None if self.deadline_ms is None else self.deadline_ms * 1000.0

    @property
    def profile_name(self) -> str:
        return self.profile if isinstance(self.profile, str) else self.profile.name


@dataclass(frozen=True, slots=True)
class Schedule:
    step_plan: StepPlan
    cost: PlanCost
    feasible: bool
    explanation: dict[str, Any]
    expected_tau: float | None = None


class Scheduler:
    """Choose a feasible step plan under latency and acceptance constraints."""

    def __init__(
        self,
        cost_model: CostModel,
        *,
        acceptance_model: AcceptanceModel | None = None,
    ) -> None:
        self.cost_model = cost_model
        self.acceptance_model = acceptance_model

    def _predict(self, step: StepPlan, context: StepContext) -> tuple[PlanCost, float]:
        cost = self.cost_model.predict_step(
            step,
            batch=context.batch,
            unique_rows=context.unique_rows,
            profile=context.profile,
            fused=context.fused,
        )
        return cost, cost.predicted_us[context.profile_name]

    def _acceptance(
        self,
        step: StepPlan,
        context: StepContext,
    ) -> AcceptanceEstimate | None:
        if self.acceptance_model is None:
            return None
        return self.acceptance_model.predict(step, context.tree or TreeShape())

    def _feasible(
        self,
        predicted_us: float,
        estimate: AcceptanceEstimate | None,
        context: StepContext,
    ) -> bool:
        deadline = context.deadline_us
        if deadline is not None and predicted_us > deadline:
            return False
        if (
            context.min_expected_tau is not None
            and estimate is not None
            and estimate.expected_tau < context.min_expected_tau
        ):
            return False
        return True

    def choose(
        self,
        primary: StepPlan,
        context: StepContext,
        *,
        fallback: StepPlan | None = None,
    ) -> Schedule:
        primary_cost, primary_us = self._predict(primary, context)
        primary_estimate = self._acceptance(primary, context)
        primary_tau = None if primary_estimate is None else primary_estimate.expected_tau
        deadline = context.deadline_us

        if self._feasible(primary_us, primary_estimate, context):
            return Schedule(
                step_plan=primary,
                cost=primary_cost,
                feasible=True,
                expected_tau=primary_tau,
                explanation={
                    "decision": "primary",
                    "predicted_us": primary_us,
                    "expected_tau": primary_tau,
                    "deadline_us": deadline,
                },
            )

        if fallback is not None:
            fallback_cost, fallback_us = self._predict(fallback, context)
            fallback_estimate = self._acceptance(fallback, context)
            fallback_tau = None if fallback_estimate is None else fallback_estimate.expected_tau
            if self._feasible(fallback_us, fallback_estimate, context):
                return Schedule(
                    step_plan=fallback,
                    cost=fallback_cost,
                    feasible=True,
                    expected_tau=fallback_tau,
                    explanation={
                        "decision": "fallback",
                        "predicted_us": fallback_us,
                        "primary_us": primary_us,
                        "expected_tau": fallback_tau,
                        "primary_tau": primary_tau,
                        "deadline_us": deadline,
                    },
                )
            if fallback_us < primary_us:
                return Schedule(
                    step_plan=fallback,
                    cost=fallback_cost,
                    feasible=False,
                    expected_tau=fallback_tau,
                    explanation={
                        "decision": "cheapest",
                        "predicted_us": fallback_us,
                        "primary_us": primary_us,
                        "expected_tau": fallback_tau,
                        "primary_tau": primary_tau,
                        "deadline_us": deadline,
                    },
                )

        return Schedule(
            step_plan=primary,
            cost=primary_cost,
            feasible=False,
            expected_tau=primary_tau,
            explanation={
                "decision": "infeasible",
                "predicted_us": primary_us,
                "expected_tau": primary_tau,
                "deadline_us": deadline,
            },
        )

    def choose_best(
        self,
        candidates: list[StepPlan] | tuple[StepPlan, ...],
        context: StepContext,
        *,
        fallback: StepPlan | None = None,
    ) -> Schedule:
        """Pick the feasible plan with the highest expected_tau / predicted_us.

        If no candidate is feasible, fall back to :meth:`choose` on the cheapest
        candidate and the fallback plan.
        """

        if not candidates:
            raise ValueError("choose_best needs at least one candidate")

        evaluated: list[tuple[float, Schedule]] = []
        for step in candidates:
            cost, predicted_us = self._predict(step, context)
            estimate = self._acceptance(step, context)
            tau = None if estimate is None else estimate.expected_tau
            if not self._feasible(predicted_us, estimate, context):
                continue
            score = (
                (tau if tau is not None else 1.0) / predicted_us
                if predicted_us > 0
                else float("inf")
            )
            schedule = Schedule(
                step_plan=step,
                cost=cost,
                feasible=True,
                expected_tau=tau,
                explanation={
                    "decision": "best",
                    "predicted_us": predicted_us,
                    "expected_tau": tau,
                    "objective": score,
                },
            )
            evaluated.append((score, schedule))

        if evaluated:
            return max(evaluated, key=lambda item: item[0])[1]

        cheapest = min(
            candidates,
            key=lambda step: self._predict(step, context)[1],
        )
        return self.choose(cheapest, context, fallback=fallback)


class StepScheduler:
    """Production gate around :class:`Scheduler`.

    A predicted latency may only select a slice when the cost model is
    calibrated (``docs/production-architecture.md`` D4). Until then every step
    defaults to the full head, even if a profile cell is enabled. This keeps the
    cost model from becoming an unmeasured production switch.
    """

    def __init__(
        self,
        cost_model: CostModel,
        *,
        calibrated: bool | None = None,
        acceptance_model: AcceptanceModel | None = None,
        max_mape: float = 0.25,
    ) -> None:
        if not (0.0 < max_mape <= 1.0):
            raise ValueError("max_mape must be in (0, 1]")
        self.cost_model = cost_model
        self.max_mape = max_mape
        self.calibrated = (
            bool(getattr(cost_model, "calibrated", False))
            if calibrated is None
            else bool(calibrated)
        )
        self.scheduler = Scheduler(cost_model, acceptance_model=acceptance_model)

    def choose(
        self,
        primary: StepPlan,
        fallback: StepPlan,
        *,
        context: StepContext | None = None,
        cell: Any | None = None,
    ) -> Schedule:
        context = context or StepContext(batch=max(1, len(primary.nodes)))
        if cell is not None and not bool(getattr(cell, "is_profitable", True)):
            return self._full(fallback, context, FallbackReason.PROFILE_DISABLED)
        if not self.calibrated:
            return self._full(fallback, context, FallbackReason.UNCALIBRATED_COST)
        if not _strict_covers(primary):
            return self._full(fallback, context, FallbackReason.COVERAGE_MISMATCH)

        schedule = self.scheduler.choose(primary, context, fallback=fallback)
        chosen = schedule.step_plan
        if chosen.kernel == "full" or any(
            node.plan.execution.kind == "full" for node in chosen.nodes
        ):
            schedule.explanation["reason"] = FallbackReason.COST_NOT_PROFITABLE.value
            return schedule
        if not _strict_covers(chosen):
            return self._full(fallback, context, FallbackReason.COVERAGE_MISMATCH)

        slice_cost = self.scheduler._predict(chosen, context)[1]
        full_cost = self.scheduler._predict(fallback, context)[1]
        if slice_cost >= full_cost:
            return self._full(
                fallback,
                context,
                FallbackReason.COST_NOT_PROFITABLE,
                primary_us=slice_cost,
                full_us=full_cost,
            )
        schedule.explanation["reason"] = "cost_profitable"
        schedule.explanation["slice_us"] = slice_cost
        schedule.explanation["full_us"] = full_cost
        return schedule

    def _full(
        self,
        fallback: StepPlan,
        context: StepContext,
        reason: FallbackReason,
        **extra: Any,
    ) -> Schedule:
        cost, predicted_us = self.scheduler._predict(fallback, context)
        return Schedule(
            step_plan=fallback,
            cost=cost,
            feasible=True,
            explanation={
                "decision": "full",
                "reason": reason.value,
                "predicted_us": predicted_us,
                **extra,
            },
        )


def _strict_covers(step: StepPlan) -> bool:
    if step.kernel == "full":
        return True
    for node in step.nodes:
        plan = node.plan
        if plan.guarantee.kind is not GuaranteeKind.EXACT_VALID_SET:
            continue
        if not plan.execution.covers(plan.candidates):
            return False
        if not plan.candidates.issubset(step.union_rows):
            return False
    return True


__all__ = ["Schedule", "Scheduler", "StepContext", "StepScheduler"]
