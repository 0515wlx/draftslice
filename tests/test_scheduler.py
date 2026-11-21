from __future__ import annotations

import pytest

from draftslice.cost import CostModel
from draftslice.plan import (
    Execution,
    Guarantee,
    HeadPlan,
    NodePlan,
    PlanCost,
    StepPlan,
    TokenSet,
)
from draftslice.runtime import Scheduler, StepContext

VOCAB = 100_000
HIDDEN = 2_560


def slice_step(rows: int) -> StepPlan:
    ids = TokenSet(range(rows))
    plan = HeadPlan(
        candidates=ids,
        execution=Execution(kind="slice_gemv", rows=ids),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
        cost=PlanCost(),
    )
    return StepPlan(
        nodes=(NodePlan(0, plan),),
        union_rows=ids,
        kernel="slice_gemm",
        cost=PlanCost(),
    )


def full_step() -> StepPlan:
    plan = HeadPlan(
        candidates=TokenSet(range(VOCAB)),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        state_id=None,
        cost=PlanCost(),
    )
    return StepPlan(
        nodes=(NodePlan(0, plan),),
        union_rows=TokenSet(),
        kernel="full",
        cost=PlanCost(),
    )


def model() -> CostModel:
    return CostModel(vocab_size=VOCAB, hidden_size=HIDDEN)


def test_primary_is_feasible_with_a_loose_deadline() -> None:
    scheduler = Scheduler(model())
    schedule = scheduler.choose(slice_step(1_000), StepContext(deadline_ms=10.0))
    assert schedule.feasible
    assert schedule.explanation["decision"] == "primary"
    assert schedule.cost.predicted_us["a100"] > 0


def test_fallback_is_chosen_when_primary_is_too_slow() -> None:
    scheduler = Scheduler(model())
    schedule = scheduler.choose(
        slice_step(90_000),
        StepContext(deadline_ms=0.5),
        fallback=full_step(),
    )
    assert schedule.feasible
    assert schedule.explanation["decision"] == "fallback"
    assert schedule.step_plan.kernel == "full"


def test_cheapest_is_chosen_when_both_are_infeasible() -> None:
    scheduler = Scheduler(model())
    schedule = scheduler.choose(
        slice_step(90_000),
        StepContext(deadline_ms=0.1),
        fallback=full_step(),
    )
    assert not schedule.feasible
    assert schedule.explanation["decision"] == "cheapest"
    assert schedule.step_plan.kernel == "full"


def test_infeasible_without_fallback() -> None:
    scheduler = Scheduler(model())
    schedule = scheduler.choose(slice_step(90_000), StepContext(deadline_ms=0.1))
    assert not schedule.feasible
    assert schedule.explanation["decision"] == "infeasible"


def test_no_deadline_is_always_feasible() -> None:
    scheduler = Scheduler(model())
    schedule = scheduler.choose(slice_step(90_000), StepContext())
    assert schedule.feasible
    assert schedule.explanation["deadline_us"] is None


def test_step_context_validation() -> None:
    with pytest.raises(ValueError):
        StepContext(batch=0)
    with pytest.raises(ValueError):
        StepContext(deadline_ms=0.0)
    with pytest.raises(ValueError):
        StepContext(unique_rows=-1)


def test_acceptance_constraint_can_force_fallback() -> None:
    from draftslice.plan import Guarantee
    from draftslice.runtime.acceptance import CoverageAcceptanceModel, TreeShape

    ids = TokenSet(range(1_000))
    calibrated_plan = HeadPlan(
        candidates=ids,
        execution=Execution(kind="slice_gemv", rows=ids),
        guarantee=Guarantee.calibrated(0.8),
        state_id=0,
    )
    calibrated_step = StepPlan(
        nodes=(NodePlan(0, calibrated_plan),),
        union_rows=ids,
        kernel="slice_gemm",
    )
    acceptance = CoverageAcceptanceModel()
    scheduler = Scheduler(model(), acceptance_model=acceptance)
    context = StepContext(
        min_expected_tau=2.0,
        tree=TreeShape(depth=3),
    )
    schedule = scheduler.choose(calibrated_step, context, fallback=full_step())
    assert schedule.feasible
    assert schedule.step_plan.kernel == "full"
    assert schedule.expected_tau == 3.0


def test_choose_best_maximizes_tau_per_latency() -> None:
    scheduler = Scheduler(model())
    small = slice_step(1_000)
    large = slice_step(4_000)
    context = StepContext()
    schedule = scheduler.choose_best([large, small], context)
    assert schedule.feasible
    assert schedule.step_plan is small
    assert schedule.explanation["decision"] == "best"
