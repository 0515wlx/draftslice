from __future__ import annotations

import pytest

from draftslice.cost import CalibratedCostModel, CostCalibration, CostModel
from draftslice.plan import TokenSet
from draftslice.runtime import (
    PlanSession,
    PlanSessionConfig,
    PlanSessionError,
    StepContext,
    StepScheduler,
    validate_strict_step,
)

VOCAB = 64
HIDDEN = 8


def session(**kwargs) -> PlanSession:
    config = PlanSessionConfig(vocab_size=VOCAB, hidden_size=HIDDEN)
    return PlanSession(config, **kwargs)


def test_mixed_full_and_slice_nodes() -> None:
    plans = session()
    result = plans.build(
        [TokenSet([1, 2, 3]), TokenSet([4, 5])],
        action="slice",
        row_mask=[True, False],
    )
    assert result.action == "slice"
    assert result.sliced_rows == 1
    assert result.full_rows == 1
    assert result.step_plan.nodes[0].plan.execution.kind != "full"
    assert result.step_plan.nodes[1].plan.execution.kind == "full"
    assert result.step_plan.union_rows.ids == (1, 2, 3)
    validate_strict_step(result.step_plan)


def test_builtin_action_is_all_full() -> None:
    plans = session()
    result = plans.build(
        [TokenSet([1, 2]), TokenSet([3])],
        action="builtin",
    )
    assert result.action == "full"
    assert result.sliced_rows == 0
    assert result.full_rows == 2
    assert result.step_plan.kernel == "full"


def test_plan_cache_hit_and_compile_timing() -> None:
    plans = session()
    first = plans.build([TokenSet([1, 2, 3])], action="slice")
    second = plans.build([TokenSet([1, 2, 3])], action="slice")
    assert first.plan_cache_miss and not first.plan_cache_hit
    assert second.plan_cache_hit and not second.plan_cache_miss
    assert first.plan_compile_us >= 0.0
    assert plans.plan_misses == 1
    assert plans.plan_hits == 1


def test_candidate_cache_hit() -> None:
    plans = session()
    plans.build([TokenSet([1, 2, 3])], action="slice")
    second = plans.build([TokenSet([1, 2, 3])], action="slice")
    assert second.candidate_cache_hit
    assert plans.candidate_hits == 1


def test_tree_step_deduplicates_union_rows() -> None:
    plans = session()
    result = plans.build(
        [TokenSet([1, 2]), TokenSet([2, 3]), TokenSet([3, 4])],
        action="slice",
        tree=True,
    )
    assert result.step_plan.kernel == "slice_gemm"
    assert result.step_plan.union_rows.ids == (1, 2, 3, 4)
    assert len(result.plans) == 3
    validate_strict_step(result.step_plan)


def test_oversize_row_falls_back_to_full() -> None:
    config = PlanSessionConfig(vocab_size=VOCAB, hidden_size=HIDDEN, max_rows=2)
    plans = PlanSession(config)
    result = plans.build([TokenSet([1, 2, 3])], action="slice")
    assert result.sliced_rows == 0
    assert result.full_rows == 1
    assert result.step_plan.nodes[0].plan.execution.kind == "full"


def test_mask_action_never_builds_slice_nodes() -> None:
    plans = session()
    result = plans.build([TokenSet([1, 2])], action="mask_allpos")
    assert result.action == "full"
    assert result.step_plan.nodes[0].plan.execution.kind == "full"


def test_state_ids_are_cached_separately() -> None:
    plans = session()
    first = plans.build([TokenSet([1, 2])], state_ids=[7], action="slice")
    second = plans.build([TokenSet([1, 2])], state_ids=[8], action="slice")
    assert first.plan_cache_miss
    assert second.plan_cache_miss


def test_scheduler_defaults_full_when_uncalibrated() -> None:
    model = CostModel(vocab_size=VOCAB, hidden_size=HIDDEN)
    scheduler = StepScheduler(model)
    plans = session(scheduler=scheduler)
    result = plans.build([TokenSet([1, 2, 3])], action="slice", context=StepContext())
    assert result.action == "full"
    assert result.scheduler_decision == "full"
    assert result.scheduler_reason == "uncalibrated_cost"
    assert result.step_plan.kernel == "full"


def test_scheduler_can_select_slice_when_calibrated() -> None:
    calibration = CostCalibration(
        profile="a100",
        factors={"full": 1.0, "slice_gemv": 1.0, "slice_gemm": 1.0, "default": 1.0},
        mape=0.1,
        holdout_mape=0.1,
        calibrated=True,
    )
    calibrated = CalibratedCostModel(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        calibration=calibration,
    )
    plans = session(scheduler=StepScheduler(calibrated))
    result = plans.build([TokenSet([1, 2, 3])], action="slice", context=StepContext())
    assert result.action == "slice"
    assert result.scheduler_decision == "slice"


def test_validate_rejects_bad_union() -> None:
    plans = session()
    result = plans.build([TokenSet([1, 2, 3])], action="slice")
    broken = result.step_plan
    object.__setattr__(broken, "union_rows", TokenSet([1, 2]))
    with pytest.raises(PlanSessionError):
        validate_strict_step(broken)
