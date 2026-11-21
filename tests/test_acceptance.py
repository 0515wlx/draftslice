from __future__ import annotations

import pytest

from draftslice.plan import (
    Execution,
    Guarantee,
    GuaranteeKind,
    HeadPlan,
    NodePlan,
    StepPlan,
    TokenSet,
)
from draftslice.runtime.acceptance import (
    AcceptanceError,
    CoverageAcceptanceModel,
    StaticAcceptanceModel,
    TreeShape,
    coverage_from_measurements,
)


def step_plan(rows: int, *, calibrated: bool = False) -> StepPlan:
    ids = TokenSet(range(rows))
    guarantee = Guarantee.calibrated(0.05) if calibrated else Guarantee.exact_valid_set()
    plan = HeadPlan(
        candidates=ids,
        execution=Execution(kind="slice_gemv", rows=ids),
        guarantee=guarantee,
        state_id=0,
    )
    return StepPlan(nodes=(NodePlan(0, plan),), union_rows=ids, kernel="slice_gemm")


def test_tree_shape_validation() -> None:
    assert TreeShape(depth=4, width=2).depth == 4
    with pytest.raises(ValueError):
        TreeShape(depth=0)
    with pytest.raises(ValueError):
        TreeShape(width=0)


def test_strict_plan_has_full_prefix_coverage() -> None:
    model = CoverageAcceptanceModel()
    estimate = model.predict(step_plan(10), TreeShape(depth=4))
    assert estimate.prefix_coverage == (1.0, 1.0, 1.0, 1.0)
    assert estimate.expected_tau == 4.0


def test_calibrated_plan_uses_delta_proxy() -> None:
    model = CoverageAcceptanceModel(default_coverage=0.9)
    estimate = model.predict(step_plan(10, calibrated=True), TreeShape(depth=3))
    assert estimate.prefix_coverage == (0.95, 0.95, 0.95)
    assert estimate.expected_tau == pytest.approx(2.85)


def test_explicit_prefix_coverage_is_truncated_or_extended() -> None:
    model = CoverageAcceptanceModel()
    estimate = model.predict(
        step_plan(10),
        TreeShape(depth=3),
        prefix_coverage=[1.0, 0.8],
    )
    assert estimate.prefix_coverage == (1.0, 0.8, 0.8)


def test_static_model_and_measurement_conversion() -> None:
    rates = coverage_from_measurements([100, 80, 50], [100, 100, 100])
    assert rates == (1.0, 0.8, 0.5)
    estimate = StaticAcceptanceModel(rates).predict(step_plan(10), TreeShape(depth=2))
    assert estimate.expected_tau == pytest.approx(1.8)


def test_acceptance_rejects_bad_inputs() -> None:
    with pytest.raises(AcceptanceError):
        coverage_from_measurements([1], [1, 2])
    with pytest.raises(AcceptanceError):
        StaticAcceptanceModel([])
    plan = step_plan(10)
    assert plan.nodes[0].plan.guarantee.kind is GuaranteeKind.EXACT_VALID_SET
