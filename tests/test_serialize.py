from __future__ import annotations

import json

import pytest

from draftslice.plan import Execution, Guarantee, HeadPlan, NodePlan, PlanCost, StepPlan, TokenSet
from draftslice.plan.serialize import (
    PlanFormatError,
    compute_plan_id,
    dumps,
    loads,
    plan_to_dict,
    read_plan,
    step_plan_from_dict,
    step_plan_to_dict,
    verify_plan_id,
    write_plan,
)


def make_plan() -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet([1, 3, 5]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([1, 3, 5])),
        guarantee=Guarantee.exact_valid_set(),
        state_id=4,
        cost=PlanCost(dynamic_bytes=30, macs=15, launch_count=1, predicted_us={"a100": 1.5}),
        source="constraint",
    )


def test_plan_round_trip() -> None:
    plan = make_plan()
    text = dumps(plan)
    restored = loads(text)
    assert restored.candidates == plan.candidates
    assert restored.execution == plan.execution
    assert restored.guarantee == plan.guarantee
    assert restored.state_id == plan.state_id
    assert restored.cost == plan.cost
    assert restored.source == plan.source
    assert restored.plan_id
    assert verify_plan_id(restored)


def test_plan_id_is_stable() -> None:
    first = make_plan()
    second = make_plan()
    assert compute_plan_id(first) == compute_plan_id(second)
    assert plan_to_dict(first)["plan_id"] == compute_plan_id(first)


def test_write_and_read_plan(tmp_path) -> None:
    path = tmp_path / "plan.json"
    plan = make_plan()
    write_plan(path, plan)
    restored = read_plan(path)
    assert restored.candidates == plan.candidates
    assert restored.plan_id == compute_plan_id(plan)


def test_full_plan_round_trip() -> None:
    plan = HeadPlan(
        candidates=TokenSet([0, 1, 2]),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        state_id=None,
        source="full",
    )
    restored = loads(dumps(plan))
    assert restored.execution.kind == "full"
    assert restored.execution.rows is None


def test_rejects_bad_schema_version() -> None:
    value = plan_to_dict(make_plan())
    value["schema_version"] = "plan/999"
    with pytest.raises(PlanFormatError):
        loads(json.dumps(value))


def test_rejects_count_mismatch() -> None:
    value = plan_to_dict(make_plan())
    value["candidates"]["count"] = 99
    with pytest.raises(PlanFormatError):
        loads(json.dumps(value))


def test_step_plan_round_trip() -> None:
    first = make_plan()
    second = HeadPlan(
        candidates=TokenSet([3, 7]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([3, 7])),
        guarantee=Guarantee.exact_valid_set(),
        state_id=5,
    )
    step = StepPlan(
        nodes=(NodePlan(0, first), NodePlan(1, second)),
        union_rows=TokenSet([1, 3, 5, 7]),
        kernel="slice_gemm",
        cost=PlanCost(dynamic_bytes=40, macs=20, launch_count=1),
    )
    restored = step_plan_from_dict(step_plan_to_dict(step))
    assert restored.size == 2
    assert restored.union_rows == step.union_rows
    assert restored.kernel == "slice_gemm"
    assert restored.nodes[1].plan.candidates == second.candidates
