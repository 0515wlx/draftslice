from __future__ import annotations

import pytest

import draftslice
from draftslice.plan import (
    Execution,
    Guarantee,
    GuaranteeKind,
    HeadPlan,
    NodePlan,
    PlanCost,
    Slice,
    StepPlan,
    TokenSet,
)


def test_version() -> None:
    assert draftslice.__version__ == "0.0.1"


def test_token_set_normalizes_and_is_hashable() -> None:
    tokens = TokenSet([3, 1, 3, 2])
    assert tokens.ids == (1, 2, 3)
    assert len(tokens) == 3
    assert 2 in tokens
    assert 4 not in tokens
    assert {tokens: "value"}[TokenSet([1, 2, 3])] == "value"


def test_token_set_operations() -> None:
    left = TokenSet([1, 2, 3])
    right = TokenSet([3, 4])
    assert left.union(right).ids == (1, 2, 3, 4)
    assert left.intersection(right).ids == (3,)
    assert left.difference(right).ids == (1, 2)
    assert TokenSet([1, 2]).issubset(left)
    assert left.covers(TokenSet([1, 3]))


def test_token_set_rejects_invalid_ids() -> None:
    with pytest.raises(TypeError):
        TokenSet([1, "2"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        TokenSet([-1])


def test_slice_size() -> None:
    s = Slice(state_id=7, token_ids=(1, 2, 3))
    assert s.size == 3


def test_guarantees_validate() -> None:
    assert Guarantee.exact_valid_set().kind is GuaranteeKind.EXACT_VALID_SET
    assert Guarantee.full().kind is GuaranteeKind.FULL
    assert Guarantee.calibrated(0.05).delta == 0.05
    with pytest.raises(ValueError):
        Guarantee(kind=GuaranteeKind.CALIBRATED, universe="vocab", delta=None)
    with pytest.raises(ValueError):
        Guarantee(kind=GuaranteeKind.FULL, universe="valid")


def test_execution_validation() -> None:
    with pytest.raises(ValueError):
        Execution(kind="full", rows=TokenSet([1]))
    with pytest.raises(ValueError):
        Execution(kind="slice_gemv", rows=None)
    assert Execution(kind="full").covers(TokenSet([1, 2, 3]))
    assert Execution(kind="slice_gemv", rows=TokenSet([1, 2])).covers(TokenSet([1]))


def test_head_plan_from_slice() -> None:
    plan = HeadPlan.from_slice(Slice(state_id=7, token_ids=(1, 2, 3)), hidden_size=4)
    assert plan.state_id == 7
    assert plan.size == 3
    assert plan.token_ids == (1, 2, 3)
    assert plan.kernel == "slice_gemv"
    assert plan.estimated_bytes == 3 * 4 * 2
    assert plan.is_strict
    assert plan.covers([1, 3])


def test_strict_plan_requires_coverage() -> None:
    with pytest.raises(ValueError):
        HeadPlan(
            candidates=TokenSet([1, 2, 3]),
            execution=Execution(kind="slice_gemv", rows=TokenSet([1, 2])),
            guarantee=Guarantee.exact_valid_set(),
            state_id=0,
        )


def test_plan_cost_validation() -> None:
    assert PlanCost.zero().total_bytes == 0
    assert PlanCost.dense(10, 4).dynamic_bytes == 80
    with pytest.raises(ValueError):
        PlanCost(dynamic_bytes=-1)


def test_step_plan_requires_union_coverage() -> None:
    first = HeadPlan(
        candidates=TokenSet([1, 2]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([1, 2])),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )
    second = HeadPlan(
        candidates=TokenSet([2, 3]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([2, 3])),
        guarantee=Guarantee.exact_valid_set(),
        state_id=1,
    )
    step = StepPlan(
        nodes=(NodePlan(0, first), NodePlan(1, second)),
        union_rows=TokenSet([1, 2, 3]),
        kernel="slice_gemm",
    )
    assert step.covers([3])

    with pytest.raises(ValueError):
        StepPlan(
            nodes=(NodePlan(0, first), NodePlan(1, second)),
            union_rows=TokenSet([1, 2]),
            kernel="slice_gemm",
        )
