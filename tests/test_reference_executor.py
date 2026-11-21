from __future__ import annotations

import pytest

from draftslice.kernels.base import KernelError
from draftslice.plan import Execution, Guarantee, HeadPlan, TokenSet
from draftslice.runtime import ReferenceExecutor

WEIGHT = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
]


def strict_plan(ids=(0, 2, 3)) -> HeadPlan:
    tokens = TokenSet(ids)
    return HeadPlan(
        candidates=tokens,
        execution=Execution(kind="slice_gemv", rows=tokens),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )


def test_slice_execution_matches_expected_values() -> None:
    executor = ReferenceExecutor(WEIGHT)
    result = executor.execute([[3.0, 4.0], [1.0, 2.0]], strict_plan())
    assert result.rows.ids == (0, 2, 3)
    assert result.logits == ((3.0, 7.0, 6.0), (1.0, 3.0, 2.0))
    assert result.candidate_logits(strict_plan()) == result.logits


def test_single_vector_is_wrapped_as_batch() -> None:
    result = ReferenceExecutor(WEIGHT).execute([3.0, 4.0], strict_plan())
    assert result.logits == ((3.0, 7.0, 6.0),)


def test_full_fallback_projects_to_candidates() -> None:
    plan = HeadPlan(
        candidates=TokenSet([1, 3]),
        execution=Execution(kind="full"),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )
    result = ReferenceExecutor(WEIGHT).execute([[3.0, 4.0]], plan)
    assert result.rows.ids == (0, 1, 2, 3)
    assert result.candidate_logits(plan) == ((4.0, 6.0),)


def test_empty_candidates() -> None:
    plan = HeadPlan(
        candidates=TokenSet(),
        execution=Execution(kind="slice_gemv", rows=TokenSet()),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )
    result = ReferenceExecutor(WEIGHT).execute([[3.0, 4.0]], plan)
    assert result.logits == ((),)
    assert result.candidate_logits(plan) == ((),)


def test_out_of_range_row_is_rejected() -> None:
    plan = HeadPlan(
        candidates=TokenSet([9]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([9])),
        guarantee=Guarantee.calibrated(0.05),
        state_id=None,
    )
    with pytest.raises(KernelError):
        ReferenceExecutor(WEIGHT).execute([[3.0, 4.0]], plan)
