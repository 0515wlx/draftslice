from __future__ import annotations

import pytest

from draftslice.constraints import ConstraintState, FixedConstraintSession
from draftslice.plan import GuaranteeKind, TokenSet
from draftslice.plan.compiler import Budget, PlanBudgetError, PlanCompiler, PlanPolicy


def state(state_id: int = 0, ids=(1, 3, 5)) -> ConstraintState:
    return ConstraintState(state_id=state_id, valid_ids=TokenSet(ids))


def test_compile_strict_slice() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    plan = compiler.compile_strict(state())
    assert plan.guarantee.kind is GuaranteeKind.EXACT_VALID_SET
    assert plan.candidates.ids == (1, 3, 5)
    assert plan.execution.kind == "slice_gemv"
    assert plan.execution.rows == plan.candidates
    assert plan.cost.dynamic_bytes == 3 * 4 * 2
    assert plan.cost.macs == 3 * 4
    assert plan.plan_id.startswith("sha256:")


def test_compile_strict_accepts_session() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    session = FixedConstraintSession(state())
    plan = compiler.compile_strict(session)
    assert plan.candidates.ids == (1, 3, 5)


def test_budget_falls_back_to_full() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    plan = compiler.compile_strict(state(), budget=Budget(max_candidates=2))
    assert plan.guarantee.kind is GuaranteeKind.EXACT_VALID_SET
    assert plan.candidates.ids == (1, 3, 5)
    assert plan.execution.kind == "full"
    assert plan.source == "fallback"
    assert plan.cost.dynamic_bytes == 8 * 4 * 2


def test_budget_can_forbid_fallback() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    with pytest.raises(PlanBudgetError):
        compiler.compile_strict(
            state(),
            budget=Budget(max_candidates=2, allow_full_fallback=False),
        )


def test_compiler_plan_cache_hits_by_state_and_policy() -> None:
    from draftslice.cache import LRUPlanCache

    cache = LRUPlanCache()
    compiler = PlanCompiler(vocab_size=8, hidden_size=4, plan_cache=cache)
    first = compiler.compile_strict(state())
    second = compiler.compile_strict(state())
    assert first is second
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1

    fallback = compiler.compile_strict(state(), budget=Budget(max_candidates=2))
    assert fallback.source == "fallback"
    assert fallback is not first
    assert cache.stats.misses == 2


def test_empty_valid_set() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    plan = compiler.compile_strict(state(ids=()))
    assert plan.candidates.is_empty
    assert plan.execution.kind == "slice_gemv"
    assert plan.cost.total_bytes == 0


def test_full_policy() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    plan = compiler.compile_strict(state(), policy=PlanPolicy(mode="full"))
    assert plan.guarantee.kind is GuaranteeKind.FULL
    assert plan.execution.kind == "full"
    assert plan.candidates.ids == tuple(range(8))


def test_compile_step_unions_rows() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_step([state(0, (1, 2)), state(1, (2, 3)), state(2, (7,))])
    assert step.size == 3
    assert step.union_rows.ids == (1, 2, 3, 7)
    assert step.kernel == "slice_gemm"
    assert step.cost.macs == 4 * 4
    assert step.covers([3])
    assert step.plan_id.startswith("sha256:")


def test_compile_step_with_fallback_uses_full() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_step(
        [state(0, (1, 2)), state(1, (2, 3))],
        budget=Budget(max_candidates=1),
    )
    assert step.kernel == "full"
    assert all(node.plan.execution.kind == "full" for node in step.nodes)


def test_out_of_vocab_token_is_rejected() -> None:
    compiler = PlanCompiler(vocab_size=4, hidden_size=4)
    with pytest.raises(ValueError):
        compiler.compile_strict(state(ids=(1, 4)))


def test_calibrated_mode_needs_explicit_candidates() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    with pytest.raises(ValueError):
        compiler.compile_strict(state(), policy=PlanPolicy(mode="calibrated"))
