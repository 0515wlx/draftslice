from __future__ import annotations

import pytest

from draftslice.constraints import ConstraintState
from draftslice.plan import GuaranteeKind, TokenSet
from draftslice.plan.compiler import Budget, PlanBudgetError, PlanCompiler, PlanPolicy


def state(state_id: int = 0, ids=(1, 3, 5)) -> ConstraintState:
    return ConstraintState(state_id=state_id, valid_ids=TokenSet(ids))


class _Hidden:
    pass


class FakeScreener:
    name = "fake"

    def __init__(self, rows: list[TokenSet]) -> None:
        self.rows = rows
        self.calls: list[object] = []

    def candidates(self, hidden_states: object) -> list[TokenSet]:
        self.calls.append(hidden_states)
        return self.rows


def test_constrained_intersection_and_guarantee() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_screened(
        [state()],
        [TokenSet((1, 9))],
        delta=0.05,
        calibration_id="cal-1",
    )
    plan = step.nodes[0].plan
    assert plan.candidates.ids == (1,)
    assert plan.guarantee.kind is GuaranteeKind.CALIBRATED
    assert plan.guarantee.universe == "valid"
    assert plan.guarantee.delta == 0.05
    assert plan.guarantee.calibration_id == "cal-1"
    assert plan.execution.kind == "slice_gemv"
    assert plan.execution.covers(plan.candidates)
    assert plan.source == "screener"


def test_freeform_keeps_candidates() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_screened(
        [state(ids=(1, 3, 5))],
        [TokenSet((1, 7))],
        delta=0.1,
        universe="vocab",
    )
    plan = step.nodes[0].plan
    assert plan.candidates.ids == (1, 7)
    assert plan.guarantee.universe == "vocab"
    assert plan.guarantee.kind is GuaranteeKind.CALIBRATED


def test_empty_intersection_falls_back_to_full() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_screened([state()], [TokenSet((9,))], delta=0.05)
    plan = step.nodes[0].plan
    assert plan.execution.kind == "full"
    assert plan.guarantee.kind is GuaranteeKind.FULL
    assert step.kernel == "full"


def test_empty_intersection_can_forbid_fallback() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    with pytest.raises(PlanBudgetError):
        compiler.compile_screened(
            [state()],
            [TokenSet((9,))],
            delta=0.05,
            budget=Budget(allow_full_fallback=False),
        )


def test_screened_budget_falls_back_to_full() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_screened(
        [state()],
        [TokenSet((1, 3, 5))],
        delta=0.05,
        budget=Budget(max_candidates=2),
    )
    plan = step.nodes[0].plan
    assert plan.execution.kind == "full"
    assert plan.source == "fallback"
    assert plan.guarantee.kind is GuaranteeKind.CALIBRATED


def test_compile_step_requires_calibrated_mode_for_screener() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    screener = FakeScreener([TokenSet((1,))])
    with pytest.raises(ValueError):
        compiler.compile_step(
            [state()],
            screener=screener,
            hidden_states=_Hidden(),
            delta=0.05,
        )
    with pytest.raises(ValueError):
        compiler.compile_step(
            [state()],
            policy=PlanPolicy(mode="strict"),
            screener=screener,
            hidden_states=_Hidden(),
            delta=0.05,
        )


def test_compile_step_requires_delta_and_hidden_states() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    screener = FakeScreener([TokenSet((1,))])
    with pytest.raises(ValueError):
        compiler.compile_step(
            [state()], mode="calibrated", screener=screener, hidden_states=_Hidden()
        )
    with pytest.raises(ValueError):
        compiler.compile_step([state()], mode="calibrated", screener=screener, delta=0.05)


def test_compile_step_union_dedups_rows() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    screener = FakeScreener([TokenSet((1, 2, 9)), TokenSet((2, 3, 9))])
    step = compiler.compile_step(
        [state(0, (1, 2, 3)), state(1, (2, 3, 4))],
        mode="calibrated",
        screener=screener,
        hidden_states=_Hidden(),
        delta=0.05,
    )
    assert [node.plan.candidates.ids for node in step.nodes] == [(1, 2), (2, 3)]
    assert step.union_rows.ids == (1, 2, 3)
    assert step.kernel == "slice_gemm"
    assert step.cost.dynamic_bytes == 3 * 4 * 2


def test_compile_step_mode_override_full() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    step = compiler.compile_step([state()], mode="full")
    assert step.nodes[0].plan.guarantee.kind is GuaranteeKind.FULL
    assert step.nodes[0].plan.execution.kind == "full"


def test_compile_screened_length_mismatch() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    with pytest.raises(ValueError):
        compiler.compile_screened([state()], [], delta=0.05)
    with pytest.raises(ValueError):
        compiler.compile_step(
            [state()],
            mode="calibrated",
            screener=FakeScreener([TokenSet((1,)), TokenSet((2,))]),
            hidden_states=_Hidden(),
            delta=0.05,
        )


def test_compile_screened_rejects_bad_rows_and_universe() -> None:
    compiler = PlanCompiler(vocab_size=8, hidden_size=4)
    with pytest.raises(ValueError):
        compiler.compile_screened([state()], [TokenSet((1,))], delta=0.05, universe="bad")
    with pytest.raises(TypeError):
        compiler.compile_screened([state()], [("not-a-token-set",)], delta=0.05)  # type: ignore[list-item]


def test_compile_screened_plan_cache_keys_on_candidate_content() -> None:
    from draftslice.cache.plan_cache import LRUPlanCache

    cache = LRUPlanCache(max_bytes=1 << 20)
    compiler = PlanCompiler(vocab_size=8, hidden_size=4, plan_cache=cache)
    kwargs = dict(delta=0.05, calibration_id="cal-1")

    first = compiler.compile_screened([state()], [TokenSet((1, 3))], **kwargs)
    second = compiler.compile_screened([state()], [TokenSet((1, 3))], **kwargs)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert first.nodes[0].plan == second.nodes[0].plan

    # A different candidate set for the same state must not reuse the plan.
    third = compiler.compile_screened([state()], [TokenSet((1, 5))], **kwargs)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 2
    assert third.nodes[0].plan.candidates.ids == (1, 5)

    # A different delta is a different guarantee and must miss as well.
    compiler.compile_screened([state()], [TokenSet((1, 3))], delta=0.1, calibration_id="cal-1")
    assert cache.stats.hits == 1
    assert cache.stats.misses == 3
