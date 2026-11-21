from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.kernels.base import KernelError  # noqa: E402
from draftslice.kernels.torch import TorchFullKernel, TorchGatherKernel  # noqa: E402
from draftslice.plan import (  # noqa: E402
    Execution,
    Guarantee,
    HeadPlan,
    NodePlan,
    StepPlan,
    TokenSet,
)
from draftslice.runtime.union import UnionExecutor  # noqa: E402


def plan(rows, candidates=None) -> HeadPlan:
    rows = TokenSet(rows)
    return HeadPlan(
        candidates=TokenSet(candidates) if candidates is not None else rows,
        execution=Execution(kind="slice_gemv", rows=rows),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )


def full_plan(candidates) -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet(candidates),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        state_id=None,
    )


def test_union_deduplicates_and_scatters() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 16)
    hidden = torch.randn(3, 16)
    plans = [plan([1, 2, 9]), plan([2, 9, 17]), plan([17, 25])]
    result = UnionExecutor(TorchGatherKernel()).execute(hidden, weight, plans)

    assert result.union_rows.ids == (1, 2, 9, 17, 25)
    assert result.logits.shape == (3, 5)
    for row, node_plan in enumerate(plans):
        expected = hidden[row : row + 1] @ weight[list(node_plan.candidates.ids)].T
        torch.testing.assert_close(result.node_logits[row], expected)


def test_union_handles_non_candidate_execution_rows() -> None:
    torch.manual_seed(1)
    weight = torch.randn(32, 8)
    hidden = torch.randn(2, 8)
    plans = [plan([1, 2, 9], candidates=[2, 9]), plan([9, 17], candidates=[17])]
    result = UnionExecutor(TorchGatherKernel()).execute(hidden, weight, plans)
    assert result.union_rows.ids == (1, 2, 9, 17)
    assert tuple(result.node_logits[0].shape) == (1, 2)
    assert tuple(result.node_logits[1].shape) == (1, 1)


def test_union_full_fallback() -> None:
    torch.manual_seed(2)
    weight = torch.randn(16, 8)
    hidden = torch.randn(2, 8)
    plans = [plan([1, 2]), full_plan([3, 4])]
    executor = UnionExecutor(TorchGatherKernel(), full_kernel=TorchFullKernel())
    result = executor.execute(hidden, weight, plans)
    assert result.union_rows.ids == tuple(range(16))
    assert result.logits.shape == (2, 16)
    torch.testing.assert_close(result.node_logits[1], hidden[1:2] @ weight[[3, 4]].T)


def test_union_full_without_fallback_raises() -> None:
    weight = torch.randn(16, 8)
    hidden = torch.randn(1, 8)
    with pytest.raises(KernelError):
        UnionExecutor(TorchGatherKernel()).execute(hidden, weight, [full_plan([1])])


def test_step_plan_union_covers_nodes() -> None:
    torch.manual_seed(3)
    weight = torch.randn(32, 8)
    hidden = torch.randn(2, 8)
    plans = [plan([1, 2, 3]), plan([3, 4, 5])]
    step = StepPlan(
        nodes=(NodePlan(0, plans[0]), NodePlan(1, plans[1])),
        union_rows=TokenSet([1, 2, 3, 4, 5]),
        kernel="slice_gemm",
    )
    result = UnionExecutor(TorchGatherKernel()).execute_step(hidden, weight, step)
    assert result.union_rows.ids == (1, 2, 3, 4, 5)


def test_step_plan_rejects_missing_union_rows() -> None:
    weight = torch.randn(16, 8)
    hidden = torch.randn(1, 8)
    plan_a = plan([1, 2])
    step = StepPlan(
        nodes=(NodePlan(0, plan_a),),
        union_rows=TokenSet([1, 2, 3]),
        kernel="slice_gemm",
    )
    with pytest.raises(KernelError):
        UnionExecutor(TorchGatherKernel()).execute_step(hidden, weight, step)
