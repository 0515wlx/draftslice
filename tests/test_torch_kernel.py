from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.kernels.torch import TorchFullKernel, TorchGatherKernel  # noqa: E402
from draftslice.plan import Execution, Guarantee, HeadPlan, TokenSet  # noqa: E402


def plan_for(ids, *, full: bool = False) -> HeadPlan:
    tokens = TokenSet(ids)
    return HeadPlan(
        candidates=tokens,
        execution=Execution(kind="full") if full else Execution(kind="slice_gemv", rows=tokens),
        guarantee=Guarantee.full() if full else Guarantee.exact_valid_set(),
        state_id=None,
    )


def test_gather_matches_manual_index_select() -> None:
    torch.manual_seed(0)
    weight = torch.randn(16, 8, dtype=torch.float32)
    hidden = torch.randn(4, 8, dtype=torch.float32)
    plan = plan_for([5, 1, 7, 1])
    result = TorchGatherKernel().run(hidden, plan, weight)
    expected = hidden @ weight[list(plan.execution.rows.ids)].T
    torch.testing.assert_close(result.logits, expected)
    assert result.rows.ids == (1, 5, 7)
    assert result.backend == "torch_gather"


def test_gather_empty_rows() -> None:
    weight = torch.randn(8, 4)
    hidden = torch.randn(3, 4)
    plan = plan_for([])
    result = TorchGatherKernel().run(hidden, plan, weight)
    assert tuple(result.logits.shape) == (3, 0)


def test_gather_candidate_projection() -> None:
    weight = torch.randn(8, 4)
    hidden = torch.randn(2, 4)
    rows = TokenSet([1, 2, 5, 7])
    plan = HeadPlan(
        candidates=TokenSet([2, 7]),
        execution=Execution(kind="slice_gemv", rows=rows),
        guarantee=Guarantee.exact_valid_set(),
        state_id=None,
    )
    result = TorchGatherKernel().run(hidden, plan, weight)
    expected = hidden @ weight[[2, 7]].T
    torch.testing.assert_close(result.candidate_logits(plan), expected)


def test_full_kernel_matches_dense() -> None:
    weight = torch.randn(12, 6)
    hidden = torch.randn(5, 6)
    plan = plan_for(range(12), full=True)
    result = TorchFullKernel().run(hidden, plan, weight)
    torch.testing.assert_close(result.logits, hidden @ weight.T)
    assert result.rows.ids == tuple(range(12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_cuda_gather_tolerance() -> None:
    torch.manual_seed(1)
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    plan = plan_for([3, 9, 17, 63])
    result = TorchGatherKernel().run(hidden, plan, weight)
    expected = hidden @ weight[list(plan.execution.rows.ids)].T
    torch.testing.assert_close(result.logits, expected, atol=0.0, rtol=0.0)
