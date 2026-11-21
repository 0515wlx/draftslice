from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from draftslice.kernels.base import KernelError  # noqa: E402
from draftslice.kernels.triton import TritonSliceKernel  # noqa: E402
from draftslice.plan import Execution, Guarantee, HeadPlan, TokenSet  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the Triton kernel"
)


def plan_for(ids, *, candidates=None) -> HeadPlan:
    rows = TokenSet(ids)
    return HeadPlan(
        candidates=TokenSet(candidates) if candidates is not None else rows,
        execution=Execution(kind="slice_gemv", rows=rows),
        guarantee=Guarantee.exact_valid_set(),
        state_id=None,
    )


def reference(hidden, weight, ids):
    return hidden.float() @ weight[list(TokenSet(ids).ids)].float().T


@pytest.mark.parametrize("batch", [1, 4])
def test_float32_matches_torch(batch: int) -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 64, dtype=torch.float32, device="cuda")
    hidden = torch.randn(batch, 64, dtype=torch.float32, device="cuda")
    ids = [7, 1, 19, 5]
    plan = plan_for(ids)
    result = TritonSliceKernel().run(hidden, plan, weight)
    expected = reference(hidden, weight, ids)
    torch.testing.assert_close(result.logits, expected, atol=1e-4, rtol=1e-4)
    assert tuple(result.logits.shape) == (batch, len(ids))


def test_non_power_of_two_width_masks_correctly() -> None:
    torch.manual_seed(1)
    weight = torch.randn(16, 257, dtype=torch.float32, device="cuda")
    hidden = torch.randn(3, 257, dtype=torch.float32, device="cuda")
    ids = [2, 11, 15]
    result = TritonSliceKernel().run(hidden, plan_for(ids), weight)
    torch.testing.assert_close(result.logits, reference(hidden, weight, ids), atol=1e-4, rtol=1e-4)


def test_bf16_matches_torch_tolerance() -> None:
    torch.manual_seed(2)
    weight = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    ids = [3, 9, 17, 63]
    result = TritonSliceKernel().run(hidden, plan_for(ids), weight)
    expected = reference(hidden, weight, ids)
    torch.testing.assert_close(result.logits, expected, atol=0.5, rtol=0.02)


@pytest.mark.parametrize("batch", [1, 8, 32, 64])
def test_bf16_batch_shapes(batch: int) -> None:
    torch.manual_seed(10 + batch)
    weight = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(batch, 128, dtype=torch.bfloat16, device="cuda")
    ids = [1, 7, 9, 33, 63]
    result = TritonSliceKernel().run(hidden, plan_for(ids), weight)
    torch.testing.assert_close(
        result.logits,
        reference(hidden, weight, ids),
        atol=0.5,
        rtol=0.02,
    )


@pytest.mark.parametrize("batch", [1, 8])
def test_run_rows_into_scatters_and_masks_padding(batch: int) -> None:
    torch.manual_seed(4 + batch)
    weight = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(batch, 64, dtype=torch.bfloat16, device="cuda")
    rows = [[1, 7, 9], [3, 4, -1], [11, 12, 13, 14, -1], [2, -1, -1]][:batch]
    width = 5
    tokens = torch.full((batch, width), -1, dtype=torch.int32, device="cuda")
    for position, ids in enumerate(rows):
        tokens[position, : len(ids)] = torch.tensor(ids, dtype=torch.int32, device="cuda")
    out_rows = torch.arange(batch, dtype=torch.int32, device="cuda")
    out = torch.full((batch, 32), float("-inf"), dtype=torch.float32, device="cuda")
    TritonSliceKernel().run_rows_into(hidden, weight, tokens, out_rows, out)
    flat_out = torch.full((batch, 32), float("-inf"), dtype=torch.float32, device="cuda")
    TritonSliceKernel().run_rows_into(
        hidden, weight, tokens.reshape(-1), out_rows, flat_out, width=width
    )
    torch.testing.assert_close(flat_out, out)
    for position, ids in enumerate(rows):
        valid = [token for token in ids if token >= 0]
        expected = hidden[position].float() @ weight[valid].float().T
        torch.testing.assert_close(out[position, valid], expected, atol=0.5, rtol=0.02)
    masked = torch.isneginf(out)
    allowed = torch.zeros_like(masked)
    for position, ids in enumerate(rows):
        allowed[position, [token for token in ids if token >= 0]] = True
    assert bool(masked[~allowed].all())


def test_empty_rows() -> None:
    weight = torch.randn(8, 32, device="cuda")
    hidden = torch.randn(2, 32, device="cuda")
    result = TritonSliceKernel().run(hidden, plan_for([]), weight)
    assert tuple(result.logits.shape) == (2, 0)


def test_candidate_projection_with_tensor_result() -> None:
    torch.manual_seed(3)
    weight = torch.randn(16, 32, device="cuda")
    hidden = torch.randn(2, 32, device="cuda")
    plan = plan_for([1, 2, 5, 7], candidates=[2, 7])
    result = TritonSliceKernel().run(hidden, plan, weight)
    torch.testing.assert_close(
        result.candidate_logits(plan),
        hidden @ weight[[2, 7]].T,
        atol=1e-4,
        rtol=1e-4,
    )


def test_full_execution_is_rejected() -> None:
    plan = HeadPlan(
        candidates=TokenSet([0]),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        state_id=None,
    )
    with pytest.raises(KernelError):
        TritonSliceKernel().run(
            torch.randn(1, 8, device="cuda"),
            plan,
            torch.randn(4, 8, device="cuda"),
        )
