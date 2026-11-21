from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from draftslice.kernels.graph import CudaGraphCache, GraphError  # noqa: E402
from draftslice.kernels.torch import TorchGatherKernel  # noqa: E402
from draftslice.kernels.triton import TritonSliceKernel  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for CUDA graphs"
)


def _reference(hidden, weight, rows):
    return hidden.float() @ weight[rows.long()].float().T


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("rows", [1, 17, 128])
def test_triton_graph_matches_eager(batch: int, rows: int) -> None:
    torch.manual_seed(0)
    weight = torch.randn(256, 64, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(batch, 64, dtype=torch.bfloat16, device="cuda")
    index = torch.randperm(256, device="cuda")[:rows].to(torch.int32)

    eager = TritonSliceKernel().run_tensor(hidden, weight, index)
    graphed = TritonSliceKernel(use_cuda_graph=True)
    first = graphed.run_tensor(hidden, weight, index)
    second = graphed.run_tensor(hidden, weight, index)

    for value in (first, second):
        torch.testing.assert_close(value, eager, atol=0.0, rtol=0.0)
    assert graphed.graph_cache is not None
    assert graphed.graph_cache.captures == 1
    assert graphed.graph_cache.replays == 2


def test_graph_refreshes_changed_rows_and_hidden() -> None:
    torch.manual_seed(1)
    weight = torch.randn(128, 32, dtype=torch.bfloat16, device="cuda")
    kernel = TritonSliceKernel(use_cuda_graph=True)
    index_a = torch.tensor([1, 5, 9, 17], dtype=torch.int32, device="cuda")
    index_b = torch.tensor([2, 6, 10, 18], dtype=torch.int32, device="cuda")
    hidden_a = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")
    hidden_b = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")

    first = kernel.run_tensor(hidden_a, weight, index_a).clone()
    second = kernel.run_tensor(hidden_b, weight, index_b).clone()

    torch.testing.assert_close(first, _reference(hidden_a, weight, index_a), atol=0.5, rtol=0.02)
    torch.testing.assert_close(second, _reference(hidden_b, weight, index_b), atol=0.5, rtol=0.02)
    assert kernel.graph_cache is not None
    assert kernel.graph_cache.captures == 1


def test_gather_graph_matches_eager() -> None:
    torch.manual_seed(2)
    weight = torch.randn(256, 64, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    index = torch.randperm(256, device="cuda")[:31]

    eager = TorchGatherKernel().run_tensor(hidden, weight, index)
    graphed = TorchGatherKernel(use_cuda_graph=True)
    graphed.run_tensor(hidden, weight, index)
    replayed = graphed.run_tensor(hidden, weight, index)
    torch.testing.assert_close(replayed, eager, atol=0.0, rtol=0.0)


def test_graph_refreshes_in_place_mutation() -> None:
    torch.manual_seed(4)
    weight = torch.randn(128, 32, dtype=torch.bfloat16, device="cuda")
    kernel = TritonSliceKernel(use_cuda_graph=True)
    index = torch.tensor([1, 5, 9, 17], dtype=torch.int32, device="cuda")
    hidden = torch.randn(1, 32, dtype=torch.bfloat16, device="cuda")

    first = kernel.run_tensor(hidden, weight, index).clone()
    index.copy_(torch.tensor([2, 6, 10, 18], dtype=torch.int32, device="cuda"))
    hidden.copy_(torch.randn(1, 32, dtype=torch.bfloat16, device="cuda"))
    second = kernel.run_tensor(hidden, weight, index).clone()

    torch.testing.assert_close(second, _reference(hidden, weight, index), atol=0.5, rtol=0.02)
    assert kernel.graph_cache is not None
    assert kernel.graph_cache.captures == 1
    assert first.shape == second.shape


def test_autotuned_kernel_matches_eager() -> None:
    torch.manual_seed(5)
    weight = torch.randn(512, 64, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    index = torch.randperm(512, device="cuda")[:257].to(torch.int32)

    eager = TritonSliceKernel().run_tensor(hidden, weight, index)
    tuned = TritonSliceKernel(autotune=True, autotune_iterations=2)
    actual = tuned.run_tensor(hidden, weight, index)
    torch.testing.assert_close(actual, eager, atol=0.5, rtol=0.02)
    assert tuned.autotuner is not None
    assert tuned.autotuner.trials > 0


def test_graph_cache_rejects_mismatched_shapes() -> None:
    torch.manual_seed(3)
    cache = CudaGraphCache(max_entries=1)
    weight = torch.randn(64, 16, device="cuda")
    hidden_a = torch.randn(1, 16, device="cuda")
    hidden_b = torch.randn(2, 16, device="cuda")

    def fn(hidden, _rows):
        return hidden @ weight.T

    key = ("demo", weight.data_ptr(), 1, 16)
    cache.run(key, fn, hidden_a)
    with pytest.raises(GraphError):
        cache.run(key, fn, hidden_b)
