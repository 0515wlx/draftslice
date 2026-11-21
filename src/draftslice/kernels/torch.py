"""PyTorch kernels.

These are the first non-reference backends and the baseline for the fused Triton
kernel. The module imports torch lazily so the core package stays usable without
it.
"""

from __future__ import annotations

from typing import Any

from ..plan.types import HeadPlan, TokenSet
from .base import KernelError, KernelResult

try:  # pragma: no cover - exercised only when torch is installed
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise KernelError("torch is not installed")
    return torch


def _as_rows_tensor(rows: TokenSet | Any, device: Any) -> Any:
    t = _require_torch()
    if isinstance(rows, TokenSet):
        return t.tensor(rows.ids, dtype=t.long, device=device)
    return rows.to(device=device, dtype=t.long)


def _validate(hidden: Any, weight: Any) -> None:
    t = _require_torch()
    if not isinstance(hidden, t.Tensor) or not isinstance(weight, t.Tensor):
        raise KernelError("torch kernels require torch.Tensor inputs")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise KernelError("hidden and weight must be 2-D tensors")
    if hidden.shape[1] != weight.shape[1]:
        raise KernelError(f"hidden width {hidden.shape[1]} != weight width {weight.shape[1]}")


class _GraphMixin:
    """Optional CUDA-graph replay shared by the torch baselines."""

    _graphs: Any

    def _init_graphs(self, use_cuda_graph: bool, graph_cache: Any | None, max_entries: int) -> None:
        self._graphs = graph_cache
        if self._graphs is None and use_cuda_graph:
            from .graph import CudaGraphCache

            self._graphs = CudaGraphCache(max_entries=max_entries)

    @property
    def graph_cache(self) -> Any | None:
        return self._graphs

    def _run_graph(
        self,
        key: Any,
        fn: Any,
        hidden: Any,
        rows: Any | None = None,
    ) -> Any:
        if self._graphs is None:
            return fn(hidden, rows)
        return self._graphs.run(key, fn, hidden, rows)


class TorchGatherKernel(_GraphMixin):
    """Materialize gathered rows, then run a dense GEMM.

    This is intentionally the naive baseline: it measures ``W[rows]`` traffic and
    the dense matmul separately, which is what a fused kernel should beat.
    """

    name = "torch_gather"

    def __init__(
        self,
        *,
        use_cuda_graph: bool = False,
        graph_cache: Any | None = None,
        max_graph_entries: int = 32,
    ) -> None:
        self._init_graphs(use_cuda_graph, graph_cache, max_graph_entries)

    def supports(self, plan: HeadPlan, batch: int) -> bool:
        return torch is not None and plan.execution.kind != "full" and batch >= 0

    def run(
        self,
        hidden: Any,
        plan: HeadPlan,
        weight: Any,
    ) -> KernelResult:
        if plan.execution.rows is None:
            raise KernelError("torch gather kernel requires slice execution")
        logits = self.run_tensor(hidden, weight, plan.execution.rows)
        return KernelResult(rows=plan.execution.rows, logits=logits, backend=self.name)

    def run_tensor(self, hidden: Any, weight: Any, rows: TokenSet | Any) -> Any:
        t = _require_torch()
        _validate(hidden, weight)
        row_index = _as_rows_tensor(rows, hidden.device)
        if row_index.numel() == 0:
            return t.empty((hidden.shape[0], 0), dtype=hidden.dtype, device=hidden.device)
        key = (
            "torch_gather",
            weight.data_ptr(),
            int(weight.shape[0]),
            int(hidden.shape[0]),
            int(hidden.shape[1]),
            str(hidden.dtype),
            str(hidden.device),
            int(row_index.numel()),
        )
        return self._run_graph(
            key,
            lambda static_hidden, static_rows: self._gather(static_hidden, weight, static_rows),
            hidden,
            row_index,
        )

    @staticmethod
    def _gather(hidden: Any, weight: Any, row_index: Any) -> Any:
        with torch.no_grad():
            return hidden @ weight.index_select(0, row_index).T


class TorchFullKernel(_GraphMixin):
    """Dense full-head baseline."""

    name = "torch_full"

    def __init__(
        self,
        *,
        use_cuda_graph: bool = False,
        graph_cache: Any | None = None,
        max_graph_entries: int = 32,
    ) -> None:
        self._init_graphs(use_cuda_graph, graph_cache, max_graph_entries)

    def supports(self, plan: HeadPlan, batch: int) -> bool:
        return torch is not None and plan.execution.kind == "full" and batch >= 0

    def run(self, hidden: Any, plan: HeadPlan, weight: Any) -> KernelResult:
        _validate(hidden, weight)
        rows = TokenSet(range(weight.shape[0]))
        return KernelResult(rows=rows, logits=self.run_tensor(hidden, weight), backend=self.name)

    def run_tensor(self, hidden: Any, weight: Any) -> Any:
        _require_torch()
        _validate(hidden, weight)
        key = (
            "torch_full",
            weight.data_ptr(),
            int(weight.shape[0]),
            int(hidden.shape[0]),
            int(hidden.shape[1]),
            str(hidden.dtype),
            str(hidden.device),
        )
        return self._run_graph(
            key,
            lambda static_hidden, _static_rows: self._full(static_hidden, weight),
            hidden,
        )

    @staticmethod
    def _full(hidden: Any, weight: Any) -> Any:
        with torch.no_grad():
            return hidden @ weight.T


__all__ = ["TorchFullKernel", "TorchGatherKernel"]
