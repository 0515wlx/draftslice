"""Fused Triton slice kernels.

Two eager shapes cover the draft-head slice:

* a row-block GEMV kernel for a single hidden state (tiny-M regime);
* a tiled tensor-core GEMM for a batched hidden state, including tree nodes.

The launch configuration comes from :mod:`draftslice.kernels.tuning`. Both
shapes can optionally be captured by :class:`~draftslice.kernels.graph.CudaGraphCache`;
graph replay removes the launch overhead that dominates small candidate sets.
"""

from __future__ import annotations

from typing import Any

from ..plan.types import HeadPlan, TokenSet
from .base import KernelError, KernelResult
from .tuning import KernelConfig, gemm_config, gemv_config

try:  # pragma: no cover - exercised only with triton installed
    import torch
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if tl is not None:

    @triton.jit
    def _slice_gemv_kernel(
        hidden_ptr,
        weight_ptr,
        rows_ptr,
        out_ptr,
        M,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        stride_hb,
        stride_hd,
        stride_wv,
        stride_wd,
        stride_ob,
        stride_om,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        tokens = tl.load(rows_ptr + offs_m, mask=mask_m, other=0)

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(0, D, BLOCK_D):
            offs_d = start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            weight = tl.load(
                weight_ptr + tokens[:, None] * stride_wv + offs_d[None, :] * stride_wd,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            )
            hidden = tl.load(
                hidden_ptr + offs_d * stride_hd,
                mask=mask_d,
                other=0.0,
            )
            acc += tl.sum(weight.to(tl.float32) * hidden.to(tl.float32)[None, :], axis=1)

        tl.store(
            out_ptr + offs_m * stride_om,
            acc,
            mask=mask_m,
        )

    @triton.jit
    def _slice_rows_kernel(
        hidden_ptr,
        weight_ptr,
        tokens_ptr,
        out_rows_ptr,
        out_ptr,
        W,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        stride_hb,
        stride_hd,
        stride_wv,
        stride_wd,
        stride_ob,
        scale,
        soft_cap,
        HAS_SOFT_CAP: tl.constexpr,
    ):
        """One program computes BLOCK_M candidate scores for one batch row.

        ``tokens`` is a padded ``[rows, W]`` int32 matrix; ``-1`` marks a
        padding column and is skipped on both the load and the store. This
        writes directly into the full-vocabulary output, so the caller does not
        need a separate scatter kernel. Padding the candidate width to a small
        bucket set keeps Triton's compiled-shape count bounded.
        """

        row = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < W
        tokens = tl.load(tokens_ptr + row * W + offs_m, mask=mask_m, other=-1)
        valid = tokens >= 0
        out_row = tl.load(out_rows_ptr + row).to(tl.int64)

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(0, D, BLOCK_D):
            offs_d = start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            weight = tl.load(
                weight_ptr
                + tokens[:, None] * stride_wv
                + offs_d[None, :] * stride_wd,
                mask=valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            hidden = tl.load(
                hidden_ptr + out_row * stride_hb + offs_d * stride_hd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight * hidden[None, :], axis=1)

        if HAS_SOFT_CAP:
            acc = tl.tanh(acc / soft_cap) * soft_cap
        if scale != 1.0:
            acc = acc * scale
        tl.store(out_ptr + out_row * stride_ob + tokens, acc, mask=valid)

    @triton.jit
    def _slice_gemm_kernel(
        hidden_ptr,
        weight_ptr,
        rows_ptr,
        out_ptr,
        M,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        stride_hb,
        stride_hd,
        stride_wv,
        stride_wd,
        stride_ob,
        stride_om,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        tokens = tl.load(rows_ptr + offs_m, mask=mask_m, other=0)
        offs_b = tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        acc = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)
        for start in range(0, D, BLOCK_D):
            offs_d = start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            weight = tl.load(
                weight_ptr + tokens[:, None] * stride_wv + offs_d[None, :] * stride_wd,
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
            )
            hidden = tl.load(
                hidden_ptr + offs_b[:, None] * stride_hb + offs_d[None, :] * stride_hd,
                mask=mask_b[:, None] & mask_d[None, :],
                other=0.0,
            )
            acc += tl.dot(weight, tl.trans(hidden))

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_b[None, :] * stride_ob,
            acc,
            mask=mask_m[:, None] & mask_b[None, :],
        )

    @triton.jit
    def _slice_dot_kernel(
        hidden_ptr,
        weight_ptr,
        rows_ptr,
        out_ptr,
        B: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        stride_hb,
        stride_hd,
        stride_wv,
        stride_wd,
        stride_ob,
        stride_om,
    ):
        """Float32 fallback: one program per row, all batch items accumulated."""

        row = tl.program_id(0)
        token = tl.load(rows_ptr + row)
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        weight = tl.load(
            weight_ptr + token * stride_wv + offsets * stride_wd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        for batch in tl.static_range(B):
            hidden = tl.load(
                hidden_ptr + batch * stride_hb + offsets * stride_hd,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc = tl.sum(hidden * weight, axis=0)
            tl.store(out_ptr + batch * stride_ob + row * stride_om, acc)


def _availability() -> tuple[bool, str]:
    if torch is None:
        return False, "torch is not installed"
    if triton is None:
        return False, "triton is not installed"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    return True, ""


class TritonSliceKernel:
    """Fused gather + dot/GEMM for slice executions."""

    name = "triton_slice"

    def __init__(
        self,
        *,
        use_cuda_graph: bool = False,
        graph_cache: Any | None = None,
        max_graph_entries: int = 32,
        row_cache_entries: int = 64,
        autotune: bool = False,
        autotune_warmup: int = 2,
        autotune_iterations: int = 5,
    ) -> None:
        self._row_cache: dict[tuple[int, int], tuple[TokenSet, Any]] = {}
        self._row_cache_entries = row_cache_entries
        self._graphs = graph_cache
        if self._graphs is None and use_cuda_graph:
            from .graph import CudaGraphCache

            self._graphs = CudaGraphCache(max_entries=max_graph_entries)
        self._autotuner = None
        if autotune:
            from .autotune import KernelAutotuner

            self._autotuner = KernelAutotuner(
                warmup=autotune_warmup,
                iterations=autotune_iterations,
            )
        self.last_config: KernelConfig | None = None

    @classmethod
    def is_available(cls) -> bool:
        return _availability()[0]

    @property
    def graph_cache(self) -> Any | None:
        return self._graphs

    @property
    def autotuner(self) -> Any | None:
        return self._autotuner

    def supports(self, plan: HeadPlan, batch: int) -> bool:
        available, _ = _availability()
        return available and plan.execution.kind != "full" and batch > 0

    def run(self, hidden: Any, plan: HeadPlan, weight: Any) -> KernelResult:
        if plan.execution.kind == "full":
            raise KernelError("triton slice kernel does not execute a full head")
        if plan.execution.rows is None:
            raise KernelError("slice execution has no rows")
        logits = self.run_tensor(hidden, weight, plan.execution.rows)
        return KernelResult(rows=plan.execution.rows, logits=logits, backend=self.name)

    def _row_index(self, rows: TokenSet | Any, device: Any) -> Any:
        if torch is None:
            raise KernelError("torch is not installed")
        if not isinstance(rows, TokenSet):
            return rows.to(device=device, dtype=torch.int32)
        key = (id(rows), len(rows))
        cached = self._row_cache.get(key)
        if cached is not None and cached[0] is rows:
            return cached[1]
        tensor = torch.tensor(rows.ids, dtype=torch.int32, device=device)
        self._row_cache[key] = (rows, tensor)
        while len(self._row_cache) > self._row_cache_entries:
            self._row_cache.pop(next(iter(self._row_cache)))
        return tensor

    def run_tensor(self, hidden: Any, weight: Any, rows: TokenSet | Any) -> Any:
        available, reason = _availability()
        if not available:
            raise KernelError(reason)
        assert torch is not None and triton is not None

        if not isinstance(hidden, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise KernelError("triton kernel requires torch.Tensor inputs")
        if hidden.ndim != 2 or weight.ndim != 2:
            raise KernelError("hidden and weight must be 2-D tensors")
        if hidden.shape[1] != weight.shape[1]:
            raise KernelError(f"hidden width {hidden.shape[1]} != weight width {weight.shape[1]}")
        if hidden.device.type != "cuda" or weight.device.type != "cuda":
            raise KernelError("triton kernel requires CUDA tensors")

        row_index = self._row_index(rows, hidden.device)
        batch, width = hidden.shape
        count = row_index.numel()

        if self._graphs is not None and count > 0:
            key = (
                "triton_slice",
                weight.data_ptr(),
                int(weight.shape[0]),
                int(hidden.shape[0]),
                int(hidden.shape[1]),
                str(hidden.dtype),
                str(hidden.device),
                int(count),
            )
            return self._graphs.run(
                key,
                lambda static_hidden, static_rows: self._launch(static_hidden, weight, static_rows),
                hidden,
                row_index,
            )
        return self._launch(hidden, weight, row_index)

    supports_rows_into = True

    def run_rows_into(
        self,
        hidden: Any,
        weight: Any,
        tokens: Any,
        out_rows: Any,
        out: Any,
        *,
        width: int | None = None,
        scale: float = 1.0,
        soft_cap: float | None = None,
    ) -> Any:
        """Write per-row sliced logits directly into a full-vocabulary output.

        ``tokens`` is a padded int32 matrix (``-1`` marks a padding column):
        either ``[rows, W]`` or a flat ``[rows * W]`` buffer with ``width=W``.
        ``out_rows`` maps each matrix row to its batch row, and ``out`` is
        ``[batch, vocab]``. Unlike :meth:`run_tensor` this fuses the gather, dot
        product, optional logit post-processing, and scatter into one launch,
        which removes the per-grammar-state launch that dominated the vLLM hot
        path.
        """

        if torch is None or triton is None:
            raise KernelError("torch and triton are required")
        if not isinstance(tokens, torch.Tensor) or not isinstance(out, torch.Tensor):
            raise KernelError("triton kernel requires torch.Tensor inputs")
        if hidden.ndim != 2 or weight.ndim != 2:
            raise KernelError("hidden and weight must be 2-D")
        if hidden.shape[1] != weight.shape[1]:
            raise KernelError(f"hidden width {hidden.shape[1]} != weight width {weight.shape[1]}")
        if tokens.ndim == 2:
            rows, width = int(tokens.shape[0]), int(tokens.shape[1])
        else:
            if tokens.ndim != 1 or width is None:
                raise KernelError("flat tokens require an explicit width")
            rows, width = int(out_rows.numel()), int(width)
            if tokens.numel() != rows * width:
                raise KernelError("flat tokens size does not match rows * width")
        if out_rows.numel() != rows:
            raise KernelError("tokens rows must match out_rows")
        if out.ndim != 2 or out.shape[0] < hidden.shape[0]:
            raise KernelError("out must be [batch, vocab]")
        if out.shape[1] < weight.shape[0]:
            raise KernelError("out is narrower than the head weight")
        if hidden.device.type != "cuda" or weight.device.type != "cuda":
            raise KernelError("triton kernel requires CUDA tensors")
        if tokens.numel() == 0:
            return out
        if tokens.device != hidden.device:
            tokens = tokens.to(device=hidden.device, dtype=torch.int32)
        if out_rows.device != hidden.device:
            out_rows = out_rows.to(device=hidden.device, dtype=torch.int32)
        if not out.is_contiguous():
            raise KernelError("out must be contiguous")

        config = gemv_config(width, int(hidden.shape[1]), element_bytes=hidden.element_size())
        _slice_rows_kernel[(rows, triton.cdiv(width, config.block_m))](
            hidden,
            weight,
            tokens,
            out_rows,
            out,
            width,
            D=int(hidden.shape[1]),
            BLOCK_M=config.block_m,
            BLOCK_D=config.block_d,
            stride_hb=hidden.stride(0),
            stride_hd=hidden.stride(1),
            stride_wv=weight.stride(0),
            stride_wd=weight.stride(1),
            stride_ob=out.stride(0),
            scale=float(scale),
            soft_cap=0.0 if soft_cap is None else float(soft_cap),
            HAS_SOFT_CAP=soft_cap is not None,
            num_warps=config.num_warps,
        )
        self.last_config = config
        return out

    def _launch(self, hidden: Any, weight: Any, row_index: Any) -> Any:
        assert torch is not None and triton is not None
        batch, width = hidden.shape
        count = row_index.numel()
        out = torch.empty((batch, count), dtype=torch.float32, device=hidden.device)
        if count == 0:
            return out

        element_bytes = hidden.element_size()
        if batch == 1:
            if self._autotuner is not None:
                from .autotune import gemv_candidates

                key = (
                    "gemv",
                    count,
                    width,
                    str(hidden.dtype),
                    str(hidden.device),
                )
                config = self._autotuner.choose(
                    key,
                    gemv_candidates(count, width, element_bytes=element_bytes),
                    lambda config: self._run_gemv(hidden, weight, row_index, out, config),
                )
            else:
                config = gemv_config(count, width, element_bytes=element_bytes)
            return self._run_gemv(hidden, weight, row_index, out, config)

        if hidden.dtype == torch.float32:
            block_d = triton.next_power_of_2(int(width))
            config = KernelConfig(
                block_m=1,
                block_d=block_d,
                block_b=triton.next_power_of_2(batch),
                num_warps=8 if block_d >= 2048 else 4,
                num_stages=1,
            )
            self.last_config = config
            _slice_dot_kernel[(count,)](
                hidden,
                weight,
                row_index,
                out,
                B=batch,
                D=width,
                BLOCK_D=block_d,
                stride_hb=hidden.stride(0),
                stride_hd=hidden.stride(1),
                stride_wv=weight.stride(0),
                stride_wd=weight.stride(1),
                stride_ob=out.stride(0),
                stride_om=out.stride(1),
                num_warps=config.num_warps,
            )
            return out

        if self._autotuner is not None:
            from .autotune import gemm_candidates

            key = (
                "gemm",
                count,
                batch,
                width,
                str(hidden.dtype),
                str(hidden.device),
            )
            config = self._autotuner.choose(
                key,
                gemm_candidates(count, batch, width, element_bytes=element_bytes),
                lambda config: self._run_gemm(hidden, weight, row_index, out, config),
            )
        else:
            config = gemm_config(count, batch, width, element_bytes=element_bytes)
        return self._run_gemm(hidden, weight, row_index, out, config)

    def _run_gemv(
        self,
        hidden: Any,
        weight: Any,
        row_index: Any,
        out: Any,
        config: KernelConfig,
    ) -> Any:
        assert triton is not None
        count = int(row_index.numel())
        width = int(hidden.shape[1])
        _slice_gemv_kernel[(triton.cdiv(count, config.block_m),)](
            hidden,
            weight,
            row_index,
            out,
            count,
            D=width,
            BLOCK_M=config.block_m,
            BLOCK_D=config.block_d,
            stride_hb=hidden.stride(0),
            stride_hd=hidden.stride(1),
            stride_wv=weight.stride(0),
            stride_wd=weight.stride(1),
            stride_ob=out.stride(0),
            stride_om=out.stride(1),
            num_warps=config.num_warps,
        )
        self.last_config = config
        return out

    def _run_gemm(
        self,
        hidden: Any,
        weight: Any,
        row_index: Any,
        out: Any,
        config: KernelConfig,
    ) -> Any:
        assert triton is not None
        batch, width = hidden.shape
        count = int(row_index.numel())
        for attempt in range(4):
            try:
                _slice_gemm_kernel[(triton.cdiv(count, config.block_m),)](
                    hidden,
                    weight,
                    row_index,
                    out,
                    count,
                    B=batch,
                    D=width,
                    BLOCK_M=config.block_m,
                    BLOCK_B=config.block_b,
                    BLOCK_D=config.block_d,
                    stride_hb=hidden.stride(0),
                    stride_hd=hidden.stride(1),
                    stride_wv=weight.stride(0),
                    stride_wd=weight.stride(1),
                    stride_ob=out.stride(0),
                    stride_om=out.stride(1),
                    num_warps=config.num_warps,
                    num_stages=config.num_stages,
                )
                self.last_config = config
                return out
            except Exception as exc:
                # Devices with a smaller shared-memory budget may reject a static
                # configuration even though the tuned table targets A100/H100.
                if type(exc).__name__ != "OutOfResources" or attempt == 3:
                    raise
                config = KernelConfig(
                    block_m=max(16, config.block_m // 2),
                    block_d=max(64, config.block_d // 2),
                    block_b=config.block_b,
                    num_warps=config.num_warps,
                    num_stages=max(1, config.num_stages - 1),
                )
        return out


__all__ = ["TritonSliceKernel"]
