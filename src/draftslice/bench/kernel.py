"""Kernel performance runner.

The runner measures the full-head and naive-gather baselines next to the fused
Triton slice kernels on the same shapes. Eager and CUDA-graph-replay timings are
reported for every implementation, so a speedup is never an artefact of giving
only one side graph capture.

The runner never locks clocks itself; callers must pass a locked
:class:`~draftslice.bench.gpu.GpuClockController` snapshot, which keeps the
control flow explicit and makes an unlocked run visibly different in the result.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gpu import GpuSnapshot, query_gpu

try:  # pragma: no cover - exercised when torch is installed
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised when triton is installed
    import triton
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]

from ..kernels.graph import CudaGraphCache
from ..kernels.torch import TorchFullKernel, TorchGatherKernel
from ..kernels.triton import TritonSliceKernel

KERNEL_NAMES = ("full", "torch_gather", "triton_slice")


class KernelBenchError(RuntimeError):
    """Raised when a kernel benchmark cannot run."""


@dataclass(frozen=True, slots=True)
class KernelBenchConfig:
    vocab_size: int = 248_320
    hidden_size: int = 2_560
    batch_sizes: tuple[int, ...] = (1, 8)
    row_counts: tuple[int, ...] = (128, 1_024, 4_096, 8_192, 16_384)
    dtype: str = "bf16"
    warmup: int = 10
    iterations: int = 30
    seed: int = 0
    device: str = "cuda:0"
    kernels: tuple[str, ...] = KERNEL_NAMES
    cuda_graph: bool = True
    graph_rows_limit: int = 65_536

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if not self.batch_sizes or any(batch <= 0 for batch in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")
        if not self.row_counts or any(rows <= 0 for rows in self.row_counts):
            raise ValueError("row_counts must be positive")
        if any(rows > self.vocab_size for rows in self.row_counts):
            raise ValueError("row_counts must not exceed vocab_size")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be bf16, fp16, or fp32")
        unknown = set(self.kernels) - set(KERNEL_NAMES)
        if unknown:
            raise ValueError(f"unknown kernels: {', '.join(sorted(unknown))}")
        if not self.kernels:
            raise ValueError("kernels must not be empty")
        if self.graph_rows_limit <= 0:
            raise ValueError("graph_rows_limit must be positive")


def _torch_dtype(name: str) -> Any:
    if torch is None:
        raise KernelBenchError("torch is not installed")
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _require_torch_cuda() -> None:
    if torch is None:
        raise KernelBenchError("torch is not installed")
    if not torch.cuda.is_available():
        raise KernelBenchError("CUDA is not available")


def _dtype_bytes(name: str) -> int:
    return {"bf16": 2, "fp16": 2, "fp32": 4}[name]


def _device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.rsplit(":", 1)[1])
    except ValueError:
        return 0


def _synchronize(device: Any | None = None) -> None:
    assert torch is not None
    torch.cuda.synchronize(device)


def _measure(fn: Callable[[], Any], *, warmup: int, iterations: int, device: Any) -> list[float]:
    assert torch is not None
    for _ in range(warmup):
        fn()
    _synchronize(device)
    times: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _synchronize(device)
        times.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    return times


def _stats(times: list[float]) -> dict[str, float]:
    if not times:
        raise KernelBenchError("no timing samples")
    ordered = sorted(times)
    mean = statistics.fmean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "median_us": statistics.median(times),
        "mean_us": mean,
        "p90_us": ordered[p90_index],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "cv": 0.0 if mean == 0 else std / mean,
    }


def _result(
    *,
    kernel: str,
    mode: str,
    batch: int,
    rows: int,
    times: list[float],
    weight_bytes: int,
) -> dict[str, Any]:
    stats = _stats(times)
    seconds = stats["median_us"] / 1e6
    return {
        "kernel": kernel,
        "mode": mode,
        "batch": batch,
        "rows": rows,
        "weight_bytes": weight_bytes,
        "weight_gbps": 0.0 if seconds == 0 else weight_bytes / seconds / 1e9,
        **stats,
    }


def _numerical_diff(actual: Any, expected: Any) -> float:
    return float((actual - expected.float()).abs().max().item())


def run_kernel_bench(
    config: KernelBenchConfig,
    *,
    clock: GpuSnapshot | None = None,
) -> dict[str, Any]:
    """Run the full/gather/fused kernel matrix and return a JSON-ready result."""

    _require_torch_cuda()
    assert torch is not None
    if "cuda" not in config.device:
        raise KernelBenchError("kernel benchmark requires a CUDA device")

    device = torch.device(config.device)
    torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    dtype = _torch_dtype(config.dtype)
    element_bytes = _dtype_bytes(config.dtype)

    weight = torch.randn(
        config.vocab_size,
        config.hidden_size,
        dtype=dtype,
        device=device,
    )
    max_rows = max(config.row_counts)
    # Keep a native int32 view for Triton and a long view for torch index ops so
    # graph replay does not pay an implicit dtype-conversion copy on either side.
    row_permutation = torch.randperm(config.vocab_size, device=device)[:max_rows]
    row_permutation_int = row_permutation.to(torch.int32)
    row_permutation_long = row_permutation.to(torch.long)

    selected = set(config.kernels)
    graph_available = config.cuda_graph and CudaGraphCache.is_available()
    full_kernel = TorchFullKernel()
    gather_kernel = TorchGatherKernel()
    triton_kernel = TritonSliceKernel()
    full_graph = TorchFullKernel(use_cuda_graph=True) if graph_available else None
    gather_graph = TorchGatherKernel(use_cuda_graph=True) if graph_available else None
    triton_graph = TritonSliceKernel(use_cuda_graph=True) if graph_available else None

    results: list[dict[str, Any]] = []
    for batch in config.batch_sizes:
        hidden = torch.randn(batch, config.hidden_size, dtype=dtype, device=device)
        full_bytes = config.vocab_size * config.hidden_size * element_bytes
        full_result: dict[str, Any] | None = None

        def add(
            kernel: str,
            mode: str,
            rows: int,
            times: list[float],
            weight_bytes: int,
            batch: int = batch,
        ) -> dict[str, Any]:
            row = _result(
                kernel=kernel,
                mode=mode,
                batch=batch,
                rows=rows,
                times=times,
                weight_bytes=weight_bytes,
            )
            results.append(row)
            return row

        if "full" in selected:
            full_times = _measure(
                lambda hidden=hidden: full_kernel.run_tensor(hidden, weight),
                warmup=config.warmup,
                iterations=config.iterations,
                device=device,
            )
            full_result = add("full", "eager", config.vocab_size, full_times, full_bytes)
            if full_graph is not None:
                graph_times = _measure(
                    lambda hidden=hidden: full_graph.run_tensor(hidden, weight),
                    warmup=max(1, config.warmup),
                    iterations=config.iterations,
                    device=device,
                )
                graph_result = add(
                    "full_graph", "graph", config.vocab_size, graph_times, full_bytes
                )
                graph_result["speedup_vs_eager"] = (
                    full_result["median_us"] / graph_result["median_us"]
                )

        for rows in config.row_counts:
            row_index = row_permutation_long[:rows]
            triton_index = row_permutation_int[:rows]
            row_bytes = rows * config.hidden_size * element_bytes
            gather_result: dict[str, Any] | None = None
            triton_result: dict[str, Any] | None = None
            eager_reference: Any | None = None

            if "torch_gather" in selected:
                gather_times = _measure(
                    lambda hidden=hidden, row_index=row_index: gather_kernel.run_tensor(
                        hidden, weight, row_index
                    ),
                    warmup=config.warmup,
                    iterations=config.iterations,
                    device=device,
                )
                gather_result = add("torch_gather", "eager", rows, gather_times, row_bytes)
                eager_reference = gather_kernel.run_tensor(hidden, weight, row_index)

            if "triton_slice" in selected:
                triton_times = _measure(
                    lambda hidden=hidden, triton_index=triton_index: triton_kernel.run_tensor(
                        hidden, weight, triton_index
                    ),
                    warmup=config.warmup,
                    iterations=config.iterations,
                    device=device,
                )
                triton_result = add("triton_slice", "eager", rows, triton_times, row_bytes)
                if gather_result is not None:
                    triton_result["speedup_vs_torch_gather"] = (
                        gather_result["median_us"] / triton_result["median_us"]
                    )
                if eager_reference is not None:
                    triton_result["numerical_max_abs_diff"] = _numerical_diff(
                        triton_kernel.run_tensor(hidden, weight, triton_index),
                        eager_reference,
                    )
                if full_result is not None:
                    triton_result["speedup_vs_full"] = (
                        full_result["median_us"] / triton_result["median_us"]
                    )

            if graph_available and rows <= config.graph_rows_limit:
                gather_graph_result: dict[str, Any] | None = None
                if gather_graph is not None and "torch_gather" in selected:
                    gather_graph_times = _measure(
                        lambda hidden=hidden, row_index=row_index: gather_graph.run_tensor(
                            hidden, weight, row_index
                        ),
                        warmup=max(1, config.warmup),
                        iterations=config.iterations,
                        device=device,
                    )
                    gather_graph_result = add(
                        "torch_gather_graph", "graph", rows, gather_graph_times, row_bytes
                    )
                    if gather_result is not None:
                        gather_graph_result["speedup_vs_eager"] = (
                            gather_result["median_us"] / gather_graph_result["median_us"]
                        )
                    if eager_reference is not None:
                        gather_graph_result["numerical_max_abs_diff"] = _numerical_diff(
                            gather_graph.run_tensor(hidden, weight, row_index),
                            eager_reference,
                        )
                if triton_graph is not None and "triton_slice" in selected:
                    triton_graph_times = _measure(
                        lambda hidden=hidden, triton_index=triton_index: triton_graph.run_tensor(
                            hidden, weight, triton_index
                        ),
                        warmup=max(1, config.warmup),
                        iterations=config.iterations,
                        device=device,
                    )
                    triton_graph_result = add(
                        "triton_slice_graph", "graph", rows, triton_graph_times, row_bytes
                    )
                    if gather_graph_result is not None:
                        triton_graph_result["speedup_vs_torch_gather"] = (
                            gather_graph_result["median_us"] / triton_graph_result["median_us"]
                        )
                    if triton_result is not None:
                        triton_graph_result["speedup_vs_eager"] = (
                            triton_result["median_us"] / triton_graph_result["median_us"]
                        )
                    if full_result is not None:
                        triton_graph_result["speedup_vs_full"] = (
                            full_result["median_us"] / triton_graph_result["median_us"]
                        )
                    if eager_reference is not None:
                        triton_graph_result["numerical_max_abs_diff"] = _numerical_diff(
                            triton_graph.run_tensor(hidden, weight, triton_index),
                            eager_reference,
                        )

    if clock is None:
        try:
            gpu = query_gpu(_device_index(config.device)).to_dict()
        except Exception:
            gpu = None
    else:
        gpu = clock.to_dict()

    graph_stats = {
        "available": graph_available,
        "captures": 0,
        "replays": 0,
        "fallbacks": 0,
        "disabled_reason": None,
    }
    disabled: list[str] = []
    for kernel in (full_graph, gather_graph, triton_graph):
        cache = None if kernel is None else kernel.graph_cache
        if cache is None:
            continue
        graph_stats["captures"] += cache.captures
        graph_stats["replays"] += cache.replays
        graph_stats["fallbacks"] += cache.fallbacks
        if cache.disabled_reason:
            disabled.append(cache.disabled_reason)
    if disabled:
        graph_stats["disabled_reason"] = "; ".join(sorted(set(disabled)))

    return {
        "schema_version": "kernel_bench/1",
        "config": asdict(config),
        "gpu": gpu,
        "clock": None if clock is None else clock.to_dict(),
        "clock_fixed": clock is not None,
        "torch_version": None if torch is None else torch.__version__,
        "triton_version": None if triton is None else triton.__version__,
        "triton_available": TritonSliceKernel.is_available(),
        "graph": graph_stats,
        "results": results,
    }


def write_kernel_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "KERNEL_NAMES",
    "KernelBenchConfig",
    "KernelBenchError",
    "run_kernel_bench",
    "write_kernel_result",
]
