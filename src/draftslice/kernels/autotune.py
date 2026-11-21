"""Runtime kernel autotuning.

The static table in :mod:`draftslice.kernels.tuning` is a good default, but the
best tile depends on the GPU, dtype, and exact shape. This module benchmarks a
small set of nearby configurations on first use and caches the winner per shape.

Usage is opt-in (``TritonSliceKernel(autotune=True)``) because the first call for
a shape pays several short kernel runs. Cache keys include device, dtype, and
shape, not model identity.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from .tuning import KernelConfig, gemm_config, gemv_config

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class AutotuneError(RuntimeError):
    """Raised when no candidate configuration runs."""


def _dedupe(configs: Sequence[KernelConfig]) -> tuple[KernelConfig, ...]:
    seen = set()
    result = []
    for config in configs:
        key = (
            config.block_m,
            config.block_d,
            config.block_b,
            config.num_warps,
            config.num_stages,
        )
        if key not in seen:
            seen.add(key)
            result.append(config)
    return tuple(result)


def _shared_ok(config: KernelConfig, element_bytes: int, limit: int = 160 * 1024) -> bool:
    buffers = min(2, config.num_stages)
    shared = 2 * element_bytes * buffers * (config.block_m + config.block_b) * config.block_d
    return shared <= limit


def gemv_candidates(rows: int, width: int, *, element_bytes: int = 2) -> tuple[KernelConfig, ...]:
    base = gemv_config(rows, width, element_bytes=element_bytes)
    configs = [base]
    for block_m in (max(1, base.block_m // 2), base.block_m * 2):
        for block_d in (max(64, base.block_d // 2), base.block_d):
            configs.append(replace(base, block_m=block_m, block_d=block_d))
    return _dedupe(configs)


def gemm_candidates(
    rows: int,
    batch: int,
    width: int,
    *,
    element_bytes: int = 2,
) -> tuple[KernelConfig, ...]:
    base = gemm_config(rows, batch, width, element_bytes=element_bytes)
    configs = [base]
    for block_m in (max(16, base.block_m // 2), base.block_m * 2):
        configs.append(replace(base, block_m=block_m))
    for block_d in (max(64, base.block_d // 2),):
        configs.append(replace(base, block_d=block_d))
    for stages in (max(1, base.num_stages - 1), base.num_stages + 1):
        configs.append(replace(base, num_stages=stages))
    return tuple(config for config in _dedupe(configs) if _shared_ok(config, element_bytes))


class KernelAutotuner:
    """Benchmark nearby launch configurations and cache the fastest."""

    def __init__(self, *, warmup: int = 2, iterations: int = 5) -> None:
        if warmup < 0 or iterations <= 0:
            raise ValueError("warmup must be >= 0 and iterations > 0")
        self.warmup = warmup
        self.iterations = iterations
        self._best: dict[Any, KernelConfig] = {}
        self.timings: dict[Any, float] = {}
        self.trials = 0

    def choose(
        self,
        key: Any,
        candidates: Sequence[KernelConfig],
        run: Callable[[KernelConfig], Any],
    ) -> KernelConfig:
        if key in self._best:
            return self._best[key]
        if torch is None or not torch.cuda.is_available():
            return candidates[0]

        best: KernelConfig | None = None
        best_time = float("inf")
        for config in candidates:
            try:
                time_us = self._measure(lambda config=config: run(config))
            except Exception:
                continue
            self.trials += 1
            if time_us < best_time:
                best = config
                best_time = time_us
        if best is None:
            raise AutotuneError(f"no candidate configuration ran for {key!r}")
        self._best[key] = best
        self.timings[key] = best_time
        return best

    def _measure(self, fn: Callable[[], Any]) -> float:
        assert torch is not None
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return statistics.median(times)

    def clear(self) -> None:
        self._best.clear()
        self.timings.clear()
        self.trials = 0


__all__ = [
    "AutotuneError",
    "KernelAutotuner",
    "gemm_candidates",
    "gemv_candidates",
]
