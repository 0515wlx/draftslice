"""Draft-head kernels.

The reference kernel pins semantics. The torch and Triton kernels provide the
baseline and optimized GPU paths, including an optional CUDA-graph replay cache
for launch-bound small candidate sets.
"""

from __future__ import annotations

from .autotune import AutotuneError, KernelAutotuner, gemm_candidates, gemv_candidates
from .base import HeadKernel, KernelError, KernelResult
from .graph import CudaGraphCache, GraphError
from .reference import ReferenceKernel
from .registry import KernelRegistry
from .tuning import KernelConfig, gemm_config, gemv_config

__all__ = [
    "AutotuneError",
    "CudaGraphCache",
    "GraphError",
    "HeadKernel",
    "KernelAutotuner",
    "KernelConfig",
    "KernelError",
    "KernelRegistry",
    "KernelResult",
    "ReferenceKernel",
    "gemm_candidates",
    "gemm_config",
    "gemv_candidates",
    "gemv_config",
]
