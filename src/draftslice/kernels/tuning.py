"""Static kernel launch configurations.

The slice kernel has three regimes:

* ``batch == 1`` and a small row count: the launch is dominant. A row-block
  GEMV kernel that keeps several weight rows per program is the fastest eager
  shape and is the best candidate for CUDA-graph replay.
* ``batch >= 2`` and a small row count: a tensor-core GEMM over the gathered
  rows wins because the batch dimension amortises the weight traffic.
* ``batch >= 2`` and a large row count: the same GEMM is bandwidth-bound; the
  block shape matters less than keeping the gather coalesced.

The values below are a conservative starting point measured on A100-80GB with
bf16 weights. They are intentionally data-only so a future runtime autotuner can
replace them without changing the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """One Triton launch configuration."""

    block_m: int
    block_d: int
    block_b: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name in ("block_m", "block_d", "block_b", "num_warps", "num_stages"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("block_m", "block_d", "block_b"):
            value = getattr(self, name)
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def gemv_config(rows: int, width: int, *, element_bytes: int = 2) -> KernelConfig:
    """Pick a row-block GEMV configuration for a single hidden state.

    ``block_d`` is capped at the next power of two of the hidden width so the
    last program does not request a mostly masked tile.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if rows <= 4_096:
        block_m = 8
    elif rows <= 16_384:
        block_m = 16
    else:
        block_m = 8
    block_d = 512 if rows <= 16_384 else 256
    if element_bytes >= 4 and rows <= 8_192:
        block_d = min(block_d, 256)
    block_d = min(block_d, _next_power_of_two(width))
    return KernelConfig(
        block_m=block_m,
        block_d=block_d,
        block_b=1,
        num_warps=4,
        num_stages=1,
    )


def gemm_config(
    rows: int,
    batch: int,
    width: int,
    *,
    element_bytes: int = 2,
) -> KernelConfig:
    """Pick a tensor-core slice-GEMM configuration."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    block_b = max(16, _next_power_of_two(batch))
    if block_b >= 32:
        if rows <= 1_024:
            block_m, block_d, num_stages = 16, 64, 3
        elif rows <= 8_192:
            if block_b >= 64:
                block_m, block_d, num_stages = 64, 64, 4
            else:
                block_m, block_d, num_stages = 32, 128, 3
        elif rows <= 16_384:
            block_m, block_d, num_stages = 64, 128, 3 if block_b == 32 else 2
        elif block_b == 32:
            block_m, block_d, num_stages = 32, 128, 4
        else:
            block_m, block_d, num_stages = 64, 64, 4
    elif rows <= 1_024:
        block_m, block_d, num_stages = 32, 128, 4
    elif rows <= 4_096:
        block_m, block_d, num_stages = 64, 256, 3
    elif rows <= 8_192:
        block_m, block_d, num_stages = 64, 128, 3
    else:
        block_m, block_d, num_stages = 64, 128, 3

    if element_bytes >= 4:
        block_d = min(block_d, 128)
        num_stages = min(num_stages, 2)
    block_d = min(block_d, _next_power_of_two(width))

    # Conservative shared-memory guard for devices below the A100/H100 budget.
    # The measured A100 usage is usually one pipeliner buffer lower; this bound
    # only shrinks a config that is clearly over the hardware limit.
    shared_limit = 160 * 1024
    buffers = min(2, num_stages)
    while num_stages > 1 and (
        2 * element_bytes * buffers * (block_m + block_b) * block_d > shared_limit
    ):
        num_stages -= 1
        buffers = min(2, num_stages)
    while block_d > 64 and (
        2 * element_bytes * buffers * (block_m + block_b) * block_d > shared_limit
    ):
        block_d //= 2
    while block_m > 16 and (
        2 * element_bytes * buffers * (block_m + block_b) * block_d > shared_limit
    ):
        block_m //= 2

    return KernelConfig(
        block_m=block_m,
        block_d=block_d,
        block_b=block_b,
        num_warps=4,
        num_stages=num_stages,
    )


__all__ = ["KernelConfig", "gemv_config", "gemm_config"]
