"""Kernel selection and fallback."""

from __future__ import annotations

from collections.abc import Iterable

from ..plan.types import HeadPlan
from .base import HeadKernel, KernelError
from .reference import ReferenceKernel


class KernelRegistry:
    """Select the first registered kernel that supports a plan."""

    def __init__(self, kernels: Iterable[HeadKernel] | None = None) -> None:
        self._kernels: list[HeadKernel] = list(kernels or [ReferenceKernel()])
        if not self._kernels:
            raise ValueError("registry needs at least one kernel")

    def register(self, kernel: HeadKernel, *, front: bool = False) -> None:
        if front:
            self._kernels.insert(0, kernel)
        else:
            self._kernels.append(kernel)

    def select(self, plan: HeadPlan, batch: int) -> HeadKernel:
        for kernel in self._kernels:
            if kernel.supports(plan, batch):
                return kernel
        raise KernelError(f"no kernel supports plan {plan.plan_id or '<anonymous>'}")

    @property
    def kernels(self) -> tuple[HeadKernel, ...]:
        return tuple(self._kernels)


__all__ = ["KernelRegistry"]
