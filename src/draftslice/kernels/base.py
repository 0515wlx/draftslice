"""Kernel interfaces and result types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..plan.types import HeadPlan, TokenSet


class KernelError(RuntimeError):
    """Raised when a kernel cannot execute a plan."""


@dataclass(frozen=True, slots=True)
class KernelResult:
    """Execution rows and their logits.

    ``logits[b][i]`` (or ``logits[b, i]`` for a tensor) is the score for
    ``rows.ids[i]`` for batch item ``b``. Python kernels return tuples; GPU
    kernels may return a backend tensor so timing does not force a device sync.
    Use :meth:`candidate_logits` to project onto the logical candidate set when
    the execution rows are a superset.
    """

    rows: TokenSet
    logits: Any
    backend: str = "python"

    def __post_init__(self) -> None:
        shape = getattr(self.logits, "shape", None)
        if shape is not None and len(shape) == 2 and int(shape[1]) != len(self.rows):
            raise KernelError(f"logit width {shape[1]} does not match {len(self.rows)} rows")

    def candidate_logits(self, plan: HeadPlan) -> Any:
        index = {token_id: position for position, token_id in enumerate(self.rows.ids)}
        missing = [token_id for token_id in plan.candidates.ids if token_id not in index]
        if missing:
            raise KernelError(f"execution rows miss candidates: {missing[:8]}")
        positions = [index[token_id] for token_id in plan.candidates.ids]
        if isinstance(self.logits, Sequence):
            return tuple(
                tuple(batch_row[position] for position in positions) for batch_row in self.logits
            )
        return self.logits[:, positions]


class HeadKernel(Protocol):
    """A head kernel backend."""

    name: str

    def supports(self, plan: HeadPlan, batch: int) -> bool: ...

    def run(
        self,
        hidden: Sequence[Sequence[float]],
        plan: HeadPlan,
        weight: Sequence[Sequence[float]],
    ) -> KernelResult: ...


__all__ = ["HeadKernel", "KernelError", "KernelResult"]
