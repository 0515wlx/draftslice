"""A dependency-free reference kernel.

This is intentionally slow. It exists to pin down semantics, validate optimized
backends, and let the core run in tests and on CPU without torch.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..plan.types import HeadPlan, TokenSet
from .base import KernelError, KernelResult


def _validate_matrix(weight: Sequence[Sequence[float]], name: str) -> int:
    if not weight:
        raise KernelError(f"{name} must not be empty")
    width = len(weight[0])
    if width == 0:
        raise KernelError(f"{name} rows must not be empty")
    for index, row in enumerate(weight):
        if len(row) != width:
            raise KernelError(f"{name} row {index} has width {len(row)}, expected {width}")
    return width


def _validate_hidden(hidden: Sequence[Sequence[float]], width: int) -> None:
    if not hidden:
        raise KernelError("hidden must contain at least one batch item")
    for index, row in enumerate(hidden):
        if len(row) != width:
            raise KernelError(f"hidden[{index}] has width {len(row)}, expected {width}")


class ReferenceKernel:
    """Dense dot-product reference implementation."""

    name = "reference"

    def supports(self, plan: HeadPlan, batch: int) -> bool:
        return batch >= 0

    def run(
        self,
        hidden: Sequence[Sequence[float]],
        plan: HeadPlan,
        weight: Sequence[Sequence[float]],
    ) -> KernelResult:
        width = _validate_matrix(weight, "weight")
        _validate_hidden(hidden, width)

        if plan.execution.kind == "full":
            rows = TokenSet(range(len(weight)))
        else:
            rows = plan.execution.rows
            if rows is None:
                raise KernelError("slice execution has no rows")

        for token_id in rows:
            if token_id >= len(weight):
                raise KernelError(f"row {token_id} is outside the weight matrix")

        logits: list[tuple[float, ...]] = []
        for hidden_row in hidden:
            scores = []
            for token_id in rows:
                weight_row = weight[token_id]
                scores.append(sum(h * w for h, w in zip(hidden_row, weight_row, strict=True)))
            logits.append(tuple(scores))
        return KernelResult(rows=rows, logits=tuple(logits))


__all__ = ["ReferenceKernel"]
