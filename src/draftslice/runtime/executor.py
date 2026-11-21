"""Plan validation and execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..kernels.base import KernelError
from ..kernels.registry import KernelRegistry
from ..plan.types import HeadPlan, TokenSet


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Kernel output plus the execution metadata needed by callers."""

    plan_id: str
    kernel: str
    rows: TokenSet
    logits: Any

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


def _as_batch(hidden: Sequence[float] | Sequence[Sequence[float]]):
    if not hidden:
        raise KernelError("hidden must not be empty")
    first = hidden[0]
    if isinstance(first, (int, float)):
        return [hidden]  # type: ignore[list-item]
    return hidden


class ReferenceExecutor:
    """Execute a plan with a reference kernel.

    The executor is deliberately strict: a plan that claims exact valid-set
    coverage is rejected before the kernel runs if its execution rows do not
    cover the candidate set.
    """

    def __init__(
        self,
        weight: Sequence[Sequence[float]],
        *,
        registry: KernelRegistry | None = None,
    ) -> None:
        if not weight:
            raise ValueError("weight must not be empty")
        self.weight = weight
        self.registry = registry or KernelRegistry()

    def execute(
        self,
        hidden: Sequence[float] | Sequence[Sequence[float]],
        plan: HeadPlan,
    ) -> ExecutionResult:
        if plan.guarantee.kind.value == "exact_valid_set" and not plan.execution.covers(
            plan.candidates
        ):
            raise KernelError("strict plan does not cover its candidates")

        batch = _as_batch(hidden)
        kernel = self.registry.select(plan, len(batch))
        result = kernel.run(batch, plan, self.weight)

        if plan.guarantee.kind.value == "exact_valid_set" and not result.rows.covers(
            plan.candidates
        ):
            raise KernelError("kernel result does not cover the strict candidate set")

        return ExecutionResult(
            plan_id=plan.plan_id,
            kernel=kernel.name,
            rows=result.rows,
            logits=result.logits,
        )


__all__ = ["ExecutionResult", "ReferenceExecutor"]
