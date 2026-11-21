"""Union and tree execution.

A step plan contains one ``HeadPlan`` per draft-tree node (or per request). Many
nodes share rows, so the executor deduplicates the union, runs one slice kernel
over the union, and scatters the result back to each node. This is the runtime
lever behind tree drafting: the union is much smaller than the sum of the
per-node candidate sets.

The executor is backend-agnostic. It only requires a kernel with
``run_tensor(hidden, weight, rows) -> [batch, len(rows)]`` and, for full
executions, ``run_tensor(hidden, weight) -> [batch, vocab]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..kernels.base import KernelError
from ..plan.types import HeadPlan, StepPlan, TokenSet


@dataclass(frozen=True, slots=True)
class UnionResult:
    """Result of one union execution."""

    union_rows: TokenSet
    kernel: str
    logits: Any
    node_logits: tuple[Any, ...]

    @property
    def size(self) -> int:
        return len(self.union_rows)


class UnionExecutor:
    """Execute per-node plans with one union kernel launch."""

    def __init__(self, kernel: Any, *, full_kernel: Any | None = None) -> None:
        if kernel is None:
            raise ValueError("a slice kernel is required")
        self.kernel = kernel
        self.full_kernel = full_kernel

    def union(self, plans: tuple[HeadPlan, ...] | list[HeadPlan]) -> TokenSet:
        rows: list[int] = []
        for plan in plans:
            if plan.execution.kind == "full":
                continue
            execution_rows = plan.execution.rows
            if execution_rows is None:
                raise KernelError("slice plan has no execution rows")
            if not execution_rows.covers(plan.candidates):
                raise KernelError("execution rows do not cover the candidate set")
            rows.extend(execution_rows.ids)
        return TokenSet(rows)

    def execute(
        self,
        hidden: Any,
        weight: Any,
        plans: tuple[HeadPlan, ...] | list[HeadPlan],
    ) -> UnionResult:
        if not plans:
            raise KernelError("union execution needs at least one plan")
        if int(hidden.shape[0]) != len(plans):
            raise KernelError(
                f"hidden has {hidden.shape[0]} rows but {len(plans)} plans were supplied"
            )

        if any(plan.execution.kind == "full" for plan in plans):
            return self._execute_full(hidden, weight, plans)

        union = self.union(plans)
        if not union:
            return UnionResult(
                union_rows=union,
                kernel=getattr(self.kernel, "name", "unknown"),
                logits=hidden.new_empty((hidden.shape[0], 0)),
                node_logits=tuple(hidden.new_empty((1, len(plan.candidates))) for plan in plans),
            )

        logits = self.kernel.run_tensor(hidden, weight, union)
        node_logits = tuple(
            _select(logits, row, union, plan.candidates) for row, plan in enumerate(plans)
        )
        return UnionResult(
            union_rows=union,
            kernel=getattr(self.kernel, "name", "unknown"),
            logits=logits,
            node_logits=node_logits,
        )

    def execute_step(self, hidden: Any, weight: Any, step: StepPlan) -> UnionResult:
        plans = [node.plan for node in step.nodes]
        if not plans:
            raise KernelError("step plan has no nodes")
        if step.kernel == "full" or any(plan.execution.kind == "full" for plan in plans):
            return self._execute_full(hidden, weight, plans)
        union = self.union(plans)
        if not union.covers(step.union_rows):
            # The step's declared union is a contract; never silently execute a
            # smaller set.
            raise KernelError("step union rows do not cover the declared union")
        logits = self.kernel.run_tensor(hidden, weight, union)
        node_logits = tuple(
            _select(logits, row, union, plan.candidates) for row, plan in enumerate(plans)
        )
        return UnionResult(
            union_rows=union,
            kernel=getattr(self.kernel, "name", "unknown"),
            logits=logits,
            node_logits=node_logits,
        )

    def _execute_full(
        self,
        hidden: Any,
        weight: Any,
        plans: tuple[HeadPlan, ...] | list[HeadPlan],
    ) -> UnionResult:
        if self.full_kernel is None:
            raise KernelError("full execution requires full_kernel")
        logits = self.full_kernel.run_tensor(hidden, weight)
        vocab_rows = TokenSet(range(int(logits.shape[-1])))
        node_logits = tuple(
            _select(logits, row, vocab_rows, plan.candidates) for row, plan in enumerate(plans)
        )
        return UnionResult(
            union_rows=vocab_rows,
            kernel=getattr(self.full_kernel, "name", "full"),
            logits=logits,
            node_logits=node_logits,
        )


def _select(logits: Any, row: int, rows: TokenSet, candidates: TokenSet) -> Any:
    index = {token_id: position for position, token_id in enumerate(rows.ids)}
    missing = [token_id for token_id in candidates.ids if token_id not in index]
    if missing:
        raise KernelError(f"union rows miss candidates for node {row}: {missing[:8]}")
    positions = [index[token_id] for token_id in candidates.ids]
    return logits[row : row + 1, positions]


__all__ = ["UnionExecutor", "UnionResult"]
