"""Draft-head runtime interfaces and the reference executor."""

from __future__ import annotations

from typing import Any, Protocol

from ..plan import HeadPlan


class DraftHeadExecutor(Protocol):
    """Execute a head plan against a draft hidden state."""

    def execute(self, hidden: Any, plan: HeadPlan) -> Any: ...


from .acceptance import (  # noqa: E402
    AcceptanceError,
    AcceptanceEstimate,
    AcceptanceModel,
    CoverageAcceptanceModel,
    StaticAcceptanceModel,
    TreeShape,
    coverage_from_measurements,
    estimate_from_stats,
)
from .executor import ExecutionResult, ReferenceExecutor  # noqa: E402
from .scheduler import Schedule, Scheduler, StepContext  # noqa: E402
from .union import UnionExecutor, UnionResult  # noqa: E402

__all__ = [
    "AcceptanceError",
    "AcceptanceEstimate",
    "AcceptanceModel",
    "CoverageAcceptanceModel",
    "DraftHeadExecutor",
    "ExecutionResult",
    "ReferenceExecutor",
    "Schedule",
    "Scheduler",
    "StaticAcceptanceModel",
    "StepContext",
    "TreeShape",
    "UnionExecutor",
    "UnionResult",
    "coverage_from_measurements",
    "estimate_from_stats",
]
