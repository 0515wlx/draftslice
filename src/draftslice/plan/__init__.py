"""Head-plan types and compiler interfaces.

A head plan is the executable form of a grammar state's valid-token set (strict
mode) or of a screened candidate set (calibrated mode). It is data-only so it can
be serialized, cached, and passed across the compiler/runtime boundary.
"""

from __future__ import annotations

from .compiler import Budget, PlanBudgetError, PlanCompiler, PlanPolicy
from .types import (
    ContractMode,
    DraftContract,
    Execution,
    FallbackRule,
    Guarantee,
    GuaranteeKind,
    GuaranteeScope,
    HeadPlan,
    KernelKind,
    LayoutKind,
    NodePlan,
    PlanCost,
    ServingPoint,
    Slice,
    StepPlan,
    TokenSet,
    Universe,
)

__all__ = [
    "Budget",
    "ContractMode",
    "DraftContract",
    "Execution",
    "FallbackRule",
    "Guarantee",
    "GuaranteeKind",
    "GuaranteeScope",
    "HeadPlan",
    "KernelKind",
    "LayoutKind",
    "NodePlan",
    "PlanBudgetError",
    "PlanCompiler",
    "PlanCost",
    "PlanPolicy",
    "ServingPoint",
    "Slice",
    "StepPlan",
    "TokenSet",
    "Universe",
]
