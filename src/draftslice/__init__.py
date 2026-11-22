"""DraftSlice: exact, constraint-aware draft-head slicing.

Public interfaces live under ``draftslice.plan``, ``draftslice.constraints``,
``draftslice.cache``, ``draftslice.runtime``, ``draftslice.kernels``, and
``draftslice.bench``.
"""

from .plan import (
    DraftContract,
    Execution,
    Guarantee,
    GuaranteeKind,
    HeadPlan,
    PlanCost,
    ServingPoint,
    Slice,
    StepPlan,
    TokenSet,
)

__version__ = "0.0.1"

__all__ = [
    "DraftContract",
    "Execution",
    "Guarantee",
    "GuaranteeKind",
    "HeadPlan",
    "PlanCost",
    "ServingPoint",
    "Slice",
    "StepPlan",
    "TokenSet",
    "__version__",
]
