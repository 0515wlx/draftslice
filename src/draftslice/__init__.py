"""DraftSlice: exact, constraint-aware draft-head slicing.

Public interfaces live under ``draftslice.plan``, ``draftslice.constraints``,
``draftslice.cache``, ``draftslice.runtime``, ``draftslice.kernels``, and
``draftslice.bench``.
"""

from .plan import (
    Execution,
    Guarantee,
    GuaranteeKind,
    HeadPlan,
    PlanCost,
    Slice,
    StepPlan,
    TokenSet,
)

__version__ = "0.0.1"

__all__ = [
    "Execution",
    "Guarantee",
    "GuaranteeKind",
    "HeadPlan",
    "PlanCost",
    "Slice",
    "StepPlan",
    "TokenSet",
    "__version__",
]
