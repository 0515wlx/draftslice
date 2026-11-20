"""DraftSlice: exact, constraint-aware draft-head slicing.

This package is a pre-alpha skeleton. Public interfaces live under
``draftslice.plan``, ``draftslice.cache``, ``draftslice.runtime``, and
``draftslice.kernels``.
"""

from .plan import HeadPlan, Slice

__version__ = "0.0.1"

__all__ = ["HeadPlan", "Slice", "__version__"]
