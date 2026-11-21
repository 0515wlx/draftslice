"""Cache interfaces for head plans and hot rows."""

from __future__ import annotations

from typing import Protocol

from ..plan import HeadPlan


class PlanCache(Protocol):
    """State -> head plan cache, shared across requests and decode steps."""

    def get(self, state_id: int) -> HeadPlan | None: ...

    def put(self, plan: HeadPlan) -> None: ...

    def clear(self) -> None: ...


from .candidate_cache import CandidateSetCache  # noqa: E402
from .plan_cache import CacheStats, LRUPlanCache, plan_bytes  # noqa: E402
from .row_cache import PackedRowCache, PackedRows, pack_weight_rows, rowset_key  # noqa: E402

__all__ = [
    "CacheStats",
    "CandidateSetCache",
    "LRUPlanCache",
    "PackedRowCache",
    "PackedRows",
    "PlanCache",
    "pack_weight_rows",
    "plan_bytes",
    "rowset_key",
]
