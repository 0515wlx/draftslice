"""Cache interfaces for head plans and hot rows."""

from __future__ import annotations

from typing import Protocol

from ..plan import HeadPlan


class PlanCache(Protocol):
    """State -> head plan cache, shared across requests and decode steps."""

    def get(self, state_id: int) -> HeadPlan | None: ...

    def put(self, plan: HeadPlan) -> None: ...

    def clear(self) -> None: ...


__all__ = ["PlanCache"]
