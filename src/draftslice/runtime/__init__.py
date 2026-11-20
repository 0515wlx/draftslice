"""Draft-head runtime interfaces."""

from __future__ import annotations

from typing import Any, Protocol

from ..plan import HeadPlan


class DraftHeadExecutor(Protocol):
    """Execute a head plan against a draft hidden state."""

    def execute(self, hidden: Any, plan: HeadPlan) -> Any: ...


__all__ = ["DraftHeadExecutor"]
