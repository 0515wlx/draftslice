"""Constraint-session protocol.

DraftSlice does not implement grammars. A backend (xgrammar, outlines, an engine
adapter, or the reference fixture backend) supplies the current valid-token set
and the transition for a chosen token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..plan.types import TokenSet


class ConstraintError(RuntimeError):
    """Raised when a token is not valid or a transition does not exist."""


@dataclass(frozen=True, slots=True)
class ConstraintState:
    """A snapshot of one grammar state."""

    state_id: int
    valid_ids: TokenSet
    final: bool = False

    def __post_init__(self) -> None:
        if self.state_id < 0:
            raise ValueError("state_id must be non-negative")


class ConstraintSession(Protocol):
    """A mutable cursor over a constraint state machine."""

    def current(self) -> ConstraintState: ...

    def advance(self, token_id: int) -> None: ...

    def fork(self) -> ConstraintSession: ...

    def is_final(self) -> bool: ...


@dataclass(slots=True)
class FixedConstraintSession:
    """A session with one fixed valid set and no transitions.

    This is the reference backend for tests and plan fixtures. It is deliberately
    not a grammar engine: ``advance`` always fails.
    """

    state: ConstraintState

    def current(self) -> ConstraintState:
        return self.state

    def advance(self, token_id: int) -> None:
        raise ConstraintError(f"fixed constraint session has no transitions (token {token_id})")

    def fork(self) -> FixedConstraintSession:
        return FixedConstraintSession(state=self.state)

    def is_final(self) -> bool:
        return self.state.final


__all__ = [
    "ConstraintError",
    "ConstraintSession",
    "ConstraintState",
    "FixedConstraintSession",
]
