"""Candidate-screen interfaces.

A screener proposes a candidate token set ``A``. The executor rescores ``A``
exactly. In strict mode the candidates must equal the grammar-valid set; in
calibrated mode they may be a subset with a reported coverage delta.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..plan.types import TokenSet


class ScreenError(ValueError):
    """Raised when a screener result is malformed."""


class Screener(Protocol):
    name: str

    def candidates(self, hidden_states: Any) -> list[TokenSet]: ...


def intersect_universe(
    candidates: list[TokenSet],
    universe: list[TokenSet] | None,
) -> list[TokenSet]:
    """Restrict candidates to a per-row universe (for example the valid set)."""

    if universe is None:
        return candidates
    if len(candidates) != len(universe):
        raise ScreenError("candidates and universe must have equal length")
    return [
        candidate if valid is None else candidate.intersection(valid)
        for candidate, valid in zip(candidates, universe, strict=True)
    ]


def validate_candidates(candidates: list[TokenSet], *, batch: int) -> None:
    if len(candidates) != batch:
        raise ScreenError(f"screener returned {len(candidates)} rows for batch={batch}")
    for token_set in candidates:
        if not isinstance(token_set, TokenSet):
            raise ScreenError("screener must return TokenSet rows")


__all__ = [
    "ScreenError",
    "Screener",
    "intersect_universe",
    "validate_candidates",
]
