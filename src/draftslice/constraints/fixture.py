"""Reference constraint-fixture backend.

The public CLI needs a backend that works without a model tokenizer. This module
defines a small JSON fixture format:

    {"state_id": 0, "vocab_size": 8, "hidden_size": 4, "valid_ids": [1, 3]}

or multiple states:

    {"states": [{"state_id": 0, "valid_ids": [1, 3]},
                {"state_id": 1, "valid_ids": [2]}]}

It is a fixture format, not a grammar engine. Real JSON Schema / CFG support is
provided by an adapter in a later milestone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..plan.types import TokenSet
from .base import ConstraintState


class ConstraintFixtureError(ValueError):
    """Raised when a constraint fixture is malformed."""


@dataclass(frozen=True, slots=True)
class ConstraintFixture:
    states: tuple[ConstraintState, ...]
    vocab_size: int | None = None
    hidden_size: int | None = None

    def __post_init__(self) -> None:
        if not self.states:
            raise ConstraintFixtureError("fixture must contain at least one state")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ConstraintFixtureError("vocab_size must be positive")
        if self.hidden_size is not None and self.hidden_size <= 0:
            raise ConstraintFixtureError("hidden_size must be positive")
        seen: set[int] = set()
        for state in self.states:
            if state.state_id in seen:
                raise ConstraintFixtureError(f"duplicate state_id: {state.state_id}")
            seen.add(state.state_id)
            if self.vocab_size is not None:
                for token_id in state.valid_ids:
                    if token_id >= self.vocab_size:
                        raise ConstraintFixtureError(
                            f"token {token_id} is outside vocab_size={self.vocab_size}"
                        )

    @property
    def primary(self) -> ConstraintState:
        return self.states[0]


def _parse_state(value: Any, index: int) -> ConstraintState:
    if not isinstance(value, dict):
        raise ConstraintFixtureError(f"state {index} must be an object")
    state_id = value.get("state_id", index)
    if not isinstance(state_id, int) or isinstance(state_id, bool) or state_id < 0:
        raise ConstraintFixtureError(f"state {index} has invalid state_id")
    ids = value.get("valid_ids")
    if not isinstance(ids, list):
        raise ConstraintFixtureError(f"state {state_id} must contain a valid_ids list")
    final = value.get("final", False)
    if not isinstance(final, bool):
        raise ConstraintFixtureError(f"state {state_id} has non-boolean final")
    try:
        return ConstraintState(state_id=state_id, valid_ids=TokenSet(ids), final=final)
    except (TypeError, ValueError) as exc:
        raise ConstraintFixtureError(f"state {state_id}: {exc}") from exc


def fixture_from_dict(value: Any) -> ConstraintFixture:
    if not isinstance(value, dict):
        raise ConstraintFixtureError("constraint fixture must be a JSON object")

    vocab_size = value.get("vocab_size")
    hidden_size = value.get("hidden_size")
    for name, item in (("vocab_size", vocab_size), ("hidden_size", hidden_size)):
        if item is not None and (not isinstance(item, int) or isinstance(item, bool) or item <= 0):
            raise ConstraintFixtureError(f"{name} must be a positive integer")

    if "states" in value:
        states_value = value["states"]
        if not isinstance(states_value, list):
            raise ConstraintFixtureError("'states' must be a list")
        states = tuple(_parse_state(item, index) for index, item in enumerate(states_value))
    elif "valid_ids" in value:
        state_id = value.get("state_id", 0)
        states = (_parse_state(value | {"state_id": state_id}, 0),)
    else:
        raise ConstraintFixtureError(
            "unsupported constraint fixture: expected 'valid_ids' or 'states'"
        )

    try:
        return ConstraintFixture(
            states=states,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
    except ConstraintFixtureError:
        raise
    except ValueError as exc:
        raise ConstraintFixtureError(str(exc)) from exc


def load_constraint_fixture(path: str | Path) -> ConstraintFixture:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConstraintFixtureError(f"cannot read fixture: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConstraintFixtureError(f"invalid JSON: {exc}") from exc
    return fixture_from_dict(value)


__all__ = [
    "ConstraintFixture",
    "ConstraintFixtureError",
    "fixture_from_dict",
    "load_constraint_fixture",
]
