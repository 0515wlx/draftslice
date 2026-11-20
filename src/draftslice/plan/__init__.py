"""Head-plan types.

A head plan is the executable form of a grammar state's valid-token set. It is
deliberately data-only so it can be serialized, cached, and passed across the
compiler/runtime boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KernelKind = Literal["auto", "slice_gemv", "slice_gemm", "full"]


@dataclass(frozen=True, slots=True)
class Slice:
    """The exact valid-token set for one grammar state."""

    state_id: int
    token_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class HeadPlan:
    """An executable slice of the draft LM head."""

    slice: Slice
    kernel: KernelKind = "auto"
    packed: bool = False
    estimated_bytes: int | None = None


__all__ = ["HeadPlan", "KernelKind", "Slice"]
