"""Acceptance and prefix-coverage models for scheduling.

The scheduler needs an estimate of how many draft tokens survive verification.
The first model is deliberately conservative and explainable:

* strict plans cover the valid set exactly, so every constrained prefix survives
  the grammar check;
* full plans cover everything;
* calibrated plans supply a prefix-coverage vector from B1/B4 measurements, or
  fall back to the calibrated ``1 - delta`` as a constant proxy.

``expected_tau = sum_j prefix_coverage_j`` is the expected number of accepted
draft positions before the first rejection. It is a proxy, not a promise; the
scheduler must check it against measured tau before using it for a headline
claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..plan.types import GuaranteeKind, HeadPlan, StepPlan


class AcceptanceError(ValueError):
    """Raised when an acceptance model cannot predict a plan."""


@dataclass(frozen=True, slots=True)
class TreeShape:
    depth: int = 1
    width: int = 1

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("tree depth must be positive")
        if self.width <= 0:
            raise ValueError("tree width must be positive")


@dataclass(frozen=True, slots=True)
class AcceptanceEstimate:
    expected_tau: float
    prefix_coverage: tuple[float, ...]
    source: str = "coverage"

    def __post_init__(self) -> None:
        if self.expected_tau < 0:
            raise ValueError("expected_tau must be non-negative")
        for value in self.prefix_coverage:
            if not 0.0 <= value <= 1.0:
                raise ValueError("prefix coverage must be in [0, 1]")


class AcceptanceModel(Protocol):
    def predict(
        self,
        plan: HeadPlan | StepPlan,
        tree: TreeShape | None = None,
        *,
        prefix_coverage: Sequence[float] | None = None,
    ) -> AcceptanceEstimate: ...


def _plan_kind(plan: HeadPlan | StepPlan) -> GuaranteeKind | None:
    if isinstance(plan, HeadPlan):
        return plan.guarantee.kind
    kinds = {node.plan.guarantee.kind for node in plan.nodes}
    if len(kinds) == 1:
        return next(iter(kinds))
    return None


class CoverageAcceptanceModel:
    """Map a plan's guarantee and prefix coverage to expected accepted length."""

    def __init__(
        self,
        *,
        default_coverage: float = 0.95,
        strict_coverage: float = 1.0,
    ) -> None:
        if not 0.0 <= default_coverage <= 1.0:
            raise ValueError("default_coverage must be in [0, 1]")
        if not 0.0 <= strict_coverage <= 1.0:
            raise ValueError("strict_coverage must be in [0, 1]")
        self.default_coverage = default_coverage
        self.strict_coverage = strict_coverage

    def predict(
        self,
        plan: HeadPlan | StepPlan,
        tree: TreeShape | None = None,
        *,
        prefix_coverage: Sequence[float] | None = None,
    ) -> AcceptanceEstimate:
        tree = tree or TreeShape()
        depth = tree.depth

        if prefix_coverage is not None:
            rates = tuple(float(value) for value in prefix_coverage)
            if len(rates) < depth:
                rates = rates + (rates[-1] if rates else self.default_coverage,) * (
                    depth - len(rates)
                )
            rates = rates[:depth]
        else:
            kind = _plan_kind(plan)
            if kind in (GuaranteeKind.FULL, GuaranteeKind.EXACT_VALID_SET):
                base = self.strict_coverage
            elif kind is GuaranteeKind.CALIBRATED:
                delta = _calibrated_delta(plan)
                base = 1.0 - delta if delta is not None else self.default_coverage
            else:
                base = self.default_coverage
            rates = (base,) * depth

        for value in rates:
            if not 0.0 <= value <= 1.0:
                raise AcceptanceError("prefix coverage must be in [0, 1]")
        return AcceptanceEstimate(
            expected_tau=float(sum(rates)),
            prefix_coverage=tuple(rates),
        )


def _calibrated_delta(plan: HeadPlan | StepPlan) -> float | None:
    if isinstance(plan, HeadPlan):
        return plan.guarantee.delta
    deltas = {node.plan.guarantee.delta for node in plan.nodes}
    deltas.discard(None)
    if len(deltas) == 1:
        return next(iter(deltas))
    return None


class StaticAcceptanceModel:
    """Use a measured prefix-coverage vector; useful for offline replay."""

    def __init__(self, prefix_coverage: Sequence[float], *, depth: int | None = None) -> None:
        rates = tuple(float(value) for value in prefix_coverage)
        if not rates:
            raise AcceptanceError("prefix_coverage must not be empty")
        for value in rates:
            if not 0.0 <= value <= 1.0:
                raise AcceptanceError("prefix coverage must be in [0, 1]")
        self.prefix_coverage = rates
        self.depth = len(rates) if depth is None else depth
        if self.depth <= 0 or self.depth > len(rates):
            raise AcceptanceError("depth must be in 1..len(prefix_coverage)")

    def predict(
        self,
        plan: HeadPlan | StepPlan,
        tree: TreeShape | None = None,
        *,
        prefix_coverage: Sequence[float] | None = None,
    ) -> AcceptanceEstimate:
        del plan
        depth = self.depth if tree is None else tree.depth
        rates = (
            tuple(float(value) for value in prefix_coverage)
            if prefix_coverage is not None
            else self.prefix_coverage[:depth]
        )
        if len(rates) < depth:
            fill = rates[-1] if rates else 0.0
            rates = rates + (fill,) * (depth - len(rates))
        return AcceptanceEstimate(
            expected_tau=float(sum(rates[:depth])),
            prefix_coverage=tuple(rates[:depth]),
            source="static",
        )


def coverage_from_measurements(
    prefix_counts: Sequence[int],
    totals: Sequence[int],
) -> tuple[float, ...]:
    """Convert prefix hit counts and totals into a coverage vector."""

    if len(prefix_counts) != len(totals):
        raise AcceptanceError("prefix_counts and totals must have equal length")
    rates = []
    for count, total in zip(prefix_counts, totals, strict=True):
        if count < 0 or total < 0:
            raise AcceptanceError("counts must be non-negative")
        if total == 0:
            continue
        rates.append(min(1.0, count / total))
    if not rates:
        raise AcceptanceError("no non-empty prefix measurements")
    return tuple(rates)


def estimate_from_stats(plan: HeadPlan | StepPlan, stats: dict[str, Any]) -> AcceptanceEstimate:
    """Read a coverage vector from a B1/B4 stats record."""

    raw = stats.get("prefix_coverage")
    if raw is None:
        raise AcceptanceError("stats has no prefix_coverage")
    return CoverageAcceptanceModel().predict(plan, prefix_coverage=raw)


__all__ = [
    "AcceptanceError",
    "AcceptanceEstimate",
    "AcceptanceModel",
    "CoverageAcceptanceModel",
    "StaticAcceptanceModel",
    "TreeShape",
    "coverage_from_measurements",
    "estimate_from_stats",
]
