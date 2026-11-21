"""Split-conformal calibration for candidate coverage.

The guarantee is about the event ``true top-k subset of A``. Calibration scores
are nonconformity values on a held-out split; the finite-sample quantile is

    q = ceil((n + 1) * (1 - delta)) / n            (clipped to 1)

This is the standard split-conformal quantile. It is not a claim that the
candidate set contains every token, and it must be validated on an independent
test split.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from ..plan.types import TokenSet


class CalibrationError(ValueError):
    """Raised when a calibration split or target delta is invalid."""


def conformal_quantile(scores: Iterable[float], delta: float) -> float:
    """Return the split-conformal ``1 - delta`` quantile of nonconformity scores."""

    if not 0.0 < delta < 1.0:
        raise CalibrationError("delta must be in (0, 1)")
    values = sorted(float(score) for score in scores)
    n = len(values)
    if n == 0:
        raise CalibrationError("calibration split is empty")
    rank = math.ceil((n + 1) * (1.0 - delta))
    if rank >= n:
        return values[-1]
    return values[max(0, rank - 1)]


@dataclass(frozen=True, slots=True)
class ConformalCalibrator:
    """Fit one nonconformity threshold per target delta."""

    delta: float
    quantile: float
    n: int
    calibration_id: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.delta < 1.0:
            raise CalibrationError("delta must be in (0, 1)")
        if self.n <= 0:
            raise CalibrationError("calibration size must be positive")

    @classmethod
    def fit(
        cls,
        scores: Iterable[float],
        *,
        delta: float,
        calibration_id: str = "",
    ) -> ConformalCalibrator:
        values = [float(score) for score in scores]
        return cls(
            delta=delta,
            quantile=conformal_quantile(values, delta),
            n=len(values),
            calibration_id=calibration_id,
        )

    def accepts(self, score: float) -> bool:
        return float(score) <= self.quantile


@dataclass(frozen=True, slots=True)
class CoverageReport:
    n: int
    covered: int
    nominal: float
    empirical: float
    gap: float

    def __post_init__(self) -> None:
        if self.n < 0 or self.covered < 0 or self.covered > self.n:
            raise ValueError("invalid coverage counts")
        if not 0.0 <= self.nominal <= 1.0:
            raise ValueError("nominal coverage must be in [0, 1]")

    @property
    def within(self) -> bool:
        return abs(self.gap) <= 0.02


def coverage_report(covered: Iterable[bool], *, delta: float) -> CoverageReport:
    flags = [bool(value) for value in covered]
    n = len(flags)
    hits = sum(flags)
    nominal = 1.0 - delta
    empirical = 0.0 if n == 0 else hits / n
    return CoverageReport(
        n=n,
        covered=hits,
        nominal=nominal,
        empirical=empirical,
        gap=empirical - nominal,
    )


def nonconformity_from_candidates(
    candidates: TokenSet,
    truth: TokenSet,
    *,
    universe_size: int,
) -> float:
    """Fraction of the universe outside the candidate set.

    A smaller value means a larger candidate set; the score is monotone in the
    candidate budget, which is what a conformal threshold needs.
    """

    if universe_size <= 0:
        raise CalibrationError("universe_size must be positive")
    missing = len(truth.difference(candidates))
    return missing / universe_size


__all__ = [
    "CalibrationError",
    "ConformalCalibrator",
    "CoverageReport",
    "conformal_quantile",
    "coverage_report",
    "nonconformity_from_candidates",
]
