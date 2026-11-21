"""Accuracy and coverage metrics for candidate-set methods.

All functions take one candidate set per query and one true top-k list per query.
The true list is the full-head top-k restricted to the measurement universe:
the valid set for constrained modes, or the full vocabulary for free-form modes.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterable, Sequence


class MetricsError(ValueError):
    """Raised when metric inputs are inconsistent."""


def _candidate_sets(candidates: Sequence[Iterable[int]]) -> list[set[int]]:
    return [set(candidate) for candidate in candidates]


def _check_lengths(candidates: Sequence[Iterable[int]], truth: Sequence[Sequence[int]]) -> None:
    if len(candidates) != len(truth):
        raise MetricsError(f"candidate and truth lengths differ: {len(candidates)} != {len(truth)}")
    if not candidates:
        raise MetricsError("metrics need at least one query")


def _top_k(truth_row: Sequence[int], k: int) -> tuple[int, ...]:
    if k <= 0:
        raise MetricsError("k must be positive")
    if len(truth_row) < k:
        raise MetricsError(f"truth row has {len(truth_row)} entries, need {k}")
    return tuple(truth_row[:k])


def top1(candidates: Sequence[Iterable[int]], truth: Sequence[Sequence[int]]) -> float:
    """Fraction of queries whose true top-1 is in the candidate set."""

    _check_lengths(candidates, truth)
    sets = _candidate_sets(candidates)
    hits = sum(
        1 for candidate, truth_row in zip(sets, truth, strict=True) if truth_row[0] in candidate
    )
    return hits / len(sets)


def set_recall(
    candidates: Sequence[Iterable[int]],
    truth: Sequence[Sequence[int]],
    k: int = 5,
) -> float:
    """Mean fraction of the true top-k that appears in the candidate set."""

    _check_lengths(candidates, truth)
    sets = _candidate_sets(candidates)
    total = 0.0
    for candidate, truth_row in zip(sets, truth, strict=True):
        top = _top_k(truth_row, k)
        total += sum(1 for token_id in top if token_id in candidate) / k
    return total / len(sets)


def prefix_coverage(
    candidates: Sequence[Iterable[int]],
    truth: Sequence[Sequence[int]],
    j: int,
) -> float:
    """Fraction of queries where the first ``j`` true tokens are all present."""

    _check_lengths(candidates, truth)
    sets = _candidate_sets(candidates)
    hits = 0
    for candidate, truth_row in zip(sets, truth, strict=True):
        top = _top_k(truth_row, j)
        hits += int(all(token_id in candidate for token_id in top))
    return hits / len(sets)


def all_k(
    candidates: Sequence[Iterable[int]],
    truth: Sequence[Sequence[int]],
    k: int = 5,
) -> float:
    """Fraction of queries where the entire true top-k is present."""

    return prefix_coverage(candidates, truth, k)


def order_accuracy(
    candidates: Sequence[Iterable[int]],
    truth: Sequence[Sequence[int]],
    weights: Sequence[float] | None = None,
) -> float:
    """Weighted prefix coverage.

    By default the weights are uniform over the first ``k`` prefixes. The result
    is a number in ``[0, 1]``.
    """

    _check_lengths(candidates, truth)
    k = len(truth[0])
    if k == 0:
        raise MetricsError("truth rows must not be empty")
    if any(len(row) != k for row in truth):
        raise MetricsError("all truth rows must have the same length")
    if weights is None:
        weights = [1.0 / k] * k
    if len(weights) != k:
        raise MetricsError(f"expected {k} weights, got {len(weights)}")
    if any(weight < 0 for weight in weights):
        raise MetricsError("weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise MetricsError("weights must have positive sum")
    return sum(
        weight / total_weight * prefix_coverage(candidates, truth, j + 1)
        for j, weight in enumerate(weights)
    )


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise MetricsError("cannot take a percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise MetricsError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    values: Sequence[float],
    *,
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a mean."""

    if not values:
        raise MetricsError("bootstrap needs at least one value")
    if not 0.0 < level < 1.0:
        raise MetricsError("level must be in (0, 1)")
    if n_boot <= 0:
        raise MetricsError("n_boot must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    alpha = (1.0 - level) / 2.0
    return _percentile(means, alpha), _percentile(means, 1.0 - alpha)


__all__ = [
    "MetricsError",
    "all_k",
    "bootstrap_ci",
    "order_accuracy",
    "prefix_coverage",
    "set_recall",
    "top1",
]
