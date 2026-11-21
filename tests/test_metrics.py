from __future__ import annotations

import pytest

from draftslice.bench import (
    MetricsError,
    all_k,
    bootstrap_ci,
    order_accuracy,
    prefix_coverage,
    set_recall,
    top1,
)

CANDIDATES = [{1, 2, 3}, {1, 4}]
TRUTH = [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]


def test_top1() -> None:
    assert top1(CANDIDATES, TRUTH) == 1.0
    assert top1([{2}, {4}], TRUTH) == 0.5


def test_set_recall() -> None:
    assert set_recall(CANDIDATES, TRUTH, k=5) == pytest.approx(0.4)


def test_prefix_coverage() -> None:
    assert prefix_coverage(CANDIDATES, TRUTH, 1) == 1.0
    assert prefix_coverage(CANDIDATES, TRUTH, 2) == 0.5
    assert prefix_coverage(CANDIDATES, TRUTH, 3) == 0.5
    assert all_k(CANDIDATES, TRUTH, 5) == 0.0


def test_order_accuracy() -> None:
    assert order_accuracy(CANDIDATES, TRUTH) == pytest.approx((1.0 + 0.5 + 0.5 + 0.0 + 0.0) / 5)
    weighted = order_accuracy(CANDIDATES, TRUTH, weights=[1, 0, 0, 0, 0])
    assert weighted == 1.0


def test_bootstrap_ci() -> None:
    low, high = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_boot=100)
    assert low == 1.0
    assert high == 1.0
    low, high = bootstrap_ci([0.0, 1.0], n_boot=200, seed=1)
    assert low <= high


def test_metric_validation() -> None:
    with pytest.raises(MetricsError):
        top1([], [])
    with pytest.raises(MetricsError):
        top1([{1}], [[1], [2]])
    with pytest.raises(MetricsError):
        set_recall([{1}], [[1]], k=0)
    with pytest.raises(MetricsError):
        order_accuracy([{1}], [[1]], weights=[])
