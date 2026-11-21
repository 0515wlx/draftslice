"""Candidate screening, partitions, and split-conformal calibration."""

from __future__ import annotations

from .base import Screener, ScreenError, intersect_universe, validate_candidates
from .conformal import (
    CalibrationError,
    ConformalCalibrator,
    CoverageReport,
    conformal_quantile,
    coverage_report,
    nonconformity_from_candidates,
)
from .partition import (
    PartitionError,
    VocabularyPartition,
    build_partition,
    partition_tokens_for_groups,
)
from .policies import (
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    OracleGroupPolicy,
    RouterPolicy,
    StaticPolicy,
    group_centroids,
    make_group_policy,
)
from .registry import (
    ScreenerFactory,
    available_screeners,
    get_screener,
    register_screener,
)

__all__ = [
    "CalibrationError",
    "ConformalCalibrator",
    "CoverageReport",
    "FullPolicy",
    "GroupPolicy",
    "LowRankPolicy",
    "OracleGroupPolicy",
    "PartitionError",
    "RouterPolicy",
    "ScreenError",
    "Screener",
    "ScreenerFactory",
    "StaticPolicy",
    "VocabularyPartition",
    "available_screeners",
    "build_partition",
    "conformal_quantile",
    "coverage_report",
    "get_screener",
    "group_centroids",
    "intersect_universe",
    "make_group_policy",
    "nonconformity_from_candidates",
    "partition_tokens_for_groups",
    "register_screener",
    "validate_candidates",
]
