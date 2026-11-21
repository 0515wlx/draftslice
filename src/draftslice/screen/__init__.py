"""Candidate screening, partitions, and split-conformal calibration."""

from __future__ import annotations

from .artifacts import (
    ARTIFACT_SCREENER_SCHEMA,
    build_screener_bundle,
    load_screener_bundle,
    screener_calibration,
    screener_from_bundle,
    screener_to_arrays,
    static_rows_from_tokens,
)
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
from .router import (
    RouterTrainingReport,
    group_index,
    group_mass_from_logits,
    group_mass_from_tokens,
    train_router,
)

__all__ = [
    "ARTIFACT_SCREENER_SCHEMA",
    "CalibrationError",
    "ConformalCalibrator",
    "CoverageReport",
    "FullPolicy",
    "GroupPolicy",
    "LowRankPolicy",
    "OracleGroupPolicy",
    "PartitionError",
    "RouterPolicy",
    "RouterTrainingReport",
    "ScreenError",
    "Screener",
    "ScreenerFactory",
    "StaticPolicy",
    "VocabularyPartition",
    "available_screeners",
    "build_partition",
    "build_screener_bundle",
    "conformal_quantile",
    "coverage_report",
    "get_screener",
    "group_centroids",
    "group_index",
    "group_mass_from_logits",
    "group_mass_from_tokens",
    "intersect_universe",
    "load_screener_bundle",
    "make_group_policy",
    "nonconformity_from_candidates",
    "partition_tokens_for_groups",
    "register_screener",
    "screener_calibration",
    "screener_from_bundle",
    "screener_to_arrays",
    "static_rows_from_tokens",
    "train_router",
    "validate_candidates",
]
