"""Fair candidate-set baselines (compatibility re-exports).

The reference policies now live in :mod:`draftslice.screen.policies`, so the
plan compiler and the offline benchmark share one ``Screener`` contract. This
module keeps the historical ``draftslice.bench.baselines`` imports working.
"""

from __future__ import annotations

from ..screen.base import Screener as CandidatePolicy
from ..screen.base import ScreenError as BaselineError
from ..screen.policies import (
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    OracleGroupPolicy,
    RouterPolicy,
    StaticPolicy,
    group_centroids,
    make_group_policy,
)

__all__ = [
    "BaselineError",
    "CandidatePolicy",
    "FullPolicy",
    "GroupPolicy",
    "LowRankPolicy",
    "OracleGroupPolicy",
    "RouterPolicy",
    "StaticPolicy",
    "group_centroids",
    "make_group_policy",
]
