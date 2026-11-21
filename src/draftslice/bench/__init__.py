"""Benchmark utilities.

B1 metric definitions and the offline harness live here, together with the B2
kernel runner, B3 constraint replay, and B4 end-to-end runner. B4 lazily imports
an inference engine; the other tiers stay engine-free.
"""

from __future__ import annotations

from .constraint import (
    CONSTRAINT_SCHEMA,
    ConstraintBenchError,
    ConstraintRecord,
    load_constraint_records,
    run_constraint_replay,
    write_constraint_result,
)
from .e2e import (
    DEFAULT_CASES,
    DEFAULT_METHODS,
    E2E_SCHEMA,
    E2EConfig,
    E2EError,
    PromptCase,
    compare_methods,
    run_matrix,
    run_method,
    write_e2e_result,
)
from .harness import (
    BENCH_SCHEMA,
    BenchmarkResult,
    CandidateRecord,
    HarnessError,
    evaluate,
    load_records,
    write_records,
    write_result,
)
from .metrics import (
    MetricsError,
    all_k,
    bootstrap_ci,
    order_accuracy,
    prefix_coverage,
    set_recall,
    top1,
)

__all__ = [
    "BENCH_SCHEMA",
    "CONSTRAINT_SCHEMA",
    "BenchmarkResult",
    "CandidateRecord",
    "ConstraintBenchError",
    "ConstraintRecord",
    "DEFAULT_CASES",
    "DEFAULT_METHODS",
    "E2E_SCHEMA",
    "E2EConfig",
    "E2EError",
    "HarnessError",
    "MetricsError",
    "PromptCase",
    "all_k",
    "bootstrap_ci",
    "compare_methods",
    "evaluate",
    "load_constraint_records",
    "load_records",
    "order_accuracy",
    "prefix_coverage",
    "run_constraint_replay",
    "run_matrix",
    "run_method",
    "set_recall",
    "top1",
    "write_constraint_result",
    "write_e2e_result",
    "write_records",
    "write_result",
]
