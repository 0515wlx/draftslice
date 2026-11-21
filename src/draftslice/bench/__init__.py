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
from .factorial import (
    FACTORIAL_SCHEMA,
    FactorialCase,
    cell_metrics,
    load_cases,
    run_factorial,
    write_factorial,
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
from .release import (
    KNOWN_SCHEMAS,
    RELEASE_SCHEMA,
    ReleaseError,
    ReleaseManifest,
    build_manifest,
    build_release,
    compare_manifests,
    file_hash,
    load_result,
    render_report,
    validate_result,
    verify_manifest,
)

__all__ = [
    "BENCH_SCHEMA",
    "CONSTRAINT_SCHEMA",
    "FACTORIAL_SCHEMA",
    "KNOWN_SCHEMAS",
    "RELEASE_SCHEMA",
    "BenchmarkResult",
    "CandidateRecord",
    "ConstraintBenchError",
    "ConstraintRecord",
    "DEFAULT_CASES",
    "DEFAULT_METHODS",
    "E2E_SCHEMA",
    "E2EConfig",
    "E2EError",
    "FactorialCase",
    "HarnessError",
    "MetricsError",
    "PromptCase",
    "ReleaseError",
    "ReleaseManifest",
    "all_k",
    "bootstrap_ci",
    "build_manifest",
    "build_release",
    "cell_metrics",
    "compare_manifests",
    "compare_methods",
    "evaluate",
    "file_hash",
    "load_cases",
    "load_constraint_records",
    "load_records",
    "load_result",
    "order_accuracy",
    "prefix_coverage",
    "render_report",
    "run_constraint_replay",
    "run_factorial",
    "run_matrix",
    "run_method",
    "set_recall",
    "top1",
    "validate_result",
    "verify_manifest",
    "write_constraint_result",
    "write_e2e_result",
    "write_factorial",
    "write_records",
    "write_result",
]
