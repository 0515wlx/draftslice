"""Draft-head runtime interfaces and the reference executor."""

from __future__ import annotations

from typing import Any, Protocol

from ..plan import HeadPlan


class DraftHeadExecutor(Protocol):
    """Execute a head plan against a draft hidden state."""

    def execute(self, hidden: Any, plan: HeadPlan) -> Any: ...


from .acceptance import (  # noqa: E402
    AcceptanceError,
    AcceptanceEstimate,
    AcceptanceModel,
    CoverageAcceptanceModel,
    StaticAcceptanceModel,
    TreeShape,
    coverage_from_measurements,
    estimate_from_stats,
)
from .config import (  # noqa: E402
    AdapterConfig,
    ConfigError,
    EngineAdapterConfig,
    ObservabilityConfig,
    ProductionConfig,
    SafetyConfig,
    ServerConfig,
)
from .executor import (  # noqa: E402
    ExecutionResult,
    ProductionExecutor,
    ProductionExecutorError,
    ProductionNodeResult,
    ProductionResult,
    ReferenceExecutor,
)
from .plan_session import (  # noqa: E402
    PlanSession,
    PlanSessionConfig,
    PlanSessionError,
    PlanSessionResult,
    validate_strict_step,
)
from .policy import (  # noqa: E402
    PolicyDecision,
    PolicyError,
    ProfilePolicy,
    RowFacts,
    StepFacts,
    policy_from_env,
)
from .profile import (  # noqa: E402
    ProfileError,
    ProfileHashError,
    ProfileMatch,
    ProfileParseError,
    ProfileTable,
    ProfileValidationError,
    ProfitCell,
    batch_bucket,
    build_table,
    valid_bucket,
)
from .reasons import FallbackReason, fallback_reason, reason_code, reason_info  # noqa: E402
from .scheduler import Schedule, Scheduler, StepContext, StepScheduler  # noqa: E402
from .trace import (  # noqa: E402
    StepTrace,
    TraceCache,
    TraceDecision,
    TraceError,
    TraceIdentity,
    TraceOutcome,
    TraceTiming,
)
from .union import UnionExecutor, UnionResult  # noqa: E402

__all__ = [
    "AcceptanceError",
    "AdapterConfig",
    "ConfigError",
    "EngineAdapterConfig",
    "ObservabilityConfig",
    "ProductionConfig",
    "SafetyConfig",
    "ServerConfig",
    "AcceptanceEstimate",
    "AcceptanceModel",
    "CoverageAcceptanceModel",
    "DraftHeadExecutor",
    "ExecutionResult",
    "FallbackReason",
    "PlanSession",
    "PlanSessionConfig",
    "PlanSessionError",
    "PlanSessionResult",
    "ProductionExecutor",
    "ProductionExecutorError",
    "ProductionNodeResult",
    "ProductionResult",
    "ProfileError",
    "ProfileHashError",
    "ProfileMatch",
    "ProfileParseError",
    "ProfileTable",
    "ProfileValidationError",
    "ProfitCell",
    "PolicyDecision",
    "PolicyError",
    "ProfilePolicy",
    "ReferenceExecutor",
    "RowFacts",
    "Schedule",
    "Scheduler",
    "StaticAcceptanceModel",
    "StepContext",
    "StepFacts",
    "StepScheduler",
    "StepTrace",
    "TraceCache",
    "TraceDecision",
    "TraceError",
    "TraceIdentity",
    "TraceOutcome",
    "TraceTiming",
    "TreeShape",
    "UnionExecutor",
    "UnionResult",
    "batch_bucket",
    "build_table",
    "coverage_from_measurements",
    "estimate_from_stats",
    "fallback_reason",
    "policy_from_env",
    "reason_code",
    "reason_info",
    "validate_strict_step",
    "valid_bucket",
]
