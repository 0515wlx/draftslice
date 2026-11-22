"""Stable fallback reason codes.

Every fail-closed decision emits one code. The wire value is the enum value, so
stats, JSONL traces, and dashboards stay compatible with the pre-policy reason
strings that the engine adapters already write. New decisions must reuse a code
instead of inventing a free-form message; :func:`fallback_reason` maps legacy or
unknown strings onto the closest stable code and records them as
:attr:`FallbackReason.UNKNOWN` rather than raising, so telemetry can never crash
a decode step.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA = "reasons/1"


class FallbackReason(str, Enum):
    """Why a step did not take the requested DraftSlice action."""

    # --- policy / profile table (P1) -------------------------------------
    NO_CELL = "no_cell"  # no enabled cell matches this step
    PROFILE_DISABLED = "profile_disabled"  # the matching cell exists, enabled=false
    PROFILES_UNAVAILABLE = "profiles_unavailable"  # table missing or empty
    PROFILE_HASH_MISMATCH = "profile_hash_mismatch"  # artifact integrity failure
    ENGINE_MISMATCH = "engine_mismatch"  # cell is for another engine
    MODE_OFF = "mode_off"  # profile_mode=off
    MODE_OBSERVE = "mode_observe"  # profile_mode=observe executes built-in
    UNCALIBRATED_COST = "uncalibrated_cost"  # scheduler has no fitted cost model
    MIXED_CELL_ACTIONS = "mixed_cell_actions"  # rows want different actions

    # --- capabilities -----------------------------------------------------
    NO_GRAMMAR_STATE = "no_grammar_state"
    NO_VALID_SETS = "no_valid_sets"
    NO_CANDIDATES = "no_candidates"
    SCREENER_EMPTY = "screener_empty"
    LATER_STEP = "later_step"
    GRAPH_LATER_STEP = "graph_later_step"
    FULL_CUDA_GRAPH = "full_cuda_graph"
    GRAPH_MASK_UNAVAILABLE = "graph_mask_unavailable"
    SLICING_DISABLED = "slicing_disabled"
    SHADOW_DISABLED = "shadow_disabled"
    ROUTER_FREEFORM = "router_freeform"
    DUMMY_RUN = "dummy_run"
    UNSUPPORTED_LOGITS_POSTPROCESS = "unsupported_logits_postprocess"
    INVALID_VOCAB_SIZE = "invalid_vocab_size"
    TP_COLLECTIVE_UNAVAILABLE = "tp_collective_unavailable"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"

    # --- head layout / weight --------------------------------------------
    NO_LM_HEAD = "no_lm_head"
    NO_WEIGHT = "no_weight"
    NO_CUDA_WEIGHT = "no_cuda_weight"
    UNSUPPORTED_DTYPE = "unsupported_dtype"
    UNSUPPORTED_HEAD_LAYOUT = "unsupported_head_layout"
    UNSUPPORTED_HEAD = "unsupported_head"
    PADDED_OR_SHARDED_HEAD = "padded_or_sharded_head"
    QUANTIZED_HEAD = "quantized_head"
    UNSUPPORTED_QUANT_SCHEME = "unsupported_quant_scheme"

    # --- execution --------------------------------------------------------
    BUILD_ERROR = "build_error"
    PLAN_COMPILE_ERROR = "plan_compile_error"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Legacy strings that do not already equal a canonical value.
_ALIASES: dict[str, FallbackReason] = {
    "profile_miss": FallbackReason.NO_CELL,
    "no_profile": FallbackReason.NO_CELL,
    "cell_disabled": FallbackReason.PROFILE_DISABLED,
    "disabled_cell": FallbackReason.PROFILE_DISABLED,
    "no_table": FallbackReason.PROFILES_UNAVAILABLE,
    "empty_table": FallbackReason.PROFILES_UNAVAILABLE,
    "hash_mismatch": FallbackReason.PROFILE_HASH_MISMATCH,
    "engine_mismatch": FallbackReason.ENGINE_MISMATCH,
    "observe": FallbackReason.MODE_OBSERVE,
    "observe_mode": FallbackReason.MODE_OBSERVE,
    "uncalibrated": FallbackReason.UNCALIBRATED_COST,
    "no_valid_set": FallbackReason.NO_VALID_SETS,
    "missing_grammar_state": FallbackReason.NO_GRAMMAR_STATE,
    "no_shadow": FallbackReason.SHADOW_DISABLED,
    "freeform": FallbackReason.ROUTER_FREEFORM,
    "slicing_off": FallbackReason.SLICING_DISABLED,
    "dummy": FallbackReason.DUMMY_RUN,
    "step_gt_zero": FallbackReason.LATER_STEP,
    "later_draft_step": FallbackReason.LATER_STEP,
    "tp_collectives": FallbackReason.TP_COLLECTIVE_UNAVAILABLE,
    "collective_unavailable": FallbackReason.TP_COLLECTIVE_UNAVAILABLE,
    "no_head": FallbackReason.NO_LM_HEAD,
    "cpu_weight": FallbackReason.NO_CUDA_WEIGHT,
    "quant_scheme": FallbackReason.UNSUPPORTED_QUANT_SCHEME,
    "build": FallbackReason.BUILD_ERROR,
    "exception": FallbackReason.BUILD_ERROR,
}

_BY_VALUE: dict[str, FallbackReason] = {member.value: member for member in FallbackReason}


def fallback_reason(value: Any) -> FallbackReason:
    """Map a code, enum member, or legacy string onto a stable reason code."""

    if value is None:
        return FallbackReason.NO_CELL
    if isinstance(value, FallbackReason):
        return value
    text = str(value).strip()
    if not text:
        return FallbackReason.UNKNOWN
    if text in _BY_VALUE:
        return _BY_VALUE[text]
    alias = _ALIASES.get(text.lower())
    if alias is not None:
        return alias
    return FallbackReason.UNKNOWN


def reason_code(value: Any) -> str:
    """Wire value for a reason code (never raises)."""

    return fallback_reason(value).value


#: Reasons that are expected on a healthy deployment (a policy decision, not a
#: capability failure). Operations alarms should ignore these by default.
EXPECTED_REASONS: frozenset[FallbackReason] = frozenset(
    {
        FallbackReason.NO_CELL,
        FallbackReason.PROFILE_DISABLED,
        FallbackReason.PROFILES_UNAVAILABLE,
        FallbackReason.MODE_OFF,
        FallbackReason.MODE_OBSERVE,
        FallbackReason.ENGINE_MISMATCH,
        FallbackReason.ROUTER_FREEFORM,
        FallbackReason.LATER_STEP,
        FallbackReason.DUMMY_RUN,
    }
)


@dataclass(frozen=True, slots=True)
class ReasonInfo:
    """Static description of a reason code for docs and dashboards."""

    reason: FallbackReason
    action: str
    expected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "action": self.action,
            "expected": self.expected,
        }


_ACTIONS: dict[FallbackReason, str] = {
    FallbackReason.NO_CELL: "builtin_full",
    FallbackReason.PROFILE_DISABLED: "builtin_full",
    FallbackReason.PROFILES_UNAVAILABLE: "builtin_full",
    FallbackReason.PROFILE_HASH_MISMATCH: "hard_error",
    FallbackReason.ENGINE_MISMATCH: "builtin_full",
    FallbackReason.MODE_OFF: "builtin_full",
    FallbackReason.MODE_OBSERVE: "builtin_full",
    FallbackReason.UNCALIBRATED_COST: "builtin_full",
    FallbackReason.MIXED_CELL_ACTIONS: "builtin_full",
    FallbackReason.NO_GRAMMAR_STATE: "full_head",
    FallbackReason.NO_VALID_SETS: "full_head",
    FallbackReason.NO_CANDIDATES: "full_head",
    FallbackReason.SCREENER_EMPTY: "full_head",
    FallbackReason.LATER_STEP: "full_head",
    FallbackReason.GRAPH_LATER_STEP: "full_head",
    FallbackReason.FULL_CUDA_GRAPH: "full_head",
    FallbackReason.GRAPH_MASK_UNAVAILABLE: "full_head",
    FallbackReason.SLICING_DISABLED: "full_head",
    FallbackReason.SHADOW_DISABLED: "full_head",
    FallbackReason.ROUTER_FREEFORM: "full_head",
    FallbackReason.DUMMY_RUN: "full_head",
    FallbackReason.UNSUPPORTED_LOGITS_POSTPROCESS: "full_head",
    FallbackReason.INVALID_VOCAB_SIZE: "full_head",
    FallbackReason.TP_COLLECTIVE_UNAVAILABLE: "full_head",
    FallbackReason.CAPABILITY_UNSUPPORTED: "full_head",
    FallbackReason.NO_LM_HEAD: "full_head",
    FallbackReason.NO_WEIGHT: "full_head",
    FallbackReason.NO_CUDA_WEIGHT: "full_head",
    FallbackReason.UNSUPPORTED_DTYPE: "full_head",
    FallbackReason.UNSUPPORTED_HEAD_LAYOUT: "full_head",
    FallbackReason.UNSUPPORTED_HEAD: "full_head",
    FallbackReason.PADDED_OR_SHARDED_HEAD: "full_head",
    FallbackReason.QUANTIZED_HEAD: "full_head",
    FallbackReason.UNSUPPORTED_QUANT_SCHEME: "full_head",
    FallbackReason.BUILD_ERROR: "full_head",
    FallbackReason.PLAN_COMPILE_ERROR: "full_head",
    FallbackReason.UNKNOWN: "full_head",
}


def reason_info(value: Any) -> ReasonInfo:
    """Description of a reason code; unknown values degrade to ``UNKNOWN``."""

    reason = fallback_reason(value)
    return ReasonInfo(
        reason=reason,
        action=_ACTIONS.get(reason, "full_head"),
        expected=reason in EXPECTED_REASONS,
    )


__all__ = [
    "EXPECTED_REASONS",
    "SCHEMA",
    "FallbackReason",
    "ReasonInfo",
    "fallback_reason",
    "reason_code",
    "reason_info",
]
