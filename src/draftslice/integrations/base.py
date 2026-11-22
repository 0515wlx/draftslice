"""Shared adapter helpers for the production profile policy.

Both engine adapters import this module instead of growing a second copy of the
decision table. It owns:

* :class:`AdapterPolicy` — the resolved policy (or ``None`` for the legacy
  research override) plus the table hash for telemetry;
* :func:`load_adapter_policy` — hard-failing table/hash load;
* :func:`classify_structured_workload` — grammar request type -> cell workload;
* :func:`filter_valid_sets` — apply a per-row policy mask;
* helper math for ``row_share`` so it is never confused with the measured head
  share.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet
from ..runtime.policy import PolicyDecision, ProfilePolicy, policy_from_env
from ..runtime.trace import TraceRegistry


class AdapterPolicyError(RuntimeError):
    """The adapter cannot build a safe policy from the configuration."""


@dataclass(frozen=True, slots=True)
class AdapterPolicy:
    """Resolved policy for one engine/adapter instance."""

    policy: ProfilePolicy | None
    profile_mode: str
    table_path: str = ""
    table_hash: str = ""

    @property
    def uses_table(self) -> bool:
        return self.policy is not None

    @property
    def executes_slice(self) -> bool:
        return bool(
            self.policy is not None
            and self.profile_mode == "enforce"
            and self.policy.enabled
        )

    @property
    def hold_reason(self) -> Any | None:
        """Reason to hold the adapter on the stock path for the whole run.

        ``observe``/``off`` and a table with no enabled cell can never slice, so
        the adapter must not pay the per-step shadow/valid-set cost. Returning
        ``None`` means the step policy decides.
        """

        if self.policy is None:
            return None
        from ..runtime.reasons import FallbackReason

        if self.profile_mode == "off":
            return FallbackReason.MODE_OFF
        if self.profile_mode == "observe":
            return FallbackReason.MODE_OBSERVE
        if not self.policy.enabled:
            return FallbackReason.PROFILE_DISABLED
        return None


def load_adapter_policy(
    engine: str,
    *,
    model: str | None = None,
    method: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AdapterPolicy:
    """Build the adapter policy from the environment.

    * ``DRAFTSLICE_PROFILE_TABLE`` set -> the table is authoritative; a missing
      file, malformed document, or hash mismatch raises (never silent fallback).
    * no table -> legacy research behaviour (the adapter keeps its
      ``DRAFTSLICE_{VLLM,SGLANG}_SLICE`` switch) and the production default is
      built-in.
    """

    env = os.environ if environ is None else environ
    mode = env.get("DRAFTSLICE_PROFILE_MODE", "observe").strip() or "observe"
    try:
        policy = policy_from_env(
            engine, model=model, method=method, environ=env
        )
    except Exception as exc:  # table/hash/config errors are fatal by design
        raise AdapterPolicyError(f"cannot load profile policy: {exc}") from exc
    if policy is None:
        return AdapterPolicy(policy=None, profile_mode=mode)
    table = policy.table
    assert table is not None
    return AdapterPolicy(
        policy=policy,
        profile_mode=mode,
        table_path=env.get("DRAFTSLICE_PROFILE_TABLE", ""),
        table_hash=table.hash,
    )


_WORKLOAD_BY_REQUEST_TYPE = {
    "json": "json_schema",
    "json_object": "json_schema",
    "json_schema": "json_schema",
    "regex": "regex",
    "regex_pattern": "regex",
    "grammar": "grammar",
    "ebnf": "grammar",
    "structural_tag": "structured",
    "choice": "choice",
}


def classify_structured_workload(
    request_type: Any,
    *,
    mapping: Mapping[str, str] | None = None,
    default: str = "structured",
) -> str:
    """Map an engine structured-output request type onto a cell workload.

    Engines expose the grammar *format* (``json``/``regex``/``grammar``), not the
    benchmark workload class. A deployment that calls all JSON-schema traffic
    "tool_call" (or "json_schema") sets ``DRAFTSLICE_WORKLOAD_MAP``. Without a
    mapping, only the generic labels below are used, so a cell cannot silently
    claim the wrong workload.
    """

    text = str(request_type or "").strip().lower()
    # vLLM passes a StructuredOutputOptions enum member; ``str`` renders it as
    # "StructuredOutputOptions.JSON", so keep the bare format name.
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if not text:
        return default
    override = (mapping or {}).get(text)
    if override:
        return str(override)
    return _WORKLOAD_BY_REQUEST_TYPE.get(text, default)


def workload_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Read the deployment workload label and the request-type override map."""

    env = os.environ if environ is None else environ
    default = (env.get("DRAFTSLICE_WORKLOAD", "") or "").strip() or "structured"
    raw = (env.get("DRAFTSLICE_WORKLOAD_MAP", "") or "").strip()
    if not raw:
        return default, {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterPolicyError(
            f"DRAFTSLICE_WORKLOAD_MAP is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise AdapterPolicyError("DRAFTSLICE_WORKLOAD_MAP must be a JSON object")
    mapping: dict[str, str] = {}
    for key, label in value.items():
        if not isinstance(label, str) or not label:
            raise AdapterPolicyError(
                f"DRAFTSLICE_WORKLOAD_MAP[{key!r}] must be a non-empty string"
            )
        mapping[str(key).strip().lower()] = label
    return default, mapping


def filter_valid_sets(
    valid_sets: Sequence[TokenSet | None], row_mask: Sequence[bool]
) -> list[TokenSet | None]:
    """Drop rows the policy did not approve; the rest use the full head."""

    if len(valid_sets) != len(row_mask):
        raise AdapterPolicyError("row mask length does not match the batch")
    return [
        token_set if approved else None
        for token_set, approved in zip(valid_sets, row_mask, strict=True)
    ]


def row_share(candidates: int, rows: int, vocab_size: int) -> float:
    """Fraction of vocabulary rows a slice reads (never a timing claim)."""

    if rows <= 0 or vocab_size <= 0:
        return 0.0
    return float(candidates) / float(rows * vocab_size)


def policy_reason(decision: PolicyDecision | None) -> str | None:
    if decision is None:
        return None
    return decision.reason.value


class TraceSampler:
    """Deterministic 1-in-N step-trace sampling (no wall-clock randomness)."""

    def __init__(self, sample: float = 1.0) -> None:
        if not (0.0 <= sample <= 1.0):
            raise AdapterPolicyError("trace sample must be in [0, 1]")
        self.sample = float(sample)
        self.period = 0 if sample <= 0.0 else max(1, int(round(1.0 / sample)))
        self.seen = 0

    def __call__(self) -> bool:
        if self.period == 0:
            return False
        self.seen += 1
        return (self.seen - 1) % self.period == 0


def trace_sample_from_env(environ: Mapping[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    raw = env.get("DRAFTSLICE_TRACE_SAMPLE", "0.01").strip() or "0.01"
    try:
        value = float(raw)
    except ValueError as exc:
        raise AdapterPolicyError(
            f"DRAFTSLICE_TRACE_SAMPLE must be a float, got {raw!r}"
        ) from exc
    if not (0.0 <= value <= 1.0):
        raise AdapterPolicyError("DRAFTSLICE_TRACE_SAMPLE must be in [0, 1]")
    return value


def make_registry(capacity: int = 1024) -> TraceRegistry:
    return TraceRegistry(capacity=capacity)


__all__ = [
    "AdapterPolicy",
    "AdapterPolicyError",
    "classify_structured_workload",
    "filter_valid_sets",
    "load_adapter_policy",
    "make_registry",
    "policy_reason",
    "row_share",
    "trace_sample_from_env",
    "workload_config_from_env",
    "TraceSampler",
]
