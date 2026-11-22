from __future__ import annotations

import json

import pytest

from draftslice.runtime.reasons import (
    EXPECTED_REASONS,
    FallbackReason,
    fallback_reason,
    reason_code,
    reason_info,
)
from draftslice.runtime.trace import (
    SCHEMA,
    StepTrace,
    TraceDecision,
    TraceError,
    TraceIdentity,
    TraceOutcome,
    TraceTiming,
)


def _trace(**overrides) -> StepTrace:
    data = {
        "identity": TraceIdentity(
            engine="vllm",
            model="Qwen3.5-4B",
            tp=1,
            quant="bf16",
            graph="eager",
            workload="tool_call",
            config_hash="sha256:abc",
        ),
        "decision": TraceDecision(
            mode="strict",
            action="slice",
            profile_mode="enforce",
            source="draftslice_mtp",
            k=8,
            cell_id="Qwen3.5-4B|vllm|draftslice_mtp|tool_call|1|k8|tp1|bf16|eager|65-512",
            cell_hit=True,
            candidates=436,
            rows=1,
            row_share=436 / 151936,
            measured_head_share=0.031,
        ),
        "timing": TraceTiming(plan_compile_us=12.5, kernel_us=88.0, step_us=2400.0),
        "outcome": TraceOutcome(accepted=9, tau=7.67, validity=True),
        "step": 3,
    }
    data.update(overrides)
    return StepTrace(**data)


def test_default_trace_is_a_builtin_fallback() -> None:
    trace = StepTrace(
        identity=TraceIdentity(engine="sglang", model="m"),
        decision=TraceDecision(),
    )
    assert trace.schema == SCHEMA
    assert trace.is_fallback
    assert trace.decision.reason is None
    assert trace.to_dict()["schema"] == "trace/1"


def test_jsonl_roundtrip_and_stable_key_order() -> None:
    trace = _trace()
    line = trace.to_jsonl()
    assert json.loads(line) == trace.to_dict()
    # sort_keys keeps the wire format stable across processes.
    assert line.startswith('{"cache"')
    restored = StepTrace.from_jsonl(line)
    assert restored == trace


def test_schema_mismatch_is_rejected() -> None:
    document = _trace().to_dict()
    document["schema"] = "trace/2"
    with pytest.raises(TraceError):
        StepTrace.from_dict(document)
    with pytest.raises(TraceError):
        StepTrace.from_jsonl("{oops")


def test_validation_rules() -> None:
    with pytest.raises(TraceError):
        TraceDecision(mode="slice")
    with pytest.raises(TraceError):
        TraceDecision(action="strict_slice")
    with pytest.raises(TraceError):
        TraceDecision(profile_mode="on")
    with pytest.raises(TraceError):
        TraceDecision(row_share=1.5)
    with pytest.raises(TraceError):
        TraceDecision(measured_head_share=-0.1)
    with pytest.raises(TraceError):
        TraceTiming(step_us=-1.0)
    with pytest.raises(TraceError):
        TraceOutcome(tau=-1.0)
    with pytest.raises(TraceError):
        TraceIdentity(engine="trtllm", model="m")
    with pytest.raises(TraceError):
        TraceIdentity(engine="vllm", model="m", tp=0)
    with pytest.raises(TraceError):
        StepTrace(
            identity=TraceIdentity(engine="vllm", model="m"),
            decision=TraceDecision(),
            step=-1,
        )


def test_reason_codes_are_normalized_and_stable() -> None:
    assert reason_code("no_valid_set") == "no_valid_sets"
    assert reason_code("observe_mode") == "mode_observe"
    assert reason_code("something-new") == "unknown"
    assert fallback_reason(FallbackReason.TP_COLLECTIVE_UNAVAILABLE) is (
        FallbackReason.TP_COLLECTIVE_UNAVAILABLE
    )
    assert reason_code(None) == FallbackReason.NO_CELL.value

    trace = _trace(
        decision=TraceDecision(mode="full", action="full", reason="no_valid_set")
    )
    assert trace.decision.reason == "no_valid_sets"
    assert trace.decision.reason_enum is FallbackReason.NO_VALID_SETS


def test_reason_info_classifies_expected_decisions() -> None:
    info = reason_info("no_cell")
    assert info.action == "builtin_full"
    assert info.expected
    assert FallbackReason.NO_CELL in EXPECTED_REASONS

    capability = reason_info("unsupported_head_layout")
    assert capability.action == "full_head"
    assert not capability.expected
    assert reason_info("hash_mismatch").action == "hard_error"


def test_row_share_is_separate_from_head_share() -> None:
    trace = _trace()
    document = trace.to_dict()
    assert document["decision"]["row_share"] == pytest.approx(436 / 151936)
    assert document["decision"]["measured_head_share"] == pytest.approx(0.031)
    assert "row_share" in document["decision"]
    assert "measured_head_share" in document["decision"]


def test_trace_registry_is_bounded_and_serializable() -> None:
    from draftslice.runtime.trace import TraceRegistry

    registry = TraceRegistry(capacity=2)
    assert len(registry) == 0
    for step in range(3):
        registry.add(_trace(step=step))
    assert len(registry) == 2
    assert registry.recorded == 3
    assert [trace.step for trace in registry.recent()] == [1, 2]
    assert [trace.step for trace in registry.recent(1)] == [2]
    payload = registry.to_dict()
    assert payload["capacity"] == 2
    assert payload["recorded"] == 3
    assert len(payload["recent"]) == 2
    lines = list(registry.jsonl())
    assert len(lines) == 2 and all(StepTrace.from_jsonl(line) for line in lines)
    registry.clear()
    assert len(registry) == 0 and registry.recorded == 0


def test_trace_sampler_is_deterministic() -> None:
    from draftslice.integrations.base import TraceSampler

    sampler = TraceSampler(1.0)
    assert all(sampler() for _ in range(5))
    quarter = TraceSampler(0.25)
    assert [quarter() for _ in range(8)] == [True, False, False, False] * 2
    never = TraceSampler(0.0)
    assert not any(never() for _ in range(5))
    from draftslice.integrations.base import AdapterPolicyError

    with pytest.raises(AdapterPolicyError):
        TraceSampler(1.5)
