from __future__ import annotations

import pytest

from draftslice.runtime.policy import (
    PolicyError,
    ProfilePolicy,
    RowFacts,
    StepFacts,
    policy_from_env,
)
from draftslice.runtime.profile import (
    ProfitCell,
    batch_bucket,
    build_table,
    valid_bucket,
)
from draftslice.runtime.reasons import FallbackReason


def _cell(**overrides) -> ProfitCell:
    data = {
        "model": "Qwen3.5-4B",
        "engine": "vllm",
        "method": "draftslice_mtp",
        "workload": "tool_call",
        "batch_bucket": "1",
        "k": 8,
        "tp": 1,
        "quant": "bf16",
        "graph": "eager",
        "valid_bucket": "65-512",
        "action": "strict_slice",
        "expected_gain": 1.24,
        "measured_p99_ratio": 1.02,
        "manifest_id": "b1-tool-k8",
        "enabled": True,
    }
    data.update(overrides)
    return ProfitCell(**data)


def _facts(**overrides) -> StepFacts:
    data = {
        "engine": "vllm",
        "model": "Qwen3.5-4B",
        "method": "draftslice_mtp",
        "batch": 1,
        "k": 8,
        "tp": 1,
        "quant": "bf16",
        "graph": "eager",
        "rows": (RowFacts(workload="tool_call", valid_size=200),),
    }
    data.update(overrides)
    return StepFacts(**data)


def _table(*, enabled: bool = True, action: str = "strict_slice"):
    return build_table(
        [_cell(enabled=enabled, action=action)],
        table_id="t",
        engine="vllm",
        model="Qwen3.5-4B",
    )


def test_off_and_no_table_are_builtin() -> None:
    off = ProfilePolicy(_table(), profile_mode="off", engine="vllm")
    decision = off.decide(_facts())
    assert decision.action == "builtin" and not decision.executes_slice
    assert decision.reason is FallbackReason.MODE_OFF

    missing = ProfilePolicy(None, profile_mode="observe", engine="vllm")
    assert missing.decide(_facts()).reason is FallbackReason.PROFILES_UNAVAILABLE

    empty = ProfilePolicy(
        build_table([], table_id="empty"), profile_mode="observe", engine="vllm"
    )
    assert empty.decide(_facts()).reason is FallbackReason.PROFILES_UNAVAILABLE


def test_free_form_batch_uses_stock_path() -> None:
    policy = ProfilePolicy(_table(), profile_mode="enforce", engine="vllm")
    facts = _facts(rows=(RowFacts(workload="free", valid_size=0),))
    decision = policy.decide(facts)
    assert decision.action == "builtin"
    assert decision.reason is FallbackReason.ROUTER_FREEFORM


def test_capability_failure_is_a_full_head_with_reason() -> None:
    policy = ProfilePolicy(_table(), profile_mode="enforce", engine="vllm")
    decision = policy.decide(_facts(capabilities={"graph": False}))
    assert decision.action == "full"
    assert decision.reason is FallbackReason.FULL_CUDA_GRAPH

    decision = policy.decide(_facts(capabilities={"head_layout": False}))
    assert decision.action == "full"
    assert decision.reason is FallbackReason.UNSUPPORTED_HEAD_LAYOUT


def test_enforce_hit_slices_only_matching_rows() -> None:
    table = build_table(
        [_cell(batch_bucket="2-8")],
        table_id="t",
        engine="vllm",
        model="Qwen3.5-4B",
    )
    policy = ProfilePolicy(table, profile_mode="enforce", engine="vllm")
    facts = _facts(
        batch=4,
        rows=(
            RowFacts(workload="tool_call", valid_size=200),
            RowFacts(workload="free", valid_size=0),
            RowFacts(workload="json_schema", valid_size=200),
            RowFacts(workload="tool_call", valid_size=200),
        ),
    )
    decision = policy.decide(facts)
    assert decision.executes_slice
    assert decision.action == "slice" and decision.mode == "strict"
    assert decision.cell is not None
    assert decision.cell_hit and decision.sliced_rows == 2
    assert decision.row_mask == (True, False, False, True)


def test_observe_reports_the_cell_but_executes_builtin() -> None:
    policy = ProfilePolicy(_table(), profile_mode="observe", engine="vllm")
    decision = policy.decide(_facts())
    assert not decision.executes_slice
    assert decision.action == "builtin"
    assert decision.reason is FallbackReason.MODE_OBSERVE
    assert decision.cell_hit and decision.cell is not None
    assert decision.row_mask == (False,)


def test_disabled_miss_and_engine_mismatch_are_builtin() -> None:
    disabled = ProfilePolicy(
        _table(enabled=False), profile_mode="enforce", engine="vllm"
    )
    decision = disabled.decide(_facts())
    assert decision.action == "builtin"
    assert decision.reason is FallbackReason.PROFILE_DISABLED
    assert decision.cell is not None

    miss = ProfilePolicy(_table(), profile_mode="enforce", engine="vllm")
    decision = miss.decide(_facts(k=4))
    assert decision.reason is FallbackReason.NO_CELL
    assert decision.cell is None

    other = ProfilePolicy(_table(), profile_mode="enforce", engine="sglang")
    decision = other.decide(_facts(engine="sglang"))
    assert decision.reason is FallbackReason.ENGINE_MISMATCH


def test_mixed_actions_do_not_build_a_second_decision_table() -> None:
    table = build_table(
        [
            _cell(workload="tool_call", action="strict_slice"),
            _cell(workload="json_schema", action="mask", valid_bucket="65-512"),
        ],
        table_id="t",
        engine="vllm",
        model="Qwen3.5-4B",
    )
    policy = ProfilePolicy(table, profile_mode="enforce", engine="vllm")
    facts = _facts(
        rows=(
            RowFacts(workload="tool_call", valid_size=200),
            RowFacts(workload="json_schema", valid_size=300),
        )
    )
    assert policy.decide(facts).reason is FallbackReason.MIXED_CELL_ACTIONS


def test_enforce_requires_a_table() -> None:
    with pytest.raises(PolicyError):
        ProfilePolicy(None, profile_mode="enforce", engine="vllm")
    with pytest.raises(PolicyError):
        ProfilePolicy(
            build_table([], table_id="empty"), profile_mode="enforce", engine="vllm"
        )
    with pytest.raises(PolicyError):
        ProfilePolicy(_table(), profile_mode="on", engine="vllm")
    with pytest.raises(PolicyError):
        _facts(batch=0)
    with pytest.raises(PolicyError):
        RowFacts(valid_size=-1)


def test_policy_from_env_loads_the_hashed_table(tmp_path, monkeypatch) -> None:
    assert policy_from_env("vllm", environ={}) is None

    table = _table()
    path = tmp_path / "profiles.json"
    table.save(path)
    env = {
        "DRAFTSLICE_PROFILE_TABLE": str(path),
        "DRAFTSLICE_PROFILE_MODE": "enforce",
    }
    policy = policy_from_env(
        "vllm", model="Qwen3.5-4B", method="draftslice_mtp", environ=env
    )
    assert policy is not None
    assert policy.profile_mode == "enforce"
    decision = policy.decide(_facts())
    assert decision.executes_slice

    # A tampered artifact is a hard error, never a silent built-in.
    document = path.read_text()
    path.write_text(document.replace('"expected_gain": 1.24', '"expected_gain": 2.0'))
    from draftslice.runtime.profile import ProfileHashError

    with pytest.raises(ProfileHashError):
        policy_from_env("vllm", environ=env)

    with pytest.raises(PolicyError):
        policy_from_env(
            "vllm",
            environ={
                "DRAFTSLICE_PROFILE_TABLE": str(path),
                "DRAFTSLICE_PROFILE_MODE": "sometimes",
            },
        )


def test_bucket_labels_match_profile_helpers() -> None:
    policy = ProfilePolicy(_table(), profile_mode="enforce", engine="vllm")
    assert policy.classify(workload="tool_call", valid_size=200, batch=1) == (
        batch_bucket(1),
        valid_bucket(200),
    )


def test_workload_mapping_is_explicit() -> None:
    from draftslice.integrations.base import (
        AdapterPolicyError,
        classify_structured_workload,
        workload_config_from_env,
    )

    assert classify_structured_workload("json") == "json_schema"
    # vLLM hands over an enum member, not a bare string.
    assert (
        classify_structured_workload("StructuredOutputOptions.JSON")
        == "json_schema"
    )
    assert classify_structured_workload("regex") == "regex"
    assert classify_structured_workload("unknown-format") == "structured"
    # Engines only expose the grammar format; a deployment must opt into the
    # benchmark class, otherwise a tool cell could silently claim JSON traffic.
    assert (
        classify_structured_workload("json", mapping={"json": "tool_call"})
        == "tool_call"
    )
    assert (
        workload_config_from_env(
            {
                "DRAFTSLICE_WORKLOAD": "tool_call",
                "DRAFTSLICE_WORKLOAD_MAP": '{"json": "tool_call", "json_object": "tool_call"}',
            }
        )[1]
        == {"json": "tool_call", "json_object": "tool_call"}
    )
    assert workload_config_from_env({}) == ("structured", {})
    with pytest.raises(AdapterPolicyError):
        workload_config_from_env({"DRAFTSLICE_WORKLOAD_MAP": "not json"})
    with pytest.raises(AdapterPolicyError):
        workload_config_from_env({"DRAFTSLICE_WORKLOAD_MAP": '{"json": 3}'})


def test_filter_valid_sets_keeps_exact_rows_and_fails_closed() -> None:
    from draftslice.integrations.base import (
        AdapterPolicyError,
        filter_valid_sets,
    )
    from draftslice.plan import TokenSet

    valid = TokenSet([1, 2, 3])
    filtered = filter_valid_sets([valid, TokenSet([4])], (True, False))
    assert filtered[0] is valid  # exact candidate set, not a copy
    assert filtered[1] is None  # unapproved row -> full head
    with pytest.raises(AdapterPolicyError):
        filter_valid_sets([valid], (True, False))
