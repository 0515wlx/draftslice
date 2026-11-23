from __future__ import annotations

import json

from draftslice.bench.e2e import PromptCase
from draftslice.bench.e2e_sglang import (
    SglangArm,
    SglangBenchConfig,
    _arm,
    _compare,
    _sampling,
    _trace_summary,
)


def test_arm_names_map_to_sglang_algorithms() -> None:
    assert _arm("builtin_mtp").speculative_algorithm == "FROZEN_KV_MTP"
    assert _arm("draftslice_mtp").draftslice
    assert _arm("builtin_dflash").speculative_algorithm == "DFLASH"
    assert _arm("draftslice_dflash").method == "draftslice_dflash"
    assert _arm("draftslice_eagle").method == "draftslice_eagle3"
    assert _arm("builtin_eagle_mtp").speculative_algorithm == "EAGLE"
    assert _arm("draftslice_eagle_mtp").method == "draftslice_mtp"


def test_sampling_encodes_json_schema_as_a_string() -> None:
    case = PromptCase(
        name="tool",
        prompt="call the tool",
        schema={"type": "object", "properties": {"a": {"type": "integer"}}},
    )
    config = SglangBenchConfig(
        model="/models/x", arms=(SglangArm("builtin_mtp"),), temperature=0.0
    )
    params = _sampling(case, config, 16)
    assert params["max_new_tokens"] == 16
    assert params["json_schema"] == json.dumps(case.schema)


def test_trace_summary_counts_reasons(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "contract": {
                            "mode": "full",
                            "fallback_reason": "dflash_no_exact_position",
                        }
                    }
                ),
                json.dumps({"contract": {"mode": "mask", "fallback_reason": None}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = _trace_summary(path)
    assert summary["records"] == 2
    assert summary["reasons"] == {"dflash_no_exact_position": 1}
    assert summary["actions"] == {"full": 1, "mask": 1}


def test_compare_uses_same_run_records() -> None:
    def result(name: str, tok: float, ids):
        return {
            "records": [
                {
                    "case": "tool_call",
                    "batch": 1,
                    "repetition": 0,
                    "tokens_per_s": tok,
                    "token_ids": ids,
                }
            ],
            "summary": {"validity": 1.0, "trace": {}},
        }

    comparison = _compare(
        {
            "builtin_mtp": result("builtin_mtp", 100.0, [1, 2]),
            "draftslice_mtp": result("draftslice_mtp", 125.0, [1, 2]),
        }
    )
    assert comparison["draftslice_mtp"]["throughput_ratio"] == 1.25
    assert comparison["draftslice_mtp"]["greedy_equal"] is True
