from __future__ import annotations

import json

import pytest

from draftslice.bench.constraint import (
    ConstraintBenchError,
    ConstraintRecord,
    load_constraint_records,
    run_constraint_replay,
)
from draftslice.plan import PlanPolicy


def test_load_json_states_and_jsonl(tmp_path) -> None:
    path = tmp_path / "states.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {"state_id": 0, "valid_ids": [1, 2, 3]},
                    {"state_id": 1, "valid_ids": [4, 5]},
                ]
            }
        )
    )
    records = load_constraint_records(path)
    assert [record.state_id for record in records] == [0, 1]
    assert records[0].valid_ids == (1, 2, 3)

    lines = tmp_path / "states.jsonl"
    lines.write_text(
        "\n".join(
            [
                '{"valid_ids": [1]}',
                '{"state_id": 4, "valid_ids": [2, 3], "next_token": 2}',
            ]
        )
        + "\n"
    )
    records = load_constraint_records(lines)
    assert [record.state_id for record in records] == [0, 4]
    assert records[1].next_token == 2


def test_replay_covers_valid_sets_and_uses_cache() -> None:
    records = [
        ConstraintRecord(0, (1, 3, 5)),
        ConstraintRecord(0, (1, 3, 5)),
        ConstraintRecord(1, (2, 4, 6, 8)),
    ]
    result = run_constraint_replay(records, vocab_size=16, hidden_size=8)
    summary = result["summary"]
    assert summary["violations"] == 0
    assert summary["states"] == 3
    assert summary["unique_states"] == 2
    assert summary["cache"]["hits"] == 1
    assert summary["cache"]["misses"] == 2
    assert summary["valid_mean"] == pytest.approx(10 / 3)
    assert result["states"][0]["execution"] == "slice_gemv"


def test_replay_falls_back_over_budget_without_dropping_tokens() -> None:
    records = [ConstraintRecord(0, (1, 2, 3, 4, 5))]
    result = run_constraint_replay(
        records,
        vocab_size=8,
        hidden_size=4,
        max_candidates=2,
    )
    summary = result["summary"]
    assert summary["violations"] == 0
    assert summary["fallback_rate"] == 1.0
    assert result["states"][0]["execution"] == "full"


def test_full_policy_is_not_a_strict_plan() -> None:
    records = [ConstraintRecord(0, (1, 2))]
    result = run_constraint_replay(
        records,
        vocab_size=8,
        hidden_size=4,
        policy=PlanPolicy(mode="full"),
    )
    assert result["states"][0]["execution"] == "full"
    assert result["summary"]["violations"] == 0


def test_bad_replay_inputs(tmp_path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("")
    with pytest.raises(ConstraintBenchError):
        load_constraint_records(empty)

    bad = tmp_path / "bad.json"
    bad.write_text('[{"state_id": 0}]')
    with pytest.raises(ConstraintBenchError):
        load_constraint_records(bad)

    with pytest.raises(ConstraintBenchError):
        run_constraint_replay([], vocab_size=8, hidden_size=4)
