from __future__ import annotations

import json

import pytest

from draftslice.bench import (
    CandidateRecord,
    HarnessError,
    evaluate,
    load_records,
    write_records,
    write_result,
)


def records() -> list[CandidateRecord]:
    return [
        CandidateRecord(
            candidates=(3, 1, 2),
            truth=(1, 2, 3, 4, 5),
            valid=(1, 2, 3, 4, 5),
            domain="a",
        ),
        CandidateRecord(
            candidates=(4,),
            truth=(4, 5, 6, 7, 8),
            valid=(4, 5, 6, 7, 8),
            domain="b",
        ),
    ]


def test_record_normalizes_candidates() -> None:
    record = CandidateRecord(candidates=(3, 1, 3), truth=(3,), valid=(2, 1, 2))
    assert record.candidates == (1, 3)
    assert record.valid == (1, 2)


def test_record_validation() -> None:
    with pytest.raises(HarnessError):
        CandidateRecord(candidates=(1,), truth=())
    with pytest.raises(HarnessError):
        CandidateRecord(candidates=(-1,), truth=(1,))


def test_evaluate_metrics() -> None:
    result = evaluate(records(), k=5)
    assert result.n == 2
    assert result.metrics["top1"] == 1.0
    assert result.metrics["set_recall@5"] == pytest.approx(0.4)
    assert result.metrics["OA"] == pytest.approx(0.4)
    assert result.metrics["m_mean"] == 2.0
    assert result.metrics["valid_recall"] == pytest.approx(0.4)
    assert result.metrics["valid_precision"] == 1.0
    assert result.metrics["plan_equals_valid"] == 0.0
    assert set(result.by_domain) == {"a", "b"}
    assert result.warnings == ()


def test_evaluate_rejects_short_truth() -> None:
    with pytest.raises(HarnessError):
        evaluate([CandidateRecord(candidates=(1,), truth=(1,))], k=5)


def test_record_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    write_records(path, records())
    loaded = load_records(path)
    assert loaded[0].candidates == (1, 2, 3)
    assert loaded[1].domain == "b"


def test_result_json(tmp_path) -> None:
    result = evaluate(records(), k=5)
    path = tmp_path / "result.json"
    write_result(path, result)
    value = json.loads(path.read_text())
    assert value["schema_version"] == "bench/1"
    assert value["n"] == 2
    assert value["metrics"]["top1"] == 1.0


def test_load_rejects_bad_jsonl(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(HarnessError):
        load_records(path)
