from __future__ import annotations

import json

import pytest

from draftslice.bench.e2e import (
    DEFAULT_CASES,
    E2EConfig,
    E2EError,
    _physical_gpu_index,
    compare_methods,
    metric_map,
    run_method,
)


class _Sample:
    def __init__(self, name, value=None, values=None):
        self.name = name
        self.value = value
        self.values = values


def test_metric_map_sums_labels_and_vectors() -> None:
    samples = [
        _Sample("a", value=2.0),
        _Sample("a", value=3.0),
        _Sample("b", values=[1.0, 4.0]),
        _Sample("bad"),
    ]
    assert metric_map(samples) == {"a": 5.0, "b": 5.0}


def test_config_validates_and_selects_cases() -> None:
    config = E2EConfig(model="m", batch_sizes=(1,), case_names=("json_person", "math"))
    names = tuple(case.name for case in config.cases())
    assert names == ("json_person", "math")
    with pytest.raises(ValueError):
        E2EConfig(model="m", case_names=("missing",))
    with pytest.raises(ValueError):
        E2EConfig(model="m", batch_sizes=())
    with pytest.raises(ValueError):
        E2EConfig(model="m", repetitions=0)
    with pytest.raises(ValueError):
        E2EConfig(model="m", tensor_parallel_size=0)
    assert E2EConfig(model="m", tensor_parallel_size=2).tensor_parallel_size == 2


def test_calibrated_modes_require_and_export_a_screener() -> None:
    with pytest.raises(ValueError):
        E2EConfig(model="m", mode="constrained_calibrated")
    with pytest.raises(ValueError):
        E2EConfig(model="m", mode="strict", screener="static")
    # screener_args is allowed with a name, not with an artifact alone
    E2EConfig(
        model="m",
        mode="freeform_calibrated",
        screener="static",
        screener_args='{"rows":[1]}',
    )
    with pytest.raises(ValueError):
        E2EConfig(
            model="m",
            mode="freeform_calibrated",
            screener_artifact="/tmp/x",
            screener_args="{}",
        )

    constrained = E2EConfig(
        model="m",
        mode="constrained_calibrated",
        screener="static",
        screener_args='{"rows":[1]}',
    )
    assert constrained.screener_env() == {
        "DRAFTSLICE_VLLM_SLICE": "1",
        "DRAFTSLICE_VLLM_SCREENER": "static",
        "DRAFTSLICE_VLLM_SCREENER_ARGS": '{"rows":[1]}',
        "DRAFTSLICE_VLLM_SCREENER_UNIVERSE": "valid",
    }
    freeform = E2EConfig(
        model="m",
        mode="freeform_calibrated",
        screener_artifact="/tmp/bundle",
    )
    assert freeform.screener_env()["DRAFTSLICE_VLLM_SCREENER_UNIVERSE"] == "vocab"
    assert E2EConfig(model="m").screener_env() == {}


def test_default_cases_are_unique_and_well_formed() -> None:
    names = [case.name for case in DEFAULT_CASES]
    assert len(names) == len(set(names))
    for case in DEFAULT_CASES:
        assert case.prompt
        assert case.max_tokens > 0


def test_validation_accepts_and_rejects_structured_outputs() -> None:
    from draftslice.bench.e2e import _validate

    person = next(case for case in DEFAULT_CASES if case.name == "json_person")
    assert _validate('{"name":"Alice","age":30,"city":"Paris"}', person)[0] is True
    assert _validate('{"name":"Alice"}', person)[0] is False
    assert _validate("not json", person)[0] is False

    date = next(case for case in DEFAULT_CASES if case.name == "regex_date")
    assert _validate("2026-11-20", date)[0] is True
    assert _validate("20 November 2026", date)[0] is False

    free = next(case for case in DEFAULT_CASES if case.name == "chat")
    assert _validate("anything", free)[0] is True


def _record(case: str, batch: int, tokens, tok_s: float, tau: float = 0.0):
    return {
        "case": case,
        "domain": "chat",
        "batch": batch,
        "repetition": 0,
        "token_ids": tokens,
        "tokens_per_s": tok_s,
        "validity": 1.0,
        "tau": tau,
    }


def test_compare_methods_pairs_records_and_checks_greedy_equality() -> None:
    results = {
        "base": {
            "summary": {"validity": 1.0},
            "records": [_record("chat", 1, [[1, 2, 3]], 10.0, 1.0)],
        },
        "method": {
            "summary": {"validity": 1.0},
            "records": [_record("chat", 1, [[1, 2, 3]], 20.0, 2.0)],
        },
        "other": {
            "summary": {"validity": 1.0},
            "records": [_record("chat", 1, [[9]], 5.0, 0.5)],
        },
    }
    comparison = compare_methods(results)
    assert comparison["baseline"] == "base"
    assert comparison["methods"]["method"]["mean_speedup"] == pytest.approx(2.0)
    assert comparison["methods"]["method"]["greedy_output_equal_rate"] == 1.0
    assert comparison["methods"]["other"]["greedy_output_equal_rate"] == 0.0


def test_physical_gpu_index_honours_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _physical_gpu_index() == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    assert _physical_gpu_index() == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert _physical_gpu_index() == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    assert _physical_gpu_index() == 0


def test_run_method_rejects_a_method_outside_the_config() -> None:
    with pytest.raises(E2EError):
        run_method(E2EConfig(model="m", methods=("mtp",)), "other")


def test_e2e_result_is_json_serializable() -> None:
    payload = {
        "schema_version": "e2e/1",
        "summary": {"validity": 1.0},
        "records": [_record("chat", 1, [[1]], 1.0)],
    }
    assert json.loads(json.dumps(payload))["schema_version"] == "e2e/1"
