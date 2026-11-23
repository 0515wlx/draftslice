from __future__ import annotations

import json

import pytest

from draftslice.cost import (
    A100,
    CalibratedCostModel,
    CalibrationError,
    CostCalibration,
    CostModel,
    CostModelError,
    CostSample,
    get_profile,
)
from draftslice.plan import (
    Execution,
    Guarantee,
    HeadPlan,
    NodePlan,
    PlanCost,
    StepPlan,
    TokenSet,
)


def slice_plan(rows: int, *, static_bytes: int = 0, layout="native") -> HeadPlan:
    tokens = TokenSet(range(rows))
    return HeadPlan(
        candidates=tokens,
        execution=Execution(kind="slice_gemv", rows=tokens, layout=layout),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
        cost=PlanCost(static_bytes=static_bytes),
    )


def full_plan(vocab_size: int = 100) -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet(range(vocab_size)),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        state_id=None,
    )


def test_profile_lookup() -> None:
    assert get_profile("a100") is A100
    with pytest.raises(KeyError):
        get_profile("no-such-gpu")


def test_full_head_prediction_is_in_a_reasonable_range() -> None:
    model = CostModel(vocab_size=248_320, hidden_size=2_560)
    cost = model.predict(full_plan(248_320), profile="a100")
    assert 600.0 < cost.predicted_us["a100"] < 900.0


def test_slice_is_cheaper_than_full() -> None:
    model = CostModel(vocab_size=100_000, hidden_size=2_560)
    full = model.predict(full_plan(100_000), profile="a100")
    sliced = model.predict(slice_plan(1_000), profile="a100")
    assert sliced.predicted_us["a100"] < full.predicted_us["a100"]


def test_unique_rows_changes_traffic() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    plan = slice_plan(100)
    shared = model.predict(plan, batch=8, unique_rows=100)
    per_row = model.predict(plan, batch=8, unique_rows=800)
    assert per_row.dynamic_bytes == 8 * shared.dynamic_bytes
    assert per_row.predicted_us["a100"] > shared.predicted_us["a100"]


def test_default_batch_assumes_no_reuse() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    plan = slice_plan(100)
    batch_one = model.predict(plan, batch=1)
    batch_eight = model.predict(plan, batch=8)
    assert batch_eight.dynamic_bytes == 8 * batch_one.dynamic_bytes


def test_fused_contiguous_layout_uses_streaming_bandwidth() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    plan = slice_plan(1_000, layout="leaf_contiguous")
    gather = model.predict(plan, fused=False)
    fused = model.predict(plan, fused=True)
    assert fused.predicted_us["a100"] < gather.predicted_us["a100"]


def test_static_bytes_l2_vs_hbm() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    l2 = model.predict(slice_plan(10, static_bytes=1_000_000))
    hbm = model.predict(slice_plan(10, static_bytes=100_000_000))
    assert hbm.predicted_us["a100"] > l2.predicted_us["a100"]


def test_predict_step_slice_and_full() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    plan = slice_plan(100)
    step = StepPlan(
        nodes=(NodePlan(0, plan),),
        union_rows=TokenSet(range(100)),
        kernel="slice_gemm",
    )
    cost = model.predict_step(step)
    assert cost.dynamic_bytes == 100 * 100 * 2
    full = StepPlan(
        nodes=(NodePlan(0, full_plan(10_000)),),
        union_rows=TokenSet(),
        kernel="full",
    )
    full_cost = model.predict_step(full)
    assert full_cost.dynamic_bytes > cost.dynamic_bytes


def test_cost_model_validation() -> None:
    model = CostModel(vocab_size=10, hidden_size=10)
    with pytest.raises(ValueError):
        model.predict(slice_plan(1), batch=0)
    with pytest.raises(CostModelError):
        model.predict(full_plan(10), unique_rows=11)
    with pytest.raises(ValueError):
        CostModel(vocab_size=0, hidden_size=10)


def _samples(kind: str, factor: float, count: int = 6) -> list[CostSample]:
    return [
        CostSample(
            kind=kind,
            batch=1 if kind == "slice_gemv" else 8,
            rows=64 * (index + 1),
            predicted_us=100.0 + 10 * index,
            measured_us=(100.0 + 10 * index) * factor,
        )
        for index in range(count)
    ]


def test_calibration_fits_factors_and_passes_gate() -> None:
    samples = _samples("slice_gemv", 1.5) + _samples("full", 0.8)
    calibration = CostCalibration.fit(samples, holdout_fraction=0.0)
    assert calibration.calibrated
    assert calibration.factors["slice_gemv"] == pytest.approx(1.5)
    assert calibration.factors["full"] == pytest.approx(0.8)
    assert calibration.mape == pytest.approx(0.0)


def test_calibration_rejects_calibrated_above_gate() -> None:
    with pytest.raises(CalibrationError):
        CostCalibration(
            profile="a100",
            factors={"default": 1.0},
            mape=0.1,
            holdout_mape=0.5,
            calibrated=True,
            gate=0.25,
        )


def test_calibrated_model_applies_kind_factor() -> None:
    model = CostModel(vocab_size=10_000, hidden_size=100)
    base = model.predict(slice_plan(100)).predicted_us["a100"]
    calibration = CostCalibration(
        profile="a100",
        factors={"full": 1.0, "slice_gemv": 2.0, "slice_gemm": 2.0, "default": 2.0},
        mape=0.1,
        holdout_mape=0.1,
        calibrated=True,
    )
    calibrated = CalibratedCostModel(
        vocab_size=10_000, hidden_size=100, calibration=calibration
    )
    assert calibrated.calibrated
    assert calibrated.predict(slice_plan(100)).predicted_us["a100"] == pytest.approx(
        base * 2.0
    )
    assert not model.calibrated


def test_calibration_rejects_bad_artifacts(tmp_path) -> None:
    with pytest.raises(CalibrationError):
        CostCalibration.fit([])
    with pytest.raises(CalibrationError):
        CostSample(
            kind="bad", batch=1, rows=1, predicted_us=1.0, measured_us=1.0
        )
    calibration = CostCalibration.fit(
        _samples("slice_gemv", 1.0) + _samples("full", 1.0), holdout_fraction=0.0
    )
    path = tmp_path / "calibration.json"
    calibration.save(path)
    loaded = CostCalibration.load(path)
    assert loaded.digest() == calibration.digest()


def test_bench_report_is_loadable_as_a_calibration(tmp_path) -> None:
    calibration = CostCalibration.fit(
        _samples("slice_gemv", 1.0) + _samples("full", 1.0), holdout_fraction=0.0
    )
    path = tmp_path / "cost_model.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cost-bench/1",
                "calibration": calibration.to_dict(),
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = CostCalibration.load(path)
    assert loaded.to_dict() == calibration.to_dict()
