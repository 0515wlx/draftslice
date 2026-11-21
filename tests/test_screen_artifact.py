from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.artifact import ArtifactError, build_bundle  # noqa: E402
from draftslice.plan import GuaranteeKind, TokenSet  # noqa: E402
from draftslice.plan.compiler import Budget, PlanCompiler  # noqa: E402
from draftslice.screen import (  # noqa: E402
    ConformalCalibrator,
    build_partition,
    conformal_quantile,
    coverage_report,
    partition_tokens_for_groups,
)


def test_conformal_quantile_and_coverage() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    q = conformal_quantile(scores, 0.2)
    assert q == 0.5
    calibrator = ConformalCalibrator.fit(scores, delta=0.2, calibration_id="c1")
    assert calibrator.n == 5
    assert calibrator.accepts(0.5)
    assert not calibrator.accepts(0.6)
    report = coverage_report([True, True, False, True], delta=0.2)
    assert report.nominal == pytest.approx(0.8)
    assert report.empirical == pytest.approx(0.75)
    assert report.gap == pytest.approx(-0.05)


def test_partition_is_a_bijection_and_non_empty() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 8)
    partition = build_partition(weight, 4, iters=3, seed=1)
    assert partition.num_groups == 4
    assert partition.num_tokens == 32
    assert all(partition.groups)
    order = partition.layout()
    assert sorted(order) == list(range(32))
    reverse = partition.token_to_row()
    assert all(reverse[token] == row for row, token in enumerate(order))
    tokens = partition_tokens_for_groups(partition, [0, 1])
    assert len(tokens) == len(set(partition.groups[0]) | set(partition.groups[1]))


def test_compile_calibrated_respects_subset_and_budget() -> None:
    compiler = PlanCompiler(vocab_size=16, hidden_size=8)
    universe = TokenSet([1, 2, 3, 4, 5])
    plan = compiler.compile_calibrated(
        TokenSet([2, 4]),
        universe=universe,
        delta=0.05,
    )
    assert plan.guarantee.kind is GuaranteeKind.CALIBRATED
    assert plan.guarantee.delta == 0.05
    assert plan.execution.kind == "slice_gemv"
    assert plan.candidates.ids == (2, 4)

    with pytest.raises(ValueError):
        compiler.compile_calibrated(TokenSet([9]), universe=universe, delta=0.05)

    fallback = compiler.compile_calibrated(
        TokenSet([2, 4]),
        universe=universe,
        delta=0.05,
        budget=Budget(max_candidates=1),
    )
    assert fallback.execution.kind == "full"
    assert fallback.source == "fallback"
    assert fallback.guarantee.kind is GuaranteeKind.CALIBRATED


def test_artifact_bundle_round_trip_and_hash_check(tmp_path) -> None:
    bundle = build_bundle(
        model={"name": "tiny", "vocab_size": 16, "hidden_size": 8, "dtype": "bf16"},
        partition={"groups": 4, "layout": "leaf_contiguous"},
        screener={"kind": "static"},
        calibration={"delta": 0.05, "n": 100},
        arrays={
            "token_to_group": torch.arange(16, dtype=torch.int32) % 4,
            "router": torch.randn(4, 8),
        },
    )
    assert bundle.manifest.bundle_id.startswith("sha256:")
    bundle.verify()

    path = bundle.write(tmp_path / "bundle")
    loaded = type(bundle).load(path)
    loaded.verify()
    torch.testing.assert_close(loaded.arrays["router"], bundle.arrays["router"], atol=0.0, rtol=0.0)

    tampered = dict(loaded.arrays)
    tampered["router"] = tampered["router"] + 1
    with pytest.raises(ArtifactError):
        type(bundle)(
            manifest=loaded.manifest,
            arrays=tampered,
        ).verify()
