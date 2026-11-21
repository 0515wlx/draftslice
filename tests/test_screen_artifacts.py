from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.artifact import ArtifactError  # noqa: E402
from draftslice.screen import (  # noqa: E402
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    RouterPolicy,
    StaticPolicy,
    build_screener_bundle,
    load_screener_bundle,
    screener_calibration,
    screener_from_bundle,
    screener_to_arrays,
)


def _policies() -> list:
    groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    return [
        FullPolicy(vocab_size=8),
        StaticPolicy(rows=(1, 3, 5)),
        GroupPolicy(groups=groups, centroids=torch.randn(4, 8), top_c=2),
        RouterPolicy(
            groups=groups,
            w1=torch.randn(5, 8),
            w2=torch.randn(4, 5),
            top_c=2,
        ),
        LowRankPolicy(vr=torch.randn(8, 3), a=torch.randn(8, 3), top_m=3),
    ]


@pytest.mark.parametrize("policy", _policies())
def test_screener_bundle_roundtrip_matches_candidates(policy, tmp_path) -> None:
    bundle = build_screener_bundle(
        policy,
        vocab_size=8,
        hidden_size=8,
        calibration={"delta": 0.05, "calibration_id": "cal-1"},
    )
    bundle.write(tmp_path / "bundle")
    restored, loaded = load_screener_bundle(tmp_path / "bundle")
    assert restored.name == policy.name
    assert type(restored) is type(policy)
    assert screener_calibration(loaded) == {"delta": 0.05, "calibration_id": "cal-1"}

    hidden = torch.randn(3, 8)
    assert [c.ids for c in restored.candidates(hidden)] == [
        c.ids for c in policy.candidates(hidden)
    ]


def test_screener_to_arrays_rejects_oracle_and_unknown() -> None:
    class Unknown:
        name = "unknown"

    with pytest.raises(ArtifactError):
        screener_to_arrays(Unknown())  # type: ignore[arg-type]


def test_bundle_hash_detects_tampered_arrays() -> None:
    bundle = build_screener_bundle(
        StaticPolicy(rows=(1, 3)), vocab_size=8, hidden_size=4
    )
    restored = screener_from_bundle(bundle)
    assert restored.rows == (1, 3)
    bundle.arrays["rows"] = torch.tensor([1, 4], dtype=torch.int64)
    with pytest.raises(ArtifactError):
        bundle.verify()


def test_missing_required_array_is_rejected() -> None:
    bundle = build_screener_bundle(
        RouterPolicy(
            groups=((0, 1), (2, 3)),
            w1=torch.randn(3, 4),
            w2=torch.randn(2, 3),
            top_c=1,
        ),
        vocab_size=4,
        hidden_size=4,
    )
    del bundle.arrays["router_w2"]
    with pytest.raises(ArtifactError):
        screener_from_bundle(bundle)
