from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.bench.baselines import BaselineError  # noqa: E402
from draftslice.bench.official_baselines import (  # noqa: E402
    CFHeadPolicy,
    FlashHeadPolicy,
    make_vocabtrim_policy,
)
from draftslice.plan import TokenSet  # noqa: E402


def test_cfhead_policy_structure_and_determinism() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 8)
    policy = CFHeadPolicy(
        weight,
        tile_size=2,
        cluster_size=4,
        top_k=8,
        kmeans_iters=3,
        seed=7,
    )
    hidden = torch.randn(2, 8)
    first = policy.candidates(hidden)
    second = CFHeadPolicy(
        weight,
        tile_size=2,
        cluster_size=4,
        top_k=8,
        kmeans_iters=3,
        seed=7,
    ).candidates(hidden)
    assert first == second
    assert len(first) == 2
    assert all(len(candidate) == 8 for candidate in first)
    assert all(all(0 <= token < 32 for token in candidate) for candidate in first)
    assert policy.storage_bytes > 0


def test_flashhead_policy_structure_and_determinism() -> None:
    torch.manual_seed(1)
    weight = torch.randn(32, 8)
    policy = FlashHeadPolicy(
        weight,
        num_clusters=8,
        num_probes=2,
        cluster_iters=2,
        top_choices=4,
        seed=11,
    )
    hidden = torch.randn(3, 8)
    first = policy.candidates(hidden)
    second = FlashHeadPolicy(
        weight,
        num_clusters=8,
        num_probes=2,
        cluster_iters=2,
        top_choices=4,
        seed=11,
    ).candidates(hidden)
    assert first == second
    assert len(first) == 3
    assert all(0 < len(candidate) <= policy.candidate_tokens for candidate in first)
    assert all(all(0 <= token < 32 for token in candidate) for candidate in first)


def test_vocabtrim_policy() -> None:
    policy = make_vocabtrim_policy([1, 2, 3])
    assert policy.name == "vocabtrim"
    assert policy.candidates(torch.zeros(2, 4)) == [TokenSet([1, 2, 3])] * 2


def test_official_policy_validation() -> None:
    weight = torch.randn(16, 6)
    with pytest.raises(BaselineError):
        CFHeadPolicy(weight, tile_size=4)  # 6 is not divisible by 4
    with pytest.raises(BaselineError):
        FlashHeadPolicy(weight, num_clusters=1)
    with pytest.raises(BaselineError):
        FlashHeadPolicy(weight, num_clusters=4, num_probes=5)
