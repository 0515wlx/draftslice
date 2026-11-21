from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.bench.baselines import (  # noqa: E402
    BaselineError,
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    OracleGroupPolicy,
    RouterPolicy,
    StaticPolicy,
    group_centroids,
    make_group_policy,
)
from draftslice.plan import TokenSet  # noqa: E402


def test_full_and_static_policies() -> None:
    hidden = torch.randn(3, 4)
    assert FullPolicy(vocab_size=8).candidates(hidden) == [TokenSet(range(8))] * 3
    assert StaticPolicy((4, 1)).candidates(hidden) == [TokenSet([1, 4])] * 3


def test_group_policy_selects_top_group() -> None:
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    groups = ((0, 1), (2, 3))
    policy = make_group_policy(weight, groups, top_c=1)
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert policy.centroids.shape == (2, 2)
    assert policy.candidates(hidden) == [TokenSet([0, 1]), TokenSet([2, 3])]


def test_router_policy_with_identity_features() -> None:
    weight = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    groups = ((0, 1), (2, 3))
    centroids = group_centroids(weight, groups)
    policy = RouterPolicy(
        groups=groups,
        w1=torch.eye(2),
        w2=centroids,
        top_c=1,
    )
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert policy.candidates(hidden) == [TokenSet([0, 1]), TokenSet([2, 3])]


def test_lowrank_policy_top_m() -> None:
    torch.manual_seed(0)
    weight = torch.randn(16, 8)
    policy = LowRankPolicy.from_weight(weight, rank=4, top_m=3)
    hidden = torch.randn(2, 8)
    candidates = policy.candidates(hidden)
    assert len(candidates) == 2
    assert all(len(candidate) == 3 for candidate in candidates)
    scores = (hidden @ policy.vr) @ policy.a.T
    expected = [TokenSet(row) for row in scores.topk(3, dim=-1).indices.tolist()]
    assert candidates == expected


def test_oracle_group_policy() -> None:
    groups = ((0, 1), (2, 3))
    mass = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    policy = OracleGroupPolicy(groups=groups, group_mass=mass, top_c=1)
    hidden = torch.zeros(2, 2)
    assert policy.candidates(hidden) == [TokenSet([2, 3]), TokenSet([0, 1])]


def test_policy_validation() -> None:
    with pytest.raises(BaselineError):
        StaticPolicy(())
    with pytest.raises(BaselineError):
        GroupPolicy(groups=(), centroids=torch.zeros(0, 2), top_c=1)
    with pytest.raises(BaselineError):
        LowRankPolicy(vr=torch.zeros(4, 2), a=torch.zeros(8, 3), top_m=1)
