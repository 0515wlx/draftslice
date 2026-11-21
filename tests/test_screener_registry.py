from __future__ import annotations

import pytest

from draftslice.bench import baselines
from draftslice.screen import (
    FullPolicy,
    Screener,
    ScreenError,
    StaticPolicy,
    available_screeners,
    get_screener,
    register_screener,
)


class _Hidden:
    shape = (2, 4)


def test_builtins_are_listed() -> None:
    names = available_screeners()
    expected = (
        "cfhead",
        "flashhead",
        "full",
        "group",
        "lowrank",
        "oracle_group",
        "router",
        "static",
        "svd_lowrank",
        "trained_lowrank",
        "vocabtrim",
    )
    for name in expected:
        assert name in names
    assert names == tuple(sorted(names))


def test_get_static_normalizes_rows() -> None:
    policy = get_screener("static", rows=(3, 1, 3))
    assert isinstance(policy, StaticPolicy)
    assert policy.rows == (1, 3)


def test_get_full_returns_vocab_rows() -> None:
    policy = get_screener("full", vocab_size=8)
    assert isinstance(policy, FullPolicy)
    candidates = policy.candidates(_Hidden())
    assert len(candidates) == 2
    assert candidates[0].ids == tuple(range(8))


def test_vocabtrim_is_a_lazy_static_policy() -> None:
    policy = get_screener("vocabtrim", rows=(2, 4))
    assert isinstance(policy, StaticPolicy)
    assert policy.name == "vocabtrim"
    assert policy.rows == (2, 4)


def test_unknown_screener_raises() -> None:
    with pytest.raises(KeyError):
        get_screener("does_not_exist")
    with pytest.raises(KeyError):
        get_screener("")


def test_register_refuses_silent_shadowing() -> None:
    policy = get_screener("static", rows=(1,))
    register_screener("tmp_test_screener", lambda **kwargs: policy)
    with pytest.raises(ValueError):
        register_screener("tmp_test_screener", lambda **kwargs: policy)
    register_screener("tmp_test_screener", lambda **kwargs: policy, replace=True)
    assert get_screener("tmp_test_screener") is policy


def test_baselines_reexports_are_the_same_objects() -> None:
    assert baselines.StaticPolicy is StaticPolicy
    assert baselines.FullPolicy is FullPolicy
    assert baselines.BaselineError is ScreenError
    assert baselines.CandidatePolicy is Screener
