from __future__ import annotations

import pytest

from draftslice.cache import CandidateSetCache
from draftslice.plan import TokenSet


def test_candidate_cache_hit_and_miss() -> None:
    cache = CandidateSetCache()
    assert cache.get([3, 1]) is None
    tokens = TokenSet([10, 11])
    cache.put([3, 1], tokens)
    assert cache.get([1, 3]) is tokens
    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.entries == 1
    assert stats.hit_rate == 0.5


def test_candidate_cache_lru_eviction() -> None:
    cache = CandidateSetCache(max_bytes=2 * (64 + 4 + 4 * 2))
    cache.put([0], TokenSet([1, 2]))
    cache.put([1], TokenSet([3, 4]))
    cache.put([2], TokenSet([5, 6]))
    assert len(cache) == 2
    assert cache.get([0]) is None
    assert cache.get([2]) is not None


def test_candidate_cache_skips_oversize() -> None:
    cache = CandidateSetCache(max_bytes=1)
    cache.put([0], TokenSet([1]))
    assert len(cache) == 0
    assert cache.stats.oversize == 1


def test_candidate_cache_rejects_negative_group() -> None:
    with pytest.raises(ValueError):
        CandidateSetCache().get([-1])


def test_candidate_cache_clear() -> None:
    cache = CandidateSetCache()
    cache.put([0], TokenSet([1]))
    cache.get([0])
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 0
