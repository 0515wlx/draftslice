from __future__ import annotations

import pytest

from draftslice.cache import LRUPlanCache, plan_bytes
from draftslice.plan import Execution, Guarantee, HeadPlan, TokenSet


def make_plan(state_id: int, ids=(1, 2, 3)) -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet(ids),
        execution=Execution(kind="slice_gemv", rows=TokenSet(ids)),
        guarantee=Guarantee.exact_valid_set(),
        state_id=state_id,
    )


def test_cache_hit_and_miss() -> None:
    cache = LRUPlanCache()
    plan = make_plan(1)
    assert cache.get(1) is None
    cache.put(plan)
    assert cache.get(1) is plan
    stats = cache.stats
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.entries == 1
    assert stats.hit_rate == 0.5


def test_cache_lru_eviction() -> None:
    plan = make_plan(1)
    cache = LRUPlanCache(max_bytes=plan_bytes(plan) * 2)
    cache.put(plan)
    cache.put(make_plan(2))
    cache.put(make_plan(3))
    assert len(cache) == 2
    assert cache.get(1) is None
    assert cache.get(3) is not None
    cache.put(make_plan(4))
    assert cache.get(2) is None
    assert cache.stats.evictions >= 1


def test_cache_rejects_anonymous_plan() -> None:
    plan = HeadPlan(
        candidates=TokenSet([1]),
        execution=Execution(kind="slice_gemv", rows=TokenSet([1])),
        guarantee=Guarantee.exact_valid_set(),
        state_id=None,
    )
    with pytest.raises(ValueError):
        LRUPlanCache().put(plan)


def test_cache_skips_oversize_plan() -> None:
    cache = LRUPlanCache(max_bytes=1)
    cache.put(make_plan(1))
    assert len(cache) == 0
    assert cache.stats.oversize == 1


def test_cache_clear() -> None:
    cache = LRUPlanCache()
    cache.put(make_plan(1))
    cache.get(1)
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 0
    assert cache.stats.entries == 0
