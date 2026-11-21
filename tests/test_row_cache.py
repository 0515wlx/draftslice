from __future__ import annotations

import pytest

from draftslice.cache import PackedRowCache, rowset_key
from draftslice.plan import TokenSet


def test_rowset_key_is_content_based() -> None:
    assert rowset_key(TokenSet([3, 1, 3])) == rowset_key(TokenSet([1, 3]))
    assert rowset_key(TokenSet([1])) != rowset_key(TokenSet([2]))


def test_cache_hit_miss_and_eviction() -> None:
    cache = PackedRowCache(max_bytes=100)
    key = rowset_key(TokenSet([1, 2]))
    assert cache.get(key) is None
    cache.put(key, "a", 60)
    assert cache.get(key) == "a"
    cache.put("other", "b", 60)
    assert cache.get(key) is None
    assert cache.get("other") == "b"
    assert cache.stats.entries == 1
    assert cache.stats.evictions == 1


def test_cache_rejects_oversize_values() -> None:
    cache = PackedRowCache(max_bytes=10)
    cache.put("big", "x", 11)
    assert cache.get("big") is None
    assert cache.stats.oversize == 1


def test_get_or_build_only_builds_once() -> None:
    cache = PackedRowCache(max_bytes=100)
    calls = []

    def builder():
        calls.append(1)
        return "value", 10

    assert cache.get_or_build("k", builder) == "value"
    assert cache.get_or_build("k", builder) == "value"
    assert len(calls) == 1


def test_pack_weight_rows() -> None:
    torch = pytest.importorskip("torch")
    from draftslice.cache import pack_weight_rows

    weight = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    packed = pack_weight_rows(weight, TokenSet([7, 2]))
    torch.testing.assert_close(packed.value, weight[[2, 7]])
    assert packed.nbytes == 2 * 4 * 4

    empty = pack_weight_rows(weight, TokenSet())
    assert empty.value.shape == (0, 4)
    assert empty.nbytes == 0
