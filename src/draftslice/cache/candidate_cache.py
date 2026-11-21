"""A byte-budgeted LRU cache from group ids to candidate token sets."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterable

from ..plan.types import TokenSet
from .plan_cache import CacheStats


def _group_key(group_ids: Iterable[int]) -> tuple[int, ...]:
    key = tuple(sorted(set(group_ids)))
    if any(group_id < 0 for group_id in key):
        raise ValueError("group ids must be non-negative")
    return key


class CandidateSetCache:
    """Cache screened candidate sets by the selected vocabulary groups."""

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._items: OrderedDict[tuple[int, ...], tuple[TokenSet, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize = 0
        self._lock = threading.Lock()

    @staticmethod
    def _size(key: tuple[int, ...], tokens: TokenSet) -> int:
        return 64 + 4 * len(key) + 4 * len(tokens)

    def get(self, group_ids: Iterable[int]) -> TokenSet | None:
        key = _group_key(group_ids)
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self._misses += 1
                return None
            tokens, _ = item
            self._items.move_to_end(key)
            self._hits += 1
            return tokens

    def put(self, group_ids: Iterable[int], tokens: TokenSet) -> None:
        key = _group_key(group_ids)
        size = self._size(key, tokens)
        with self._lock:
            if size > self.max_bytes:
                self._oversize += 1
                return
            existing = self._items.pop(key, None)
            if existing is not None:
                self._bytes -= existing[1]
            self._items[key] = (tokens, size)
            self._bytes += size
            self._evict_locked()
            self._items.move_to_end(key)

    def _evict_locked(self) -> None:
        while self._bytes > self.max_bytes and self._items:
            _, (_, size) = self._items.popitem(last=False)
            self._bytes -= size
            self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._oversize = 0

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                oversize=self._oversize,
                entries=len(self._items),
                bytes=self._bytes,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


__all__ = ["CandidateSetCache"]
