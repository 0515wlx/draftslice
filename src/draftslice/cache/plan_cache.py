"""A small byte-budgeted LRU cache for strict plans."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass

from ..plan.types import HeadPlan


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversize: int = 0
    entries: int = 0
    bytes: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.requests == 0 else self.hits / self.requests


def plan_bytes(plan: HeadPlan) -> int:
    """Estimate the cache footprint of a plan in bytes."""

    size = 128 + 4 * len(plan.candidates)
    if plan.execution.rows is not None:
        size += 4 * len(plan.execution.rows)
    size += 8 * len(plan.cost.predicted_us)
    return size


class LRUPlanCache:
    """A thread-safe, byte-budgeted LRU cache keyed by state/lookup id."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._items: OrderedDict[Hashable, tuple[HeadPlan, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize = 0
        self._lock = threading.Lock()

    def get(self, state_id: Hashable) -> HeadPlan | None:
        with self._lock:
            item = self._items.get(state_id)
            if item is None:
                self._misses += 1
                return None
            plan, size = item
            self._items.move_to_end(state_id)
            self._hits += 1
            return plan

    def put(self, plan: HeadPlan) -> None:
        if plan.state_id is None:
            raise ValueError("only state-keyed plans can be cached")
        self.put_keyed(plan.state_id, plan)

    def put_keyed(self, key: Hashable, plan: HeadPlan) -> None:
        size = plan_bytes(plan)
        with self._lock:
            if size > self.max_bytes:
                self._oversize += 1
                return
            existing = self._items.pop(key, None)
            if existing is not None:
                self._bytes -= existing[1]
            self._items[key] = (plan, size)
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


__all__ = ["CacheStats", "LRUPlanCache", "plan_bytes"]
