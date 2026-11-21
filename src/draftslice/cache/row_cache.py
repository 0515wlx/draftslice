"""Byte-budgeted cache for packed weight rows.

A hot candidate set can be materialized as a contiguous submatrix once and
reused across decode steps. This is optional: the layout is only a win when the
same row set repeats and the packed matrix stays resident. The cache is
torch-agnostic; callers provide a builder and the byte size of its value.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet
from .plan_cache import CacheStats


def rowset_key(rows: TokenSet, *, layout: str = "native", dtype: str = "bf16") -> tuple[Any, ...]:
    """Stable cache key for a row set and its storage format."""

    return (tuple(rows.ids), layout, dtype)


class PackedRowCache:
    """LRU cache from a row-set key to a packed value (usually a tensor)."""

    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._items: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversize = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self._misses += 1
                return None
            value, _ = item
            self._items.move_to_end(key)
            self._hits += 1
            return value

    def get_or_build(
        self,
        key: Hashable,
        builder: Callable[[], tuple[Any, int]],
        *,
        nbytes: int | None = None,
    ) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value, size = builder()
        if nbytes is not None:
            size = nbytes
        self.put(key, value, size)
        return value

    def put(self, key: Hashable, value: Any, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        with self._lock:
            if nbytes > self.max_bytes:
                self._oversize += 1
                return
            existing = self._items.pop(key, None)
            if existing is not None:
                self._bytes -= existing[1]
            self._items[key] = (value, nbytes)
            self._bytes += nbytes
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


@dataclass(frozen=True, slots=True)
class PackedRows:
    """A packed tensor plus the row ids in its row order."""

    rows: TokenSet
    value: Any
    nbytes: int


def pack_weight_rows(weight: Any, rows: TokenSet) -> PackedRows:
    """Materialize ``weight[rows]`` as a contiguous tensor."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for pack_weight_rows") from exc
    if not rows:
        return PackedRows(rows=rows, value=weight[:0], nbytes=0)
    index = torch.tensor(list(rows.ids), dtype=torch.long, device=weight.device)
    value = weight.index_select(0, index).contiguous()
    nbytes = int(value.numel() * value.element_size())
    return PackedRows(rows=rows, value=value, nbytes=nbytes)


__all__ = ["PackedRowCache", "PackedRows", "pack_weight_rows", "rowset_key"]
