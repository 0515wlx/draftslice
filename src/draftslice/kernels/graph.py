"""Small CUDA-graph cache for draft-head kernels.

Candidate sets change every grammar step, but the *shape* of a slice is stable
and hot plans are cached. The cache therefore keeps one graph per
``(kernel, weight, batch, width, dtype, row-count)`` bucket and copies a new row
index or hidden state into the graph's static buffers when the tensor identity
changes. Replaying the graph removes the Python/Triton launch overhead that
dominates tiny slices.

Safety rules:

* graph capture failure is not fatal; the cache falls back to eager execution;
* the returned tensor is owned by the graph entry. A caller must consume it
  before the next run with the same key, which is the normal executor contract;
* entries are evicted with a simple LRU bound so a serving process cannot grow
  without limit.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - exercised when torch is installed
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class GraphError(RuntimeError):
    """Raised when a CUDA graph cannot be used."""


@dataclass(slots=True)
class _GraphEntry:
    graph: Any
    hidden: Any
    rows: Any | None
    out: Any
    source_hidden: Any
    source_rows: Any | None
    source_hidden_version: int | None = None
    source_rows_version: int | None = None


def _tensor_version(tensor: Any) -> int | None:
    return getattr(tensor, "_version", None)


class CudaGraphCache:
    """Capture and replay kernel calls on static input buffers."""

    def __init__(self, *, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries: OrderedDict[Any, _GraphEntry] = OrderedDict()
        self._disabled_reason: str | None = None
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0

    @staticmethod
    def is_available() -> bool:
        if torch is None:
            return False
        return bool(torch.cuda.is_available())

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def run(
        self,
        key: Any,
        fn: Callable[[Any, Any | None], Any],
        hidden: Any,
        rows: Any | None = None,
    ) -> Any:
        """Run ``fn(static_hidden, static_rows)`` through a cached graph.

        ``key`` must include everything that changes the kernel signature:
        backend, weight identity/pointer, batch, width, dtype, and row count.
        Row *contents* may change; they are copied into the static buffer.
        """

        if self._disabled_reason is not None or torch is None:
            self.fallbacks += 1
            return fn(hidden, rows)

        entry = self._entries.get(key)
        if entry is None:
            try:
                entry = self._capture(key, fn, hidden, rows)
            except Exception as exc:  # pragma: no cover - device dependent
                self._disabled_reason = f"{type(exc).__name__}: {exc}"
                self.fallbacks += 1
                return fn(hidden, rows)
        else:
            self._entries.move_to_end(key)
            self._refresh(entry, hidden, rows)

        entry.graph.replay()
        self.replays += 1
        return entry.out

    def _capture(
        self,
        key: Any,
        fn: Callable[[Any, Any | None], Any],
        hidden: Any,
        rows: Any | None,
    ) -> _GraphEntry:
        assert torch is not None
        if hidden.device.type != "cuda":
            raise GraphError("CUDA graphs require CUDA tensors")

        static_hidden = hidden.clone()
        static_rows = None if rows is None else rows.clone()

        stream = torch.cuda.Stream(device=hidden.device)
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(static_hidden, static_rows)
        torch.cuda.current_stream(hidden.device).wait_stream(stream)
        torch.cuda.synchronize(hidden.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = fn(static_hidden, static_rows)
        torch.cuda.synchronize(hidden.device)

        entry = _GraphEntry(
            graph=graph,
            hidden=static_hidden,
            rows=static_rows,
            out=out,
            source_hidden=hidden,
            source_rows=rows,
            source_hidden_version=_tensor_version(hidden),
            source_rows_version=None if rows is None else _tensor_version(rows),
        )
        self._entries[key] = entry
        self.captures += 1
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    @staticmethod
    def _refresh(entry: _GraphEntry, hidden: Any, rows: Any | None) -> None:
        hidden_version = _tensor_version(hidden)
        if hidden is not entry.source_hidden or hidden_version != entry.source_hidden_version:
            if hidden.shape != entry.hidden.shape or hidden.dtype != entry.hidden.dtype:
                raise GraphError("hidden shape/dtype changed inside a graph bucket")
            entry.hidden.copy_(hidden)
            entry.source_hidden = hidden
            entry.source_hidden_version = hidden_version
        if rows is not None and entry.rows is not None:
            rows_version = _tensor_version(rows)
            if rows is not entry.source_rows or rows_version != entry.source_rows_version:
                if rows.shape != entry.rows.shape:
                    raise GraphError("row shape changed inside a graph bucket")
                entry.rows.copy_(rows)
                entry.source_rows = rows
                entry.source_rows_version = rows_version


__all__ = ["CudaGraphCache", "GraphError"]
