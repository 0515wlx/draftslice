"""Draft-head logits construction for the vLLM speculator.

The helper is deliberately independent of vLLM so it can be tested on a small
matrix. The vLLM wrapper supplies the draft LM head weight and a full-logits
fallback. Rows with an exact valid set are packed into a padded candidate matrix
(one padded width bucket per kernel launch), the fused slice kernel gathers the
weight rows and writes them directly into a full-vocabulary tensor with ``-inf``
elsewhere, and rows without a strict set use the full head.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..kernels.torch import TorchGatherKernel
from ..kernels.triton import TritonSliceKernel
from ..plan.types import TokenSet

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class VllmLogitsError(RuntimeError):
    """Raised when draft logits cannot be constructed."""


@dataclass(frozen=True, slots=True)
class DraftSliceLogitsStats:
    rows: int
    sliced_rows: int
    full_rows: int
    candidate_total: int
    distinct_sets: int

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "sliced_rows": self.sliced_rows,
            "full_rows": self.full_rows,
            "candidate_total": self.candidate_total,
            "distinct_sets": self.distinct_sets,
        }


_DEFAULT_KERNELS: dict[bool, Any] = {}


def _default_kernel() -> Any:
    """Process-local default kernel.

    The integration calls this once per draft step; building a fresh kernel was
    pure overhead and discarded the kernel's row/config caches. The kernel is
    stateless apart from those caches, so one instance per graph mode is safe.
    """

    use_graph = os.environ.get("DRAFTSLICE_VLLM_CUDA_GRAPH") == "1"
    kernel = _DEFAULT_KERNELS.get(use_graph)
    if kernel is None:
        if TritonSliceKernel.is_available():
            kernel = TritonSliceKernel(use_cuda_graph=use_graph)
        else:
            kernel = TorchGatherKernel(use_cuda_graph=use_graph)
        _DEFAULT_KERNELS[use_graph] = kernel
    return kernel


_TOKEN_INDEX_CACHE: dict[tuple[int, str], tuple[Any, Any]] = {}
_TOKEN_INDEX_CACHE_MAX = 512
_TOKEN_NP_CACHE: dict[int, tuple[TokenSet, np.ndarray]] = {}
_TOKEN_NP_CACHE_MAX = 512

# Candidate widths are rounded up to a power of two so the Triton kernel sees a
# small, fixed set of shapes instead of recompiling for every grammar state.
_BUCKET_CACHE: dict[int, int] = {}


def candidate_bucket(count: int, *, max_count: int | None = None) -> int:
    """Round a candidate count up to the fixed bucket used by the fused kernel."""

    if count <= 0:
        return 0
    bucket = _BUCKET_CACHE.get(count)
    if bucket is None:
        bucket = 1 << (int(count) - 1).bit_length()
        _BUCKET_CACHE[count] = bucket
    if max_count is not None and bucket > max_count:
        bucket = int(max_count)
    return bucket


class _AsyncH2D:
    """Small non-blocking host-to-device uploads for per-step index tensors.

    ``Tensor.to("cuda")`` on pageable memory synchronizes with the current
    stream. During decode the target forward is already queued on that stream,
    so a tiny upload of a few hundred token ids made ``build_draft_logits``
    block for milliseconds even though the GPU work itself was negligible.
    Pinned staging plus ``non_blocking=True`` keeps the host path off the
    stream; an event per ring slot prevents overwriting a buffer a queued DMA
    still has to read.
    """

    def __init__(self, slots: int = 8) -> None:
        self._slots = slots
        self._buffers: list[Any] = []
        self._views: list[np.ndarray] = []
        self._events: list[Any] = []
        self._next = 0

    def _free_slot(self) -> int:
        for _ in range(self._slots):
            index = self._next
            self._next = (self._next + 1) % self._slots
            if index >= len(self._buffers) or self._events[index].query():
                return index
        return -1

    @staticmethod
    def _torch_dtype(array: np.ndarray) -> Any:
        assert torch is not None
        dtype = _TORCH_DTYPES.get(array.dtype.str)
        if dtype is None:
            dtype = torch.from_numpy(np.empty(0, dtype=array.dtype)).dtype
            _TORCH_DTYPES[array.dtype.str] = dtype
        return dtype

    def upload(self, array: np.ndarray, device: Any) -> Any:
        assert torch is not None
        dtype = self._torch_dtype(array)
        flat = array.reshape(-1)
        if device.type != "cuda":
            return torch.from_numpy(array).to(device)
        index = self._free_slot()
        if index < 0:
            # All staging slots are still in flight; correctness beats latency.
            return torch.from_numpy(array).to(device)
        try:
            if index >= len(self._buffers):
                self._buffers.append(torch.empty(0, dtype=dtype, pin_memory=True))
                self._views.append(self._buffers[index].numpy())
                self._events.append(torch.cuda.Event())
            buffer = self._buffers[index]
            if buffer.numel() < flat.size:
                buffer = torch.empty(flat.size, dtype=dtype, pin_memory=True)
                self._buffers[index] = buffer
                self._views[index] = buffer.numpy()
        except Exception:  # pragma: no cover - pinned memory unavailable
            return torch.from_numpy(array).to(device)
        # A numpy copy into the pinned view is a plain memcpy; building a
        # staging torch tensor and reshaping it through the dispatcher cost
        # more than the copy itself.
        self._views[index][: flat.size] = flat
        target = torch.empty(flat.size, dtype=dtype, device=device)
        target.copy_(buffer[: flat.size], non_blocking=True)
        self._events[index].record()
        return target


_TORCH_DTYPES: dict[str, Any] = {}
_UPLOADS: dict[str, _AsyncH2D] = {}


def _upload(array: np.ndarray, device: Any) -> Any:
    # One uploader per (device, dtype) so a staging slot never mixes an int32
    # candidate payload with an int64 row-index array.
    key = f"{device}:{array.dtype.str}"
    uploader = _UPLOADS.get(key)
    if uploader is None:
        uploader = _AsyncH2D()
        _UPLOADS[key] = uploader
    return uploader.upload(array, device)


def _fused_max_width(max_rows: int) -> int:
    """Widest candidate bucket the per-row GEMV fast path should handle.

    Above this width the gather+GEMM fallback wins because it shares the weight
    tile across rows and uses tensor cores. The default keeps the fused path to
    the tool-call/enum regime, where slicing is a clear win.
    """

    raw = os.environ.get("DRAFTSLICE_VLLM_FUSED_MAX_WIDTH")
    if raw is None:
        return min(max_rows, 8_192)
    try:
        return max(1, min(max_rows, int(raw)))
    except ValueError:
        return min(max_rows, 8_192)


def _token_index_np(token_set: TokenSet) -> np.ndarray:
    """Cache a writable int32 id array for a ``TokenSet``.

    ``TokenSet.ids`` is a Python tuple; converting it once avoids rebuilding the
    array from Python ints for every batch row that shares the grammar state.
    """

    key = id(token_set)
    cached = _TOKEN_NP_CACHE.get(key)
    if cached is not None and cached[0] is token_set:
        return cached[1]
    array = np.fromiter(token_set.ids, dtype=np.int32, count=len(token_set))
    _TOKEN_NP_CACHE[key] = (token_set, array)
    while len(_TOKEN_NP_CACHE) > _TOKEN_NP_CACHE_MAX:
        _TOKEN_NP_CACHE.pop(next(iter(_TOKEN_NP_CACHE)))
    return array


def _token_index(token_set: TokenSet, device: Any) -> Any:
    """Return a cached int32 candidate tensor for a ``TokenSet``.

    Identical grammar states share one cached ``TokenSet`` object across steps,
    so the GPU tensor can be reused instead of rebuilt from tens of thousands of
    Python ints every call. The strong reference prevents an ``id`` from being
    reused while its tensor is still cached.
    """

    assert torch is not None
    key = (id(token_set), str(device))
    cached = _TOKEN_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is token_set:
        return cached[1]
    array = np.fromiter(token_set.ids, dtype=np.int32, count=len(token_set))
    tensor = _upload(array, device)
    _TOKEN_INDEX_CACHE[key] = (token_set, tensor)
    while len(_TOKEN_INDEX_CACHE) > _TOKEN_INDEX_CACHE_MAX:
        _TOKEN_INDEX_CACHE.pop(next(iter(_TOKEN_INDEX_CACHE)))
    return tensor


def _postprocess(values: Any, *, scale: float, soft_cap: float | None) -> Any:
    assert torch is not None
    values = values.float()
    if soft_cap is not None:
        values = torch.tanh(values / soft_cap) * soft_cap
    if scale != 1.0:
        values = values * scale
    return values


def build_draft_logits(
    hidden_states: Any,
    weight: Any,
    valid_sets: Sequence[TokenSet | None],
    *,
    vocab_size: int,
    full_logits: Callable[[Any], Any],
    max_slice_rows: int | None = None,
    kernel: Any | None = None,
    scale: float = 1.0,
    soft_cap: float | None = None,
) -> tuple[Any, DraftSliceLogitsStats]:
    """Build ``[batch, vocab]`` logits with sliced rows and ``-inf`` elsewhere.

    ``valid_sets[i] is None`` means "no strict plan for this row"; those rows are
    computed by ``full_logits``. An empty or oversized valid set also falls back
    to the full head, because slicing it is not a strict-mode win.
    """

    if torch is None:
        raise VllmLogitsError("torch is not installed")
    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise VllmLogitsError("hidden_states and weight must be 2-D")
    if hidden_states.shape[1] != weight.shape[1]:
        raise VllmLogitsError("hidden width does not match the head weight")
    if len(valid_sets) != hidden_states.shape[0]:
        raise VllmLogitsError("valid_sets length does not match the batch")
    if weight.shape[0] < vocab_size:
        raise VllmLogitsError(f"weight has {weight.shape[0]} rows, need vocab_size={vocab_size}")

    max_rows = vocab_size if max_slice_rows is None else max_slice_rows
    if max_rows <= 0:
        raise VllmLogitsError("max_slice_rows must be positive")
    kernel = kernel or _default_kernel()

    out = torch.full(
        (hidden_states.shape[0], vocab_size),
        float("-inf"),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    head = weight[:vocab_size]

    fused = getattr(kernel, "supports_rows_into", False)
    fused_max_width = _fused_max_width(max_rows) if fused else 0
    full_rows: list[int] = []
    groups: dict[int, tuple[Any, list[int]]] = {}
    buckets: dict[int, list[tuple[int, TokenSet]]] = {}
    distinct_ids: set[int] = set()
    for row, token_set in enumerate(valid_sets):
        if token_set is None or not token_set.ids or len(token_set) > max_rows:
            full_rows.append(row)
            continue
        ids = token_set.ids
        # Bitmask decoding already clips to ``vocab_size`` and returns sorted
        # ids, so the old per-token Python scan was pure overhead on the hot
        # path (up to 32k iterations per row).
        if ids[-1] >= vocab_size:
            full_rows.append(row)
            continue
        distinct_ids.add(id(token_set))
        if fused and len(token_set) <= fused_max_width:
            # One launch per candidate-width bucket: grammar states in the same
            # bucket share a padded [rows, width] token matrix, so a batch with
            # many distinct states still costs one kernel instead of one launch
            # per state.
            width = 1 << (len(token_set) - 1).bit_length()
            if width > max_rows:
                width = max_rows
            entries = buckets.get(width)
            if entries is None:
                buckets[width] = [(row, token_set)]
            else:
                entries.append((row, token_set))
        else:
            key = id(token_set)
            if key not in groups:
                # Identical grammar states share one cached TokenSet object;
                # keying by identity avoids hashing a large ``ids`` tuple.
                groups[key] = (token_set, [])
            groups[key][1].append(row)

    sliced_rows = 0
    candidate_total = 0
    if buckets:
        for width in sorted(buckets):
            entries = buckets[width]
            count = len(entries)
            # One pinned upload per bucket: candidate ids followed by the
            # original batch-row indices. Two pageable uploads used to block on
            # the target forward already queued on the stream.
            payload = np.empty(count * width + count, dtype=np.int32)
            tokens_view = payload[: count * width].reshape(count, width)
            tokens_view.fill(-1)
            rows_view = payload[count * width :]
            for position, (row, token_set) in enumerate(entries):
                ids_np = _token_index_np(token_set)
                tokens_view[position, : ids_np.size] = ids_np
                rows_view[position] = row
            device_payload = _upload(payload, hidden_states.device)
            kernel.run_rows_into(
                hidden_states,
                head,
                device_payload[: count * width],
                device_payload[count * width :],
                out,
                width=width,
                scale=scale,
                soft_cap=soft_cap,
            )
            sliced_rows += len(entries)
            candidate_total += sum(len(token_set) for _, token_set in entries)
    for token_set, rows in groups.values():
        ids = token_set.ids
        row_index = _upload(np.asarray(rows, dtype=np.int64), hidden_states.device)
        token_index = _token_index(token_set, hidden_states.device)
        hidden_subset = hidden_states.index_select(0, row_index)
        values = kernel.run_tensor(hidden_subset, head, token_index)
        values = _postprocess(values, scale=scale, soft_cap=soft_cap)
        # Flat index_put is noticeably faster than a 2-D advanced-index write
        # on a [batch, vocab] tensor and avoids a second generic scatter
        # kernel.
        flat = (row_index[:, None] * vocab_size + token_index[None, :]).reshape(-1)
        out.view(-1).index_put_((flat,), values.reshape(-1))
        sliced_rows += len(rows)
        candidate_total += len(ids) * len(rows)

    if full_rows:
        row_index = _upload(np.asarray(full_rows, dtype=np.int64), hidden_states.device)
        values = full_logits(hidden_states.index_select(0, row_index))
        if values is None:
            raise VllmLogitsError("full_logits returned None")
        if values.shape[0] != len(full_rows) or values.shape[1] < vocab_size:
            raise VllmLogitsError(
                f"full_logits returned shape {tuple(values.shape)}, "
                f"expected [{len(full_rows)}, >= {vocab_size}]"
            )
        out[row_index] = values[:, :vocab_size].float()

    distinct_sets = len(distinct_ids)
    return out, DraftSliceLogitsStats(
        rows=hidden_states.shape[0],
        sliced_rows=sliced_rows,
        full_rows=len(full_rows),
        candidate_total=candidate_total,
        distinct_sets=distinct_sets,
    )


__all__ = ["DraftSliceLogitsStats", "VllmLogitsError", "build_draft_logits"]
