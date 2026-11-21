"""Bridge from vLLM structured-output bitmasks to DraftSlice valid sets.

The vLLM scheduler builds a token bitmask for structured-output requests and
sends it to the worker. The worker applies it to target logits. This module
captures the worker's bitmask for the current step so a custom speculator can
build a strict plan for the next draft token.

This is intentionally a narrow bridge:

- only the current step's bitmask is captured;
- only the first draft position is treated as exact (later draft positions do
  not have a future grammar state until the target verifies the prefix);
- an absent or malformed snapshot means "fall back to the full head".
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..plan.types import TokenSet


class GrammarBridgeError(ValueError):
    """Raised when a vLLM grammar bitmask cannot be interpreted."""


# Cross-step cache for decoded valid sets. JSON/grammar states repeat heavily
# across requests and draft steps, so a bitmask-row lookup turns the per-step
# Python construction into a dict hit. Keyed by the packed row bytes; only small
# sets are cached so a near-full-vocabulary (unconstrained string) state cannot
# pin hundreds of megabytes of Python ints.
_TOKEN_SET_CACHE: OrderedDict[tuple[bytes, int], TokenSet] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 512
_CACHE_MAX_IDS = 8192


def count_valid_ids(row: Any, vocab_size: int) -> int:
    """Count allowed tokens in a packed bitmask row without unpacking it."""

    if vocab_size <= 0:
        raise GrammarBridgeError("vocab_size must be positive")
    words = np.asarray(row, dtype=np.uint32).reshape(-1)
    if words.size == 0:
        return 0
    # Rows may carry extra trailing words; only count the words that cover the
    # vocabulary, then mask padding bits in the final word.
    needed = min(words.size, (int(vocab_size) + 31) // 32)
    words = words[:needed]
    total = int(np.bitwise_count(words).sum())
    extra = words.size * 32 - int(vocab_size)
    if extra > 0:
        keep = np.uint32((1 << (32 - extra)) - 1)
        total -= int(np.bitwise_count(np.uint32(words[-1])).sum())
        total += int(np.bitwise_count(np.uint32(words[-1] & keep)).sum())
    return total


def unpack_valid_ids(row: Any, vocab_size: int) -> TokenSet:
    """Unpack one xgrammar-style bitmask row.

    A set bit means the token is allowed (the bitmask is initialized to all
    ones and the grammar clears disallowed bits). ``row`` is an int32 array of
    packed 32-bit words. ``np.unpackbits`` is ~2x faster than a per-word shift
    broadcast and returns ascending unique ids, so the ``TokenSet`` fast path is
    safe.
    """

    if vocab_size <= 0:
        raise GrammarBridgeError("vocab_size must be positive")
    words = np.asarray(row, dtype=np.uint32).reshape(-1)
    if words.size == 0:
        return TokenSet()
    bits = np.unpackbits(words.view(np.uint8), bitorder="little")[:vocab_size]
    allowed = np.flatnonzero(bits)
    return TokenSet._from_sorted_unique(allowed.tolist())


def _cached_valid_ids(row: np.ndarray, vocab_size: int) -> TokenSet:
    key = (row.tobytes(), int(vocab_size))
    with _CACHE_LOCK:
        cached = _TOKEN_SET_CACHE.get(key)
        if cached is not None:
            _TOKEN_SET_CACHE.move_to_end(key)
            return cached
    token_set = unpack_valid_ids(row, vocab_size)
    if len(token_set) <= _CACHE_MAX_IDS:
        with _CACHE_LOCK:
            _TOKEN_SET_CACHE[key] = token_set
            while len(_TOKEN_SET_CACHE) > _CACHE_MAX_ENTRIES:
                _TOKEN_SET_CACHE.popitem(last=False)
    return token_set


def clear_valid_set_cache() -> None:
    """Drop the cross-step valid-set cache (mostly for tests/freeing memory)."""

    with _CACHE_LOCK:
        _TOKEN_SET_CACHE.clear()


@dataclass(frozen=True, slots=True)
class GrammarSnapshot:
    """A captured vLLM grammar bitmask for one batch step."""

    req_ids: tuple[str, ...]
    structured_req_ids: tuple[str, ...]
    bitmask: np.ndarray
    mask_stride: int
    vocab_size: int

    def __post_init__(self) -> None:
        if self.mask_stride <= 0:
            raise GrammarBridgeError("mask_stride must be positive")
        if self.vocab_size <= 0:
            raise GrammarBridgeError("vocab_size must be positive")
        rows = len(self.structured_req_ids) * self.mask_stride
        if self.bitmask.ndim != 2:
            raise GrammarBridgeError("bitmask must be 2-D")
        if self.bitmask.shape[0] < rows:
            raise GrammarBridgeError(
                f"bitmask has {self.bitmask.shape[0]} rows, expected at least {rows}"
            )

    def valid_ids(
        self, req_id: str, position: int = 0, *, max_ids: int | None = None
    ) -> TokenSet | None:
        if req_id not in self.structured_req_ids:
            return None
        if position < 0 or position >= self.mask_stride:
            raise GrammarBridgeError(
                f"position {position} is outside mask_stride={self.mask_stride}"
            )
        block = self.structured_req_ids.index(req_id)
        row = self.bitmask[block * self.mask_stride + position]
        if max_ids is not None and count_valid_ids(row, self.vocab_size) > max_ids:
            # Too wide to slice: the caller should use the full head. Returning
            # ``None`` keeps the same "full fallback" contract as a free-form
            # row and avoids building a huge Python token set.
            return None
        return _cached_valid_ids(np.asarray(row), self.vocab_size)

    def valid_sets(
        self, position: int = 0, *, max_ids: int | None = None
    ) -> dict[str, TokenSet | None]:
        return {
            req_id: self.valid_ids(req_id, position, max_ids=max_ids)
            for req_id in self.structured_req_ids
        }


_CURRENT: ContextVar[GrammarSnapshot | None] = ContextVar(
    "draftslice_grammar_snapshot", default=None
)
_PATCHED = False
_SCHED_PATCHED = False
_SCHED_DEBUG_COUNT = 0
_CAPTURE_DEBUG_COUNT = 0


def _debug_enabled() -> bool:
    return os.environ.get("DRAFTSLICE_VLLM_DEBUG_BITMASK") == "1"


def _install_scheduler_debug() -> None:
    """Log the scheduler-side bitmask rows and the grammar state used to fill them.

    Diagnostic only: it lets us compare the scheduler's constrained rows with
    what the worker-side bridge captures.
    """

    global _SCHED_PATCHED
    if _SCHED_PATCHED:
        return
    try:
        from vllm.v1.structured_output import StructuredOutputManager
        from vllm.v1.structured_output.utils import strip_speculative_padding
    except ImportError:  # pragma: no cover - vLLM optional
        return

    original = StructuredOutputManager.grammar_bitmask

    def grammar_bitmask(
        self: Any,
        requests: Any,
        structured_output_request_ids: Any,
        scheduled_spec_decode_tokens: Any,
    ) -> Any:
        global _SCHED_DEBUG_COUNT
        out = original(
            self, requests, structured_output_request_ids, scheduled_spec_decode_tokens
        )
        if out is not None and _SCHED_DEBUG_COUNT < 8:
            _SCHED_DEBUG_COUNT += 1
            import sys

            stride = out.shape[0] // max(1, len(structured_output_request_ids))
            vocab_size = int(self.vllm_config.model_config.get_vocab_size())
            print(
                f"[ds-sched] n={len(structured_output_request_ids)} "
                f"stride={stride} shape={out.shape}",
                file=sys.stderr,
                flush=True,
            )
            for req_id in list(structured_output_request_ids)[:2]:
                request = requests[req_id]
                tokens = strip_speculative_padding(
                    scheduled_spec_decode_tokens.get(req_id, [])
                )
                bounds = self._get_constraint_bounds(request, tokens)
                structured = request.structured_output_request
                grammar = structured.grammar if structured is not None else None
                block = list(structured_output_request_ids).index(req_id)
                counts = [
                    count_valid_ids(out[block * stride + p], vocab_size)
                    for p in range(stride)
                ]
                print(
                    f"[ds-sched] rid={req_id} toks={tokens[:8]} "
                    f"bounds={bounds} "
                    f"term={None if grammar is None else grammar.is_terminated()} "
                    f"counts={counts}",
                    file=sys.stderr,
                    flush=True,
                )
        return out

    StructuredOutputManager.grammar_bitmask = grammar_bitmask
    _SCHED_PATCHED = True



def current_grammar_snapshot() -> GrammarSnapshot | None:
    return _CURRENT.get()


def capture_grammar_snapshot(snapshot: GrammarSnapshot | None) -> None:
    """Set the current snapshot (used by the bridge and tests)."""

    _CURRENT.set(snapshot)


def install_grammar_bridge() -> None:
    """Patch vLLM's structured-output worker to capture the current bitmask.

    The patch is process-local and idempotent. It is installed by the vLLM
    plugin registration, not by importing the core package.
    """

    global _PATCHED
    if _PATCHED:
        return
    try:
        from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
    except ImportError as exc:
        raise ImportError("vLLM is required for DraftSlice's vLLM integration") from exc

    original = StructuredOutputsWorker.apply_grammar_bitmask

    def apply_grammar_bitmask(self, logits, input_batch, grammar_req_ids, grammar_bitmask):
        global _CAPTURE_DEBUG_COUNT
        try:
            stride = int(getattr(self, "mask_stride", 1))
            vocab_size = int(logits.shape[-1])
            if _debug_enabled() and _CAPTURE_DEBUG_COUNT < 8:
                _CAPTURE_DEBUG_COUNT += 1
                import sys

                counts = [
                    count_valid_ids(np.asarray(grammar_bitmask[row]), vocab_size)
                    for row in range(
                        min(len(grammar_req_ids) * stride, grammar_bitmask.shape[0])
                    )
                ]
                print(
                    f"[ds-work] reqs={len(input_batch.req_ids)} "
                    f"struct={list(grammar_req_ids)} stride={stride} "
                    f"shape={grammar_bitmask.shape} counts={counts}",
                    file=sys.stderr,
                    flush=True,
                )
            capture_grammar_snapshot(
                GrammarSnapshot(
                    req_ids=tuple(input_batch.req_ids),
                    structured_req_ids=tuple(grammar_req_ids),
                    bitmask=np.array(grammar_bitmask, copy=True),
                    mask_stride=stride,
                    vocab_size=vocab_size,
                )
            )
        except Exception:
            # Capture must never break the engine. A missing snapshot means the
            # speculator falls back to the full head.
            capture_grammar_snapshot(None)
        return original(self, logits, input_batch, grammar_req_ids, grammar_bitmask)

    StructuredOutputsWorker.apply_grammar_bitmask = apply_grammar_bitmask
    _PATCHED = True
    if _debug_enabled():
        _install_scheduler_debug()


__all__ = [
    "GrammarBridgeError",
    "GrammarSnapshot",
    "capture_grammar_snapshot",
    "clear_valid_set_cache",
    "count_valid_ids",
    "current_grammar_snapshot",
    "install_grammar_bridge",
    "unpack_valid_ids",
]
