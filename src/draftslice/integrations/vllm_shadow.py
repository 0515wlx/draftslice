"""Worker-side grammar shadow for exact draft valid sets.

The scheduler's bitmask is produced *before* the target forward and advances a
throw-away copy of the grammar through the scheduled draft tokens. The
speculator, however, proposes the next drafts *after* sampling, from the prefix
that includes the target's bonus token. The captured rows therefore lag the
draft position by one token, and the bonus token is not in any row.

This module keeps an independent xgrammar matcher in the worker. It is advanced
by the tokens the target actually sampled (accepted drafts plus the bonus),
which is exactly the prefix the next draft proposal starts from. The scheduler
remains the authority for validation; the shadow only supplies the draft-head
candidate set.

Scope: enabled only for the xgrammar backend, no reasoning parser, and TP=1.
Any unsupported case is left unregistered and falls back to the scheduler
bitmask (and then to the full head).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..plan.types import TokenSet
from .vllm_grammar import _cached_valid_ids, count_valid_ids


class ShadowError(RuntimeError):
    """Raised when the worker-side grammar shadow cannot be built or advanced."""


_STATE_CACHE: OrderedDict[bytes, TokenSet | None] = OrderedDict()
_STATE_CACHE_MAX = 4096


def _mix_key(key: bytes, tokens: list[int]) -> bytes:
    h = hashlib.blake2b(key, digest_size=8)
    for token in tokens:
        h.update(int(token).to_bytes(4, "little", signed=False))
    return h.digest()


def _clear_state_cache() -> None:
    with _STATE_CACHE_LOCK:
        _STATE_CACHE.clear()


_STATE_CACHE_LOCK = threading.Lock()


@dataclass
class _ShadowEntry:
    req_id: str
    grammar: Any
    state_key: bytes = b""
    step_grammars: dict[int, Any] = field(default_factory=dict)
    step_keys: dict[int, bytes] = field(default_factory=dict)

    def grammar_for(self, step: int) -> Any:
        return self.step_grammars.get(step, self.grammar)

    def key_for(self, step: int) -> bytes:
        return self.step_keys.get(step, self.state_key)

    def clear_steps(self) -> None:
        self.step_grammars.clear()
        self.step_keys.clear()


class GrammarShadowRegistry:
    """Per-request xgrammar matchers advanced by sampled output tokens."""

    def __init__(self, vllm_config: Any, *, max_ids: int) -> None:
        self.vllm_config = vllm_config
        self.max_ids = max_ids
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self.backend: Any | None = None
        self.tokenizer: Any | None = None
        self.entries: dict[int, _ShadowEntry] = {}
        self._bitmask: Any | None = None
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {
            "registered": 0,
            "dropped": 0,
            "advance_calls": 0,
            "advanced_reqs": 0,
            "advanced_tokens": 0,
            "valid_calls": 0,
            "valid_hits": 0,
            "valid_state_hits": 0,
            "valid_missing": 0,
            "valid_terminated": 0,
            "valid_too_wide": 0,
            "valid_errors": 0,
            "forks": 0,
            "fork_failures": 0,
            "fork_accept_failures": 0,
        }

    def debug_dict(self) -> dict[str, int]:
        out = dict(self.stats)
        out["entries"] = len(self.entries)
        return out

    @property
    def enabled(self) -> bool:
        structured = getattr(self.vllm_config, "structured_outputs_config", None)
        if structured is None:
            return False
        # Reasoning prefixes delay the grammar start; leave those to the
        # scheduler snapshot until the shadow mirrors `constraint_start`.
        if getattr(structured, "reasoning_parser", ""):
            return False
        return True

    def _ensure_backend(self) -> Any | None:
        if not self.enabled:
            return None
        if self.backend is not None:
            return self.backend
        try:
            from vllm.tokenizers.registry import cached_tokenizer_from_config
            from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
        except ImportError as exc:  # pragma: no cover - vLLM optional
            raise ShadowError("vLLM xgrammar backend is unavailable") from exc
        self.tokenizer = cached_tokenizer_from_config(
            model_config=self.vllm_config.model_config
        )
        self.backend = XgrammarBackend(
            self.vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=self.vocab_size,
        )
        return self.backend

    def register(self, req_index: int, sampling_params: Any) -> None:
        """Compile and register a shadow for one request, if supported."""

        if not self.enabled or sampling_params is None:
            return
        structured = getattr(sampling_params, "structured_outputs", None)
        if structured is None:
            return
        backend_name = getattr(structured, "_backend", None)
        if backend_name not in (None, "xgrammar"):
            return
        try:
            from vllm.v1.structured_output.request import get_structured_output_key

            request_type, grammar_spec = get_structured_output_key(structured)
            backend = self._ensure_backend()
            if backend is None:
                return
            stop_ids = sorted(
                int(x) for x in (getattr(sampling_params, "all_stop_token_ids", None) or ())
            )
            grammar = backend.compile_grammar(
                request_type,
                grammar_spec,
                stop_token_ids=set(stop_ids),
            )
            # The state cache is shared across requests. Tag it with the grammar
            # definition so two requests with the same token prefix but different
            # schemas cannot collide.
            tag = hashlib.blake2b(digest_size=8)
            tag.update(str(request_type).encode())
            tag.update(b"\0")
            tag.update(str(grammar_spec).encode())
            tag.update(b"\0")
            tag.update(",".join(str(x) for x in stop_ids).encode())
            grammar_tag = tag.digest()
        except Exception as exc:  # pragma: no cover - defensive
            raise ShadowError(f"compile failed for request {req_index}: {exc}") from exc
        req_id = getattr(sampling_params, "request_id", None) or str(req_index)
        with self._lock:
            self.entries[int(req_index)] = _ShadowEntry(
                req_id=req_id, grammar=grammar, state_key=grammar_tag
            )
            self.stats["registered"] += 1

    def drop(self, req_ids: set[str]) -> None:
        if not req_ids:
            return
        with self._lock:
            for index in [i for i, e in self.entries.items() if e.req_id in req_ids]:
                self.entries.pop(index, None)
                self.stats["dropped"] += 1

    def drop_index(self, req_index: int) -> None:
        with self._lock:
            if self.entries.pop(int(req_index), None) is not None:
                self.stats["dropped"] += 1

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
        _clear_state_cache()

    def has(self, req_index: int) -> bool:
        return int(req_index) in self.entries

    def clear_steps(self) -> None:
        for entry in self.entries.values():
            entry.clear_steps()

    def advance_sampled(
        self,
        idx_mapping_np: np.ndarray,
        sampled_tokens_np: np.ndarray,
        num_sampled_np: np.ndarray,
    ) -> None:
        """Advance each registered matcher by the tokens the target sampled.

        ``sampled_tokens`` rows are indexed by batch row; ``idx_mapping_np``
        maps a batch row to the request-state slot used by the registry.
        """

        if not self.entries:
            return
        self.stats["advance_calls"] += 1
        rows = min(len(idx_mapping_np), len(num_sampled_np), len(sampled_tokens_np))
        for row in range(rows):
            entry = self.entries.get(int(idx_mapping_np[row]))
            if entry is None:
                continue
            count = int(num_sampled_np[row])
            if count <= 0:
                continue
            tokens = [int(token) for token in sampled_tokens_np[row][:count]]
            self.stats["advanced_reqs"] += 1
            self.stats["advanced_tokens"] += len(tokens)
            try:
                entry.grammar.accept_tokens(entry.req_id, tokens)
            except Exception as exc:  # pragma: no cover - defensive
                raise ShadowError(f"advance failed for request {entry.req_id}: {exc}") from exc
            entry.state_key = _mix_key(entry.state_key, tokens)
            entry.clear_steps()

    def _new_bitmask(self) -> Any:
        if self._bitmask is None:
            import xgrammar

            self._bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)
        return self._bitmask

    def valid_ids(self, req_index: int, *, step: int = 0) -> TokenSet | None:
        """Return the next-token set at ``step``, or ``None`` for full fallback."""

        self.stats["valid_calls"] += 1
        entry = self.entries.get(int(req_index))
        if entry is None:
            self.stats["valid_missing"] += 1
            return None
        if step > 0 and step not in entry.step_grammars:
            # The fork for this position was unavailable (terminated grammar or
            # an unsupported transition). Using the step-0 state here would
            # constrain the draft to the wrong position; fall back instead.
            self.stats["valid_missing"] += 1
            return None
        grammar = entry.grammar_for(step)
        state_key = entry.key_for(step)
        with _STATE_CACHE_LOCK:
            if state_key in _STATE_CACHE:
                cached = _STATE_CACHE[state_key]
                _STATE_CACHE.move_to_end(state_key)
                self.stats["valid_state_hits"] += 1
                return cached
        try:
            if grammar.is_terminated():
                self.stats["valid_terminated"] += 1
                _cache_state(state_key, None)
                return None
            bitmask = self._new_bitmask()
            grammar.fill_bitmask(bitmask, 0)
            row = np.asarray(bitmask[0].numpy(), dtype=np.int32)
        except Exception as exc:  # pragma: no cover - defensive
            self.stats["valid_errors"] += 1
            raise ShadowError(f"bitmask failed for request {entry.req_id}: {exc}") from exc
        if count_valid_ids(row, self.vocab_size) > self.max_ids:
            self.stats["valid_too_wide"] += 1
            _cache_state(state_key, None)
            return None
        self.stats["valid_hits"] += 1
        result = _cached_valid_ids(row, self.vocab_size)
        _cache_state(state_key, result)
        return result

    def fork_step(self, req_index: int, token_id: int, *, step: int) -> bool:
        """Fork the current step matcher and accept one proposed draft token."""

        entry = self.entries.get(int(req_index))
        if entry is None:
            self.stats["fork_failures"] += 1
            return False
        try:
            parent = entry.grammar_for(step)
            if parent.is_terminated():
                self.stats["fork_failures"] += 1
                return False
            child = _fork_grammar(parent)
            if child is None:
                self.stats["fork_failures"] += 1
                return False
            if not child.accept_tokens(entry.req_id, [int(token_id)]):
                self.stats["fork_accept_failures"] += 1
                return False
        except Exception:  # pragma: no cover - defensive
            return False
        entry.step_grammars[step + 1] = child
        entry.step_keys[step + 1] = _mix_key(entry.key_for(step), [int(token_id)])
        self.stats["forks"] += 1
        return True


def _fork_grammar(grammar: Any) -> Any | None:
    """Clone an ``XgrammarGrammar`` (or return None for unknown backends)."""

    try:
        from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
    except ImportError:  # pragma: no cover - vLLM optional
        return None
    if not isinstance(grammar, XgrammarGrammar):
        return None
    return XgrammarGrammar(
        vocab_size=grammar.vocab_size,
        matcher=grammar.matcher.fork(),
        ctx=grammar.ctx,
        _is_terminated=grammar.is_terminated(),
    )


def _cache_state(key: bytes, value: TokenSet | None) -> None:
    with _STATE_CACHE_LOCK:
        _STATE_CACHE[key] = value
        while len(_STATE_CACHE) > _STATE_CACHE_MAX:
            _STATE_CACHE.popitem(last=False)


__all__ = ["GrammarShadowRegistry", "ShadowError"]
