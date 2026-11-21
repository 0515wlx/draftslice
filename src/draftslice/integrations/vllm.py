"""vLLM integration.

This is a strict-mode first integration:

- a custom vLLM speculator wraps the built-in MTP or EAGLE-3 speculator;
- the structured-output bitmask is captured from the worker;
- draft step 0 slices the draft LM head to the grammar-valid rows;
- an opt-in registry screener can further narrow those rows with constrained
  ``A := A ∩ V`` semantics (the target verifier still enforces the grammar);
- later draft steps, TP > 1, quantized heads, and unsupported dtypes fall back
  to the full draft head.

The integration is registered through the ``vllm.general_plugins`` entry point.
It deliberately does not import vLLM from the core package.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..plan.types import TokenSet
from .vllm_grammar import (
    GrammarSnapshot,
    current_grammar_snapshot,
    install_grammar_bridge,
)
from .vllm_logits import build_draft_logits

_REGISTERED = False
_SHADOW_PATCHED = False
_TIMING = {"observe": 0.0, "valid": 0.0, "build": 0.0, "screen": 0.0, "calls": 0}
_LOCK = threading.Lock()


class VllmIntegrationError(RuntimeError):
    """Raised when the vLLM integration cannot be installed."""


@dataclass
class DraftSliceVllmStats:
    calls: int = 0
    sliced_calls: int = 0
    fallback_calls: int = 0
    sliced_rows: int = 0
    full_rows: int = 0
    candidate_total: int = 0
    no_snapshot_calls: int = 0
    nonzero_step_calls: int = 0
    no_valid_sets_calls: int = 0
    slice_disabled_calls: int = 0
    propose_calls: int = 0
    propose_shadow_calls: int = 0
    graph_replay_calls: int = 0
    screener_name: str | None = None
    screener_calls: int = 0
    screener_rows: int = 0
    screener_dropped: int = 0
    screener_empty_rows: int = 0
    screener_errors: int = 0
    mask_stride: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)

    def record(self, *, sliced: bool, stats: Any = None) -> None:
        self.calls += 1
        if sliced:
            self.sliced_calls += 1
            if stats is not None:
                self.sliced_rows += stats.sliced_rows
                self.full_rows += stats.full_rows
                self.candidate_total += stats.candidate_total
        else:
            self.fallback_calls += 1

    def record_error(self, message: str) -> None:
        self.last_error = message
        self.errors.append(message)
        self.errors = self.errors[-8:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "sliced_calls": self.sliced_calls,
            "fallback_calls": self.fallback_calls,
            "sliced_rows": self.sliced_rows,
            "full_rows": self.full_rows,
            "candidate_total": self.candidate_total,
            "no_snapshot_calls": self.no_snapshot_calls,
            "nonzero_step_calls": self.nonzero_step_calls,
            "no_valid_sets_calls": self.no_valid_sets_calls,
            "slice_disabled_calls": self.slice_disabled_calls,
            "propose_calls": self.propose_calls,
            "propose_shadow_calls": self.propose_shadow_calls,
            "graph_replay_calls": self.graph_replay_calls,
            "screener_name": self.screener_name,
            "screener_calls": self.screener_calls,
            "screener_rows": self.screener_rows,
            "screener_dropped": self.screener_dropped,
            "screener_empty_rows": self.screener_empty_rows,
            "screener_errors": self.screener_errors,
            "mask_stride": self.mask_stride,
            "last_error": self.last_error,
            "errors": list(self.errors),
        }


def _enabled() -> bool:
    return os.environ.get("DRAFTSLICE_VLLM", "1") not in ("0", "false", "False")


def _slicing_enabled() -> bool:
    """Whether to enable the current draft-step-0 slice path.

    The vLLM grammar bitmask is delivered for the target sample in the same
    iteration, one grammar state ahead of the draft token. Using it directly
    drops valid draft tokens on changing grammars, so the safe default is off
    until a worker-side grammar shadow (or scheduler-provided next-state set)
    exists. Set ``DRAFTSLICE_VLLM_SLICE=1`` only for experiments.
    """

    return os.environ.get("DRAFTSLICE_VLLM_SLICE", "0") in ("1", "true", "True")


def _decode_uses_full_cudagraph(vllm_config: Any) -> bool:
    """Whether the draft decode loop runs as a captured CUDA graph.

    In that mode ``propose`` is a single replay and Python never calls
    ``sample_draft``/``compute_draft_logits`` for the draft steps, so the shadow
    and the slice cannot update per step. Slicing must stay a full-head fallback
    until a static candidate buffer with a pre-replay update exists.
    """

    compilation = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    if mode is None:
        return False
    try:
        from vllm.config.compilation import CUDAGraphMode

        decode = getattr(mode, "decode_mode", None)
        if callable(decode):
            return decode() == CUDAGraphMode.FULL
        return mode == CUDAGraphMode.FULL
    except Exception:  # pragma: no cover - older/newer vLLM configs
        return False


def _max_slice_rows(vocab_size: int) -> int:
    raw = os.environ.get("DRAFTSLICE_VLLM_MAX_ROWS")
    if raw is None:
        return min(vocab_size, 32_768)
    try:
        value = int(raw)
    except ValueError:
        return min(vocab_size, 32_768)
    return max(1, min(vocab_size, value))


_SCREENER_OFF = {"", "0", "false", "off", "none"}


@dataclass(frozen=True, slots=True)
class ScreenerConfig:
    """Opt-in configuration for a registry candidate source."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScreenedValidSets:
    """Result of intersecting screened candidates with exact valid sets."""

    sets: list[TokenSet | None]
    rows: int
    dropped: int
    empty_rows: int


def _screener_config() -> ScreenerConfig | None:
    """Read the screener name and JSON kwargs from the environment.

    Screening is opt-in because it drops grammar-valid tokens from the draft
    head. The target verifier still enforces the grammar exactly, but the draft
    distribution is approximate, so the default remains the full/exact path.
    """

    raw_name = os.environ.get("DRAFTSLICE_VLLM_SCREENER", "").strip()
    if raw_name.lower() in _SCREENER_OFF:
        return None
    raw_args = os.environ.get("DRAFTSLICE_VLLM_SCREENER_ARGS", "").strip()
    args: dict[str, Any] = {}
    if raw_args:
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise VllmIntegrationError(
                f"DRAFTSLICE_VLLM_SCREENER_ARGS is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise VllmIntegrationError(
                "DRAFTSLICE_VLLM_SCREENER_ARGS must be a JSON object"
            )
        args = parsed
    return ScreenerConfig(name=raw_name, args=args)


def _screen_valid_sets(
    valid_sets: list[TokenSet | None],
    candidates: list[TokenSet],
) -> ScreenedValidSets:
    """Apply ``A := A ∩ V`` row by row.

    Rows without an exact valid set (``None``) stay on the full-head fallback,
    because there is no universe to intersect with. An empty intersection also
    falls back to the full head instead of silently dropping every candidate.
    """

    if len(candidates) != len(valid_sets):
        raise VllmIntegrationError(
            f"screener returned {len(candidates)} rows for {len(valid_sets)} states"
        )
    screened: list[TokenSet | None] = []
    rows = 0
    dropped = 0
    empty_rows = 0
    for valid, candidate in zip(valid_sets, candidates, strict=True):
        if valid is None:
            screened.append(None)
            continue
        if not isinstance(candidate, TokenSet):
            raise VllmIntegrationError("screener must return TokenSet rows")
        selected = candidate.intersection(valid)
        rows += 1
        dropped += len(valid) - len(selected)
        if selected.is_empty:
            screened.append(None)
            empty_rows += 1
        else:
            screened.append(selected)
    return ScreenedValidSets(
        sets=screened,
        rows=rows,
        dropped=dropped,
        empty_rows=empty_rows,
    )



class DraftSliceSpeculatorMixin:
    """Mixin for vLLM EAGLE-3/MTP speculators.

    The mixin is intentionally defensive: any unsupported condition falls back to
    the built-in full-head path. Structured output validity is enforced by the
    target's grammar bitmask; DraftSlice only avoids computing rows that the
    grammar can never sample for draft step 0.
    """

    _draftslice_allow_slicing = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._draftslice_enabled = _enabled()
        self._draftslice_slicing = (
            self._draftslice_enabled and _slicing_enabled() and self._draftslice_allow_slicing
        )
        self._draftslice_stats = DraftSliceVllmStats()
        self._draftslice_graph_replay = _decode_uses_full_cudagraph(
            getattr(self, "vllm_config", None)
        )
        if self._draftslice_graph_replay:
            # CUDA-graph replay skips the Python draft hooks entirely. Emitting
            # sliced logits once during capture would freeze a stale candidate
            # set into the graph, so keep the full head as the only safe path.
            self._disable_graph_slicing()
        self._draftslice_snapshot: GrammarSnapshot | None = None
        self._draftslice_input_batch: Any | None = None
        self._draftslice_idx_mapping: Any | None = None
        self._draftslice_idx_mapping_np: np.ndarray | None = None
        self._draftslice_step = 0
        self._draftslice_spec_step = 0
        self._draftslice_call_index = 0
        self._draftslice_req_by_state: dict[int, str] = {}
        self._draftslice_shadow: Any | None = None
        self._draftslice_dummy = False
        self._draftslice_weight: Any | None = None
        self._draftslice_weight_key: tuple[Any, ...] | None = None
        self._draftslice_screener: Any | None = None
        self._draftslice_screener_config: ScreenerConfig | None = None
        self._draftslice_screener_ready = False

        if os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1":
            import sys

            print(
                f"[draftslice] init local_argmax="
                f"{getattr(self, 'use_local_argmax_reduction', None)} "
                f"draft_logits={getattr(self, 'draft_logits', None) is not None}",
                file=sys.stderr,
                flush=True,
            )

    def draftslice_stats(self) -> dict[str, Any]:
        return self._draftslice_stats.to_dict()

    def _maybe_debug_stats(self) -> None:
        if os.environ.get("DRAFTSLICE_VLLM_DEBUG") != "1":
            return
        import sys

        stats = self._draftslice_stats.to_dict()
        if self._draftslice_shadow is not None:
            stats["shadow"] = self._draftslice_shadow.debug_dict()
        if _TIMING["calls"]:
            stats["timing_ms"] = {
                key: round(value * 1000.0, 3)
                for key, value in _TIMING.items()
                if key != "calls"
            }
        print(f"[draftslice] {stats}", file=sys.stderr, flush=True)

    def _disable_graph_slicing(self) -> None:
        if not self._draftslice_slicing:
            return
        self._draftslice_slicing = False
        if os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1":
            import sys

            print(
                "[draftslice] draft decode uses a FULL CUDA graph; "
                "slicing disabled (eager-only)",
                file=sys.stderr,
                flush=True,
            )

    def capture(self) -> Any:
        # Second line of defence: if the config check above missed the resolved
        # mode, a non-None decode graph manager still means FULL decode replay.
        if getattr(self, "decode_cudagraph_manager", None) is not None:
            self._draftslice_graph_replay = True
            self._disable_graph_slicing()
        return super().capture()

    def reset_state(self) -> None:
        self._draftslice_stats = DraftSliceVllmStats()
        self._draftslice_snapshot = None
        self._draftslice_input_batch = None
        self._draftslice_idx_mapping = None
        self._draftslice_idx_mapping_np = None
        self._draftslice_step = 0
        self._draftslice_spec_step = 0
        self._draftslice_call_index = 0
        self._draftslice_req_by_state = {}
        if self._draftslice_shadow is not None:
            self._draftslice_shadow.clear()
        return super().reset_state()

    def register_shadow(self, req_index: int, sampling_params: Any) -> None:
        """Register a worker-side grammar shadow for a newly added request."""

        if not self._draftslice_slicing or sampling_params is None:
            return
        if self._draftslice_shadow is None:
            try:
                from .vllm_shadow import GrammarShadowRegistry

                vocab_size = int(
                    getattr(self, "vocab_size", 0)
                    or self.vllm_config.model_config.get_vocab_size()
                )
                self._draftslice_shadow = GrammarShadowRegistry(
                    self.vllm_config,
                    max_ids=_max_slice_rows(vocab_size),
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._draftslice_stats.record_error(f"shadow init: {exc}")
                return
        try:
            self._draftslice_shadow.register(int(req_index), sampling_params)
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"shadow register: {exc}")

    def observe_shadow_sampled(
        self,
        idx_mapping_np: np.ndarray,
        sampled_tokens_np: np.ndarray,
        num_sampled_np: np.ndarray,
    ) -> None:
        """Advance worker-side matchers by the tokens the target just sampled."""

        if self._draftslice_shadow is None:
            return
        t0 = time.perf_counter()
        try:
            self._draftslice_shadow.advance_sampled(
                np.asarray(idx_mapping_np),
                np.asarray(sampled_tokens_np),
                np.asarray(num_sampled_np),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"shadow advance: {exc}")
        _TIMING["observe"] += time.perf_counter() - t0

    def add_request(self, req_index: int) -> None:
        # A slot can be reused after preemption/finish; never keep the previous
        # request's grammar under the same index.
        if self._draftslice_shadow is not None:
            self._draftslice_shadow.drop_index(int(req_index))
        return super().add_request(req_index)

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        if self._draftslice_shadow is not None:
            self._draftslice_shadow.drop(finished_req_ids)
        return super().on_requests_finished(finished_req_ids)

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        if not self._draftslice_slicing:
            return super().propose(*args, **kwargs)
        self._draftslice_stats.propose_calls += 1
        if self._draftslice_shadow is not None and self._draftslice_shadow.entries:
            self._draftslice_stats.propose_shadow_calls += 1
        input_batch = args[0] if args else kwargs.get("input_batch")
        self._draftslice_dummy = bool(kwargs.get("dummy_run", False))
        self._draftslice_input_batch = input_batch
        self._draftslice_snapshot = current_grammar_snapshot()
        self._draftslice_call_index = 0
        self._draftslice_step = 0
        self._draftslice_spec_step = 0
        self._draftslice_req_by_state = {}
        self._draftslice_idx_mapping_np = None
        if self._draftslice_shadow is not None:
            self._draftslice_shadow.clear_steps()
        if input_batch is not None:
            try:
                # ``idx_mapping_np`` is the host mirror of the CUDA mapping and
                # is safe to read during CUDA-graph capture. Falling back to the
                # tensor copy is only for older engines.
                idx_np = getattr(input_batch, "idx_mapping_np", None)
                if idx_np is not None:
                    idx_np = np.asarray(idx_np)
                    self._draftslice_idx_mapping_np = idx_np
                    idx = idx_np.tolist()
                else:
                    self._draftslice_idx_mapping_np = None
                    idx = input_batch.idx_mapping.tolist()
                self._draftslice_req_by_state = {
                    int(state): req_id
                    for state, req_id in zip(idx, input_batch.req_ids, strict=False)
                }
            except Exception as exc:  # pragma: no cover - defensive
                self._draftslice_req_by_state = {}
                self._draftslice_idx_mapping_np = None
                self._draftslice_stats.record_error(f"idx_mapping: {exc}")
        try:
            return super().propose(*args, **kwargs)
        finally:
            self._draftslice_input_batch = None
            self._draftslice_snapshot = None
            self._draftslice_idx_mapping = None
            # Keep ``_draftslice_idx_mapping_np``: vLLM calls ``sample_draft``
            # after ``propose`` returns, and that is where the valid sets are
            # built. It is replaced on the next propose or engine reset.
            self._maybe_debug_stats()

    def sample_draft(
        self,
        hidden_states: Any,
        sample_src_positions: Any,
        idx_mapping: Any,
        temperature: Any,
        seeds: Any,
        draft_step: Any,
        draft_logits: Any,
        spec_step_idx: int = 0,
        num_steps: int = 1,
    ) -> Any:
        if not self._draftslice_slicing:
            return super().sample_draft(
                hidden_states,
                sample_src_positions,
                idx_mapping,
                temperature,
                seeds,
                draft_step,
                draft_logits,
                spec_step_idx=spec_step_idx,
                num_steps=num_steps,
            )
        self._draftslice_idx_mapping = idx_mapping
        # MTP/EAGLE in this engine calls sample_draft without spec_step_idx for
        # every draft position, so the per-propose call index is the position.
        self._draftslice_step = self._draftslice_call_index
        self._draftslice_call_index += 1
        self._draftslice_spec_step = self._draftslice_step
        if self._draftslice_snapshot is None:
            self._draftslice_snapshot = current_grammar_snapshot()
        result = super().sample_draft(
            hidden_states,
            sample_src_positions,
            idx_mapping,
            temperature,
            seeds,
            draft_step,
            draft_logits,
            spec_step_idx=spec_step_idx,
            num_steps=num_steps,
        )
        # For k>1, the next draft position must start from the grammar state
        # after this step's sampled draft token. Fork the worker shadow so each
        # position gets its own exact valid set.
        if (
            self._draftslice_shadow is not None
            and not self._draftslice_dummy
            and self._draftslice_step + 1
            < int(getattr(self, "num_speculative_steps", num_steps))
            and self._draftslice_idx_mapping_np is not None
        ):
            try:
                tokens = result.reshape(-1).tolist()
                states = self._draftslice_idx_mapping_np[: len(tokens)]
                for state, token in zip(states, tokens, strict=False):
                    self._draftslice_shadow.fork_step(
                        int(state), int(token), step=self._draftslice_step
                    )
            except Exception as exc:  # pragma: no cover - defensive
                self._draftslice_stats.record_error(f"shadow fork: {exc}")
        return result

    def _snapshot_valid_ids(self, state: int, *, max_ids: int, position: int) -> Any | None:
        if self._draftslice_snapshot is None:
            return None
        req_id = self._draftslice_req_by_state.get(int(state))
        if req_id is None:
            return None
        return self._draftslice_snapshot.valid_ids(
            req_id, position=position, max_ids=max_ids
        )

    def _valid_sets_for_rows(self, hidden_states: Any) -> list[Any] | None:
        rows = int(hidden_states.shape[0])
        shadow = self._draftslice_shadow
        try:
            # Prefer the host-side mirror captured from the input batch; the CUDA
            # tensor copy raises during CUDA-graph capture ("Cannot copy between
            # CPU and CUDA tensors during CUDA graph capture").
            if self._draftslice_idx_mapping_np is not None:
                if self._draftslice_idx_mapping_np.shape[0] < rows:
                    return None
                states = self._draftslice_idx_mapping_np[:rows].tolist()
            elif self._draftslice_idx_mapping is not None:
                states = self._draftslice_idx_mapping[:rows].tolist()
            else:
                return None
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"idx_mapping slice: {exc}")
            return None

        shadow = self._draftslice_shadow
        if shadow is not None and any(shadow.has(int(state)) for state in states):
            # The shadow was advanced by the actual sampled tokens, so its set is
            # the exact state for the next draft position. Rows without a shadow
            # (free-form or unsupported grammar) fall back to the snapshot.
            snapshot_position = 0
            max_ids = _max_slice_rows(
                int(getattr(self._draftslice_snapshot, "vocab_size", shadow.vocab_size))
            )
            if self._draftslice_snapshot is not None:
                self._draftslice_stats.mask_stride = self._draftslice_snapshot.mask_stride
                snapshot_position = min(
                    self._draftslice_spec_step + 1,
                    self._draftslice_snapshot.mask_stride - 1,
                )
            t0 = time.perf_counter()
            result = [
                shadow.valid_ids(int(state), step=self._draftslice_spec_step)
                if shadow.has(int(state))
                else self._snapshot_valid_ids(
                    int(state), max_ids=max_ids, position=snapshot_position
                )
                for state in states
            ]
            _TIMING["valid"] += time.perf_counter() - t0
            _TIMING["calls"] += 1
            return result

        if (
            self._draftslice_snapshot is None
            or not self._draftslice_req_by_state
        ):
            return None
        # The captured bitmask has one row per speculative position: row 0 is
        # the bonus/current token, row p is draft position p. Draft step 0 must
        # use position 1, not position 0.
        position = min(
            self._draftslice_spec_step + 1,
            self._draftslice_snapshot.mask_stride - 1,
        )
        self._draftslice_stats.mask_stride = self._draftslice_snapshot.mask_stride
        max_ids = _max_slice_rows(self._draftslice_snapshot.vocab_size)
        return [
            self._snapshot_valid_ids(int(state), max_ids=max_ids, position=position)
            for state in states
        ]

    def _ensure_screener(self) -> None:
        """Build the configured registry screener once per speculator."""

        if self._draftslice_screener_ready:
            return
        self._draftslice_screener_ready = True
        try:
            config = _screener_config()
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"screener config: {exc}")
            return
        if config is None:
            return
        try:
            from ..screen import get_screener

            screener = get_screener(config.name, **config.args)
        except Exception as exc:
            self._draftslice_stats.record_error(
                f"screener init {config.name!r}: {exc}"
            )
            return
        self._draftslice_screener = screener
        self._draftslice_screener_config = config
        if os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1":
            import sys

            print(
                f"[draftslice] screener={config.name!r} args={config.args} "
                "(constrained: A := A ∩ V)",
                file=sys.stderr,
                flush=True,
            )

    def _apply_screener(
        self,
        hidden_states: Any,
        valid_sets: list[TokenSet | None],
    ) -> list[TokenSet | None]:
        """Intersect screened candidates with the exact valid sets.

        This is constrained-calibrated semantics: only candidate selection is
        approximate, and the intersection keeps every proposed draft token
        inside the grammar. Any screener failure falls back to the exact sets.
        """

        self._ensure_screener()
        screener = self._draftslice_screener
        config = self._draftslice_screener_config
        if screener is None or config is None:
            return valid_sets
        self._draftslice_stats.screener_name = config.name
        t0 = time.perf_counter()
        try:
            candidates = screener.candidates(hidden_states)
            result = _screen_valid_sets(valid_sets, candidates)
        except Exception as exc:
            self._draftslice_stats.screener_errors += 1
            self._draftslice_stats.record_error(f"screener {config.name!r}: {exc}")
            return valid_sets
        _TIMING["screen"] += time.perf_counter() - t0
        self._draftslice_stats.screener_calls += 1
        self._draftslice_stats.screener_rows += result.rows
        self._draftslice_stats.screener_dropped += result.dropped
        self._draftslice_stats.screener_empty_rows += result.empty_rows
        return result.sets

    def _lm_head_weight(self) -> Any | None:
        model = getattr(self, "model", None)
        if model is None:
            return None
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            return None
        try:
            import torch
        except ImportError:  # pragma: no cover - vLLM requires torch
            return None
        if getattr(lm_head, "tp_size", 1) != 1:
            return None
        if getattr(lm_head, "quant_config", None) is not None:
            return None
        weight = getattr(lm_head, "weight", None)
        if weight is None or weight.device.type != "cuda":
            return None
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return None
        key = (weight.data_ptr(), weight.dtype, tuple(weight.shape))
        if self._draftslice_weight is None or self._draftslice_weight_key != key:
            self._draftslice_weight = weight
            self._draftslice_weight_key = key
        return self._draftslice_weight

    def _logits_postprocess(self) -> tuple[float, float | None, bool]:
        processor = getattr(getattr(self, "model", None), "logits_processor", None)
        if processor is None:
            return 1.0, None, False
        if getattr(processor, "logits_as_input", False):
            return 1.0, None, False
        scale = float(getattr(processor, "scale", 1.0))
        soft_cap = getattr(processor, "soft_cap", None)
        return scale, soft_cap, True

    def compute_draft_logits(self, hidden_states: Any, spec_step_idx: int) -> Any:
        base_compute = super().compute_draft_logits
        if not self._draftslice_enabled or not self._draftslice_slicing:
            if self._draftslice_graph_replay:
                self._draftslice_stats.graph_replay_calls += 1
            else:
                self._draftslice_stats.slice_disabled_calls += int(self._draftslice_enabled)
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)
        if self._draftslice_dummy:
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)
        if self._draftslice_snapshot is None:
            self._draftslice_snapshot = current_grammar_snapshot()
        if self._draftslice_shadow is None and self._draftslice_spec_step != 0:
            self._draftslice_stats.nonzero_step_calls += 1
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)
        if (
            self._draftslice_snapshot is None
            and self._draftslice_shadow is None
        ) or (
            self._draftslice_idx_mapping is None
            and self._draftslice_idx_mapping_np is None
        ):
            self._draftslice_stats.no_snapshot_calls += 1
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        valid_sets = self._valid_sets_for_rows(hidden_states)
        if valid_sets is None or not any(token_set is not None for token_set in valid_sets):
            self._draftslice_stats.no_valid_sets_calls += 1
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        valid_sets = self._apply_screener(hidden_states, valid_sets)
        if not any(token_set is not None for token_set in valid_sets):
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        weight = self._lm_head_weight()
        if weight is None:
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        scale, soft_cap, supported = self._logits_postprocess()
        if not supported:
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        vocab_size = min(int(getattr(self, "vocab_size", weight.shape[0])), weight.shape[0])
        t0 = time.perf_counter()
        try:
            logits, stats = build_draft_logits(
                hidden_states,
                weight[:vocab_size],
                valid_sets,
                vocab_size=vocab_size,
                full_logits=lambda rows: base_compute(rows, spec_step_idx),
                max_slice_rows=_max_slice_rows(vocab_size),
                scale=scale,
                soft_cap=soft_cap,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"{type(exc).__name__}: {exc}")
            self._draftslice_stats.record(sliced=False)
            return base_compute(hidden_states, spec_step_idx)

        _TIMING["build"] += time.perf_counter() - t0
        self._draftslice_stats.record(sliced=True, stats=stats)
        return logits


def _build_speculators() -> tuple[type, type, type, type, type, type]:
    try:
        from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
        from vllm.v1.worker.gpu.spec_decode.eagle.config import Eagle3SpeculatorConfig
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
        from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
    except ImportError as exc:
        raise ImportError("vLLM is required for DraftSlice's vLLM integration") from exc

    class DraftSliceMTPSpeculator(DraftSliceSpeculatorMixin, MTPSpeculator):
        pass

    class DraftSliceEagle3Speculator(DraftSliceSpeculatorMixin, EagleSpeculator):
        pass

    class DraftSliceDFlashSpeculator(DraftSliceSpeculatorMixin, DFlashSpeculator):
        # DFlash uses one forward pass for the whole block; a single captured
        # bitmask is not a per-position grammar state. Keep it a safe pass-through
        # until the block-position plan contract is implemented.
        _draftslice_allow_slicing = False

    class DraftSliceMTPConfig(MTPSpeculatorConfig):
        speculator_class = "draftslice.integrations.vllm.DraftSliceMTPSpeculator"

    class DraftSliceEagle3Config(Eagle3SpeculatorConfig):
        speculator_class = "draftslice.integrations.vllm.DraftSliceEagle3Speculator"

    class DraftSliceDFlashConfig(DFlashSpeculatorConfig):
        speculator_class = "draftslice.integrations.vllm.DraftSliceDFlashSpeculator"

    return (
        DraftSliceMTPSpeculator,
        DraftSliceEagle3Speculator,
        DraftSliceDFlashSpeculator,
        DraftSliceMTPConfig,
        DraftSliceEagle3Config,
        DraftSliceDFlashConfig,
    )


def _install_shadow_bridge() -> None:
    """Teach the worker to keep a grammar matcher per request.

    The model runner already owns the request lifecycle and the sampler output,
    so two small wrappers are enough: register a shadow when a request is added,
    and advance it with the exact sampled tokens before the speculator proposes.
    """

    global _SHADOW_PATCHED
    if _SHADOW_PATCHED:
        return
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:  # pragma: no cover - vLLM optional
        return

    original_add_requests = GPUModelRunner.add_requests

    def add_requests(self: Any, scheduler_output: Any) -> Any:
        result = original_add_requests(self, scheduler_output)
        speculator = getattr(self, "speculator", None)
        if speculator is None or not hasattr(speculator, "register_shadow"):
            return result
        req_states = getattr(self, "req_states", None)
        index = getattr(req_states, "req_id_to_index", {}) if req_states else {}
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()):
            req_index = index.get(getattr(new_req, "req_id", None))
            if req_index is None:
                continue
            speculator.register_shadow(
                int(req_index), getattr(new_req, "sampling_params", None)
            )
        return result

    GPUModelRunner.add_requests = add_requests

    original_postprocess = GPUModelRunner.postprocess_sampled

    def postprocess_sampled(
        self: Any,
        idx_mapping: Any,
        sampled_tokens: Any,
        num_sampled: Any,
        num_rejected: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_postprocess(
            self, idx_mapping, sampled_tokens, num_sampled, num_rejected, *args, **kwargs
        )
        speculator = getattr(self, "speculator", None)
        if speculator is None or not hasattr(speculator, "observe_shadow_sampled"):
            return result
        shadow = getattr(speculator, "_draftslice_shadow", None)
        if shadow is None or not shadow.entries:
            return result
        try:
            speculator.observe_shadow_sampled(
                idx_mapping.detach().cpu().numpy(),
                sampled_tokens.detach().cpu().numpy(),
                num_sampled.detach().cpu().numpy(),
            )
        except Exception:  # pragma: no cover - the shadow is best-effort
            pass
        return result

    GPUModelRunner.postprocess_sampled = postprocess_sampled
    _SHADOW_PATCHED = True


def register() -> None:
    """Register DraftSlice with vLLM via the general-plugin entry point."""

    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return
        try:
            from vllm.v1.worker.gpu.spec_decode.speculator_config import (
                register_speculative_method,
                registered_speculative_methods,
            )
        except ImportError as exc:
            raise ImportError("vLLM is required for DraftSlice's vLLM integration") from exc

        (
            DraftSliceMTPSpeculator,
            DraftSliceEagle3Speculator,
            DraftSliceDFlashSpeculator,
            DraftSliceMTPConfig,
            DraftSliceEagle3Config,
            DraftSliceDFlashConfig,
        ) = _build_speculators()
        globals()["DraftSliceMTPSpeculator"] = DraftSliceMTPSpeculator
        globals()["DraftSliceEagle3Speculator"] = DraftSliceEagle3Speculator
        globals()["DraftSliceDFlashSpeculator"] = DraftSliceDFlashSpeculator
        globals()["DraftSliceMTPConfig"] = DraftSliceMTPConfig
        globals()["DraftSliceEagle3Config"] = DraftSliceEagle3Config
        globals()["DraftSliceDFlashConfig"] = DraftSliceDFlashConfig

        existing = set(registered_speculative_methods())
        if "draftslice_mtp" not in existing:
            register_speculative_method("draftslice_mtp", DraftSliceMTPConfig)
        if "draftslice_eagle3" not in existing:
            register_speculative_method("draftslice_eagle3", DraftSliceEagle3Config)
        if "draftslice_dflash" not in existing:
            register_speculative_method("draftslice_dflash", DraftSliceDFlashConfig)

        try:
            install_grammar_bridge()
        except Exception:
            # The speculators still work; without the bridge they fall back to
            # the full head for structured requests.
            pass
        try:
            _install_shadow_bridge()
        except Exception:
            # The worker shadow is an exactness improvement, not a hard
            # dependency; the scheduler snapshot remains the fallback.
            pass
        _REGISTERED = True


__all__ = [
    "DraftSliceSpeculatorMixin",
    "DraftSliceVllmStats",
    "ScreenerConfig",
    "VllmIntegrationError",
    "register",
]
