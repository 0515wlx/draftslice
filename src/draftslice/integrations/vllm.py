"""vLLM integration.

This is a strict-mode first integration:

- a custom vLLM speculator wraps the built-in MTP or EAGLE-3 speculator;
- the structured-output bitmask is captured from the worker;
- draft step 0 slices the draft LM head to the grammar-valid rows;
- an opt-in registry screener can further narrow those rows with constrained
  ``A := A ∩ V`` semantics (the target verifier still enforces the grammar);
- later draft steps, quantized heads, and unsupported dtypes fall back to the
  full draft head;
- TP>1 slices each rank's contiguous vocab shard and scatters the logits back to
  global token-id columns; a rank whose valid set has no local token uses the
  full head masked to V;
- every decision emits a ``DraftContract`` with the fail-closed reason.

The integration is registered through the ``vllm.general_plugins`` entry point.
It deliberately does not import vLLM from the core package.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..plan.types import (
    DraftContract,
    Execution,
    Guarantee,
    ServingPoint,
    TokenSet,
)
from ..runtime.policy import PolicyDecision, RowFacts, StepFacts
from ..runtime.profile import batch_bucket
from ..runtime.trace import (
    StepTrace,
    TraceDecision,
    TraceIdentity,
    TraceRegistry,
    TraceTiming,
)
from .base import (
    AdapterPolicy,
    AdapterPolicyError,
    TraceSampler,
    filter_valid_sets,
    load_adapter_policy,
    make_registry,
    trace_sample_from_env,
    workload_config_from_env,
)
from .base import row_share as compute_row_share
from .head_shard import HeadShard
from .quant_dequant import (
    dequantize_rows,
    describe_quantized_head,
    is_quantized_weight,
)
from .vllm_fastmask import apply_packed_mask
from .vllm_grammar import (
    GrammarSnapshot,
    current_grammar_snapshot,
    install_grammar_bridge,
)
from .vllm_logits import (
    build_draft_logits,
    mask_logits_to_valid,
    upload_numpy,
)

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
    graph_mask_calls: int = 0
    full_mask_calls: int = 0
    screener_name: str | None = None
    screener_universe: str | None = None
    screener_calibration_id: str | None = None
    screener_delta: float | None = None
    screener_calls: int = 0
    screener_rows: int = 0
    screener_dropped: int = 0
    screener_empty_rows: int = 0
    screener_freeform_rows: int = 0
    block_step0_rows: int = 0
    block_allpos_rows: int = 0
    block_allpos_requests: int = 0
    block_allpos_fallbacks: int = 0
    routed_full_steps: int = 0
    routed_slice_steps: int = 0
    routed_full_calls: int = 0
    deterministic_rows: int = 0
    screener_errors: int = 0
    mask_stride: int = 0
    last_error: str | None = None
    errors: list[str] = field(default_factory=list)
    # Fail-closed observability: a histogram of why a step did not slice, plus
    # the last concrete DraftContract emitted by this speculator.
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    last_contract: dict[str, Any] | None = None
    tp_size: int = 1
    tp_rank: int = 0
    tp_shard_rows: int = 0
    tp_combined_calls: int = 0
    tp_agreed_full_rows: int = 0
    quantized_head: bool = False
    quantized_rows: int = 0
    # Production policy telemetry (P1). ``row_share`` is the sliced vocabulary
    # fraction; ``measured_head_share`` is only set by a benchmark harness.
    profile_mode: str = "off"
    profile_table: str = ""
    profile_hash: str = ""
    cell_id: str | None = None
    cell_hit: bool = False
    action: str = "builtin"
    policy_slice_steps: int = 0
    policy_full_steps: int = 0
    row_share: float = 0.0
    measured_head_share: float | None = None
    trace_count: int = 0
    # Actual per-step batch distribution (draft rows), as profile-table buckets.
    batch_buckets: dict[str, int] = field(default_factory=dict)

    def note_fallback(self, reason: str) -> None:
        """Record a fail-closed decision without flooding the error list."""

        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def record(self, *, sliced: bool, stats: Any = None) -> None:
        self.calls += 1
        if sliced:
            self.sliced_calls += 1
            if stats is not None:
                self.sliced_rows += stats.sliced_rows
                self.full_rows += stats.full_rows
                self.candidate_total += stats.candidate_total
                self.tp_agreed_full_rows += int(
                    getattr(stats, "agreed_full_rows", 0)
                )
                if getattr(stats, "combined", False):
                    self.tp_combined_calls += 1
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
            "graph_mask_calls": self.graph_mask_calls,
            "full_mask_calls": self.full_mask_calls,
            "screener_name": self.screener_name,
            "screener_universe": self.screener_universe,
            "screener_calibration_id": self.screener_calibration_id,
            "screener_delta": self.screener_delta,
            "screener_calls": self.screener_calls,
            "screener_rows": self.screener_rows,
            "screener_dropped": self.screener_dropped,
            "screener_empty_rows": self.screener_empty_rows,
            "screener_freeform_rows": self.screener_freeform_rows,
            "block_step0_rows": self.block_step0_rows,
            "block_allpos_rows": self.block_allpos_rows,
            "block_allpos_requests": self.block_allpos_requests,
            "block_allpos_fallbacks": self.block_allpos_fallbacks,
            "routed_full_steps": self.routed_full_steps,
            "routed_slice_steps": self.routed_slice_steps,
            "routed_full_calls": self.routed_full_calls,
            "deterministic_rows": self.deterministic_rows,
            "screener_errors": self.screener_errors,
            "mask_stride": self.mask_stride,
            "fallback_reasons": dict(self.fallback_reasons),
            "last_contract": self.last_contract,
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "tp_shard_rows": self.tp_shard_rows,
            "tp_combined_calls": self.tp_combined_calls,
            "tp_agreed_full_rows": self.tp_agreed_full_rows,
            "profile_mode": self.profile_mode,
            "profile_table": self.profile_table,
            "profile_hash": self.profile_hash,
            "cell_id": self.cell_id,
            "cell_hit": self.cell_hit,
            "action": self.action,
            "policy_slice_steps": self.policy_slice_steps,
            "policy_full_steps": self.policy_full_steps,
            "row_share": self.row_share,
            "measured_head_share": self.measured_head_share,
            "trace_count": self.trace_count,
            "batch_buckets": dict(self.batch_buckets),
            "quantized_head": self.quantized_head,
            "quantized_rows": self.quantized_rows,
            "last_error": self.last_error,
            "errors": list(self.errors),
        }


def _debug_on() -> bool:
    return os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1"


def _tp_debug(message: str) -> None:
    """Rank-tagged stderr trace of the TP collectives (debug builds only)."""

    if not _debug_on():
        return
    import sys

    print(f"[draftslice tp pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


def _enabled() -> bool:
    return os.environ.get("DRAFTSLICE_VLLM", "1") not in ("0", "false", "False")


def _slicing_flag() -> bool | None:
    """Tri-state of the legacy switch: True/False when set, None when unset."""

    raw = os.environ.get("DRAFTSLICE_VLLM_SLICE")
    if raw is None or raw == "":
        return None
    return raw in ("1", "true", "True")


def _slicing_enabled() -> bool:
    """Whether to enable the current draft-step-0 slice path.

    The vLLM grammar bitmask is delivered for the target sample in the same
    iteration, one grammar state ahead of the draft token. Using it directly
    drops valid draft tokens on changing grammars, so the safe default is off
    until a worker-side grammar shadow (or scheduler-provided next-state set)
    exists. Set ``DRAFTSLICE_VLLM_SLICE=1`` only for experiments.
    """

    return os.environ.get("DRAFTSLICE_VLLM_SLICE", "0") in ("1", "true", "True")


def _full_mask_enabled() -> bool:
    """Diagnostic control: full head + exact grammar mask, no slicing.

    This is used by the benefit-isolation experiments. It keeps the same worker
    shadow and candidate sets as the slice path, so the only difference is
    whether the draft head scores all vocabulary rows or only valid rows.
    """

    return os.environ.get("DRAFTSLICE_VLLM_FULL_MASK", "0") in ("1", "true", "True")


def _router_enabled() -> bool:
    """Opt-in workload router: skip the slice bookkeeping on free-form batches.

    A batch with no structural request can never use a grammar valid set, so the
    wrapper's shadow capture, per-step mapping, and logits checks are pure
    overhead. When enabled, those batches take the stock speculator path.
    """

    return os.environ.get("DRAFTSLICE_VLLM_ROUTER", "0") in ("1", "true", "True")


def _block_allpos_enabled() -> bool:
    """Opt-in DFlash all-position grammar masking.

    DFlash proposes a whole block in one forward. Only block position 0 has an
    exact state before sampling today; positions > 0 are left on the full head.
    Under greedy sampling the next token is the argmax, so the block can be
    masked sequentially: read position ``p``'s exact set, mask, take its argmax,
    fork the matcher, then move to ``p+1``. The engine samples from the masked
    logits afterwards and reproduces the same argmaxes.
    """

    return os.environ.get("DRAFTSLICE_VLLM_BLOCK_ALLPOS", "0") in ("1", "true", "True")


def _block_allpos_depth() -> int | None:
    """Optional cap on how many block positions get an exact grammar mask.

    Cost scales with ``batch * depth`` because each position needs a grammar
    bitmask and an argmax to reach the next state. Positions past the cap stay
    on the full head (a safe fallback).
    """

    raw = os.environ.get("DRAFTSLICE_VLLM_BLOCK_ALLPOS_DEPTH", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(1, value)


def _block_allpos_fast() -> bool:
    """Use one packed bitmask upload and one fused Triton launch per position."""

    return os.environ.get("DRAFTSLICE_VLLM_BLOCK_ALLPOS_FAST", "0") in (
        "1",
        "true",
        "True",
    )


def _deterministic_injection_enabled() -> bool:
    """Skip the draft head for rows whose exact valid set has one token."""

    return os.environ.get("DRAFTSLICE_VLLM_DETERMINISTIC", "0") in (
        "1",
        "true",
        "True",
    )


def _block_allpos_max_batch() -> int:
    """Batch ceiling for all-position masking.

    Measured on A100/Qwen3.5-DFlash: all-position masking wins at B=1 (tau
    2.67->5.1, +23% tok/s) but loses at B>=8 because the per-position grammar
    masks and argmaxes cost more than the extra acceptance earns. Larger
    batches fall back to the position-0 slice, which is the better high-batch
    point.
    """

    raw = os.environ.get("DRAFTSLICE_VLLM_BLOCK_ALLPOS_MAX_BATCH", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1




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
    artifact: str | None = None
    universe: str = "valid"
    calibration_id: str | None = None
    delta: float | None = None


@dataclass(frozen=True, slots=True)
class ScreenedValidSets:
    """Result of intersecting screened candidates with exact valid sets."""

    sets: list[TokenSet | None]
    rows: int
    dropped: int
    empty_rows: int
    freeform_rows: int = 0


def _screener_config() -> ScreenerConfig | None:
    """Read the screener name and JSON kwargs from the environment.

    Screening is opt-in because it drops grammar-valid tokens from the draft
    head. The target verifier still enforces the grammar exactly, but the draft
    distribution is approximate, so the default remains the full/exact path.
    """

    raw_name = os.environ.get("DRAFTSLICE_VLLM_SCREENER", "").strip()
    artifact = os.environ.get("DRAFTSLICE_VLLM_SCREENER_ARTIFACT", "").strip() or None
    if raw_name.lower() in _SCREENER_OFF and artifact is None:
        return None
    universe = os.environ.get("DRAFTSLICE_VLLM_SCREENER_UNIVERSE", "valid").strip().lower()
    if universe not in ("valid", "vocab"):
        raise VllmIntegrationError(
            "DRAFTSLICE_VLLM_SCREENER_UNIVERSE must be 'valid' or 'vocab'"
        )
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
    return ScreenerConfig(
        name="" if artifact is not None and raw_name.lower() in _SCREENER_OFF else raw_name,
        args=args,
        artifact=artifact,
        universe=universe,
    )


def _screen_valid_sets(
    valid_sets: list[TokenSet | None],
    candidates: list[TokenSet],
    *,
    universe: str = "valid",
) -> ScreenedValidSets:
    """Apply ``A := A ∩ V`` row by row.

    With ``universe="valid"`` rows without an exact valid set (``None``) stay on
    the full-head fallback. With ``universe="vocab"`` those rows use ``A``
    directly (free-form), while rows that do have a grammar state still
    intersect with it so the adapter never proposes a grammar-invalid draft.
    An empty intersection always falls back to the full head.
    """

    if universe not in ("valid", "vocab"):
        raise VllmIntegrationError("universe must be 'valid' or 'vocab'")
    if len(candidates) != len(valid_sets):
        raise VllmIntegrationError(
            f"screener returned {len(candidates)} rows for {len(valid_sets)} states"
        )
    screened: list[TokenSet | None] = []
    rows = 0
    dropped = 0
    empty_rows = 0
    freeform_rows = 0
    for valid, candidate in zip(valid_sets, candidates, strict=True):
        if not isinstance(candidate, TokenSet):
            raise VllmIntegrationError("screener must return TokenSet rows")
        if valid is None:
            if universe == "vocab" and not candidate.is_empty:
                screened.append(candidate)
                freeform_rows += 1
            else:
                screened.append(None)
            continue
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
        freeform_rows=freeform_rows,
    )



class DraftSliceSpeculatorMixin:
    """Mixin for vLLM EAGLE-3/MTP speculators.

    The mixin is intentionally defensive: any unsupported condition falls back to
    the built-in full-head path. Structured output validity is enforced by the
    target's grammar bitmask; DraftSlice only avoids computing rows that the
    grammar can never sample for draft step 0.
    """

    _draftslice_allow_slicing = True
    _draftslice_route_full = False
    # Class-level defaults keep tests that construct via ``__new__`` safe.
    _draftslice_shard: HeadShard | None = None
    _draftslice_head_key: tuple[Any, ...] | None = None
    _draftslice_head_cache: tuple[Any, HeadShard] | None = None
    _draftslice_tp_plan: Any | None = None
    _draftslice_adapter_policy: AdapterPolicy | None = None
    _draftslice_policy_decision: PolicyDecision | None = None
    _draftslice_row_workloads: list[str] | None = None
    _draftslice_policy_hold: str | None = None
    _draftslice_traces: TraceRegistry | None = None
    _draftslice_trace_sampler: TraceSampler | None = None
    _draftslice_screener: Any | None = None
    _draftslice_screener_config: ScreenerConfig | None = None
    _draftslice_idx_mapping_np: np.ndarray | None = None
    _draftslice_local_argmax: bool = False
    _draftslice_graph_replay: bool = False
    _draftslice_graph_packed: Any | None = None
    _draftslice_graph_rows: Any | None = None
    _draftslice_graph_vocab: int = 0
    _draftslice_quant_head: Any | None = None
    _draftslice_dequant_weight: Any | None = None
    _draftslice_dequant_filled: set[int] = field(default_factory=set)
    # DFlash gets logits for a whole parallel block in one call. Only block
    # position 0 has an exact grammar state before sampling; later positions
    # must stay on the full head.
    _draftslice_block = False

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
            # Freezing candidate ids into the graph would make them stale. Use a
            # static packed bitmask instead: it is refreshed before every replay
            # and applied by the same captured Triton kernel.
            self._init_graph_mask()
        self._draftslice_local_argmax = bool(
            getattr(self, "use_local_argmax_reduction", False)
        )
        self._draftslice_shard: HeadShard | None = None
        self._draftslice_head_key: tuple[Any, ...] | None = None
        self._draftslice_head_cache: tuple[Any, HeadShard] | None = None
        self._draftslice_tp_plan: Any | None = None
        self._draftslice_adapter_policy = None
        self._draftslice_policy_decision = None
        self._draftslice_row_workloads = None
        try:
            self._draftslice_adapter_policy = load_adapter_policy(
                "vllm",
                model=self._policy_model_name(),
                method=self._policy_method_name(),
            )
        except AdapterPolicyError:
            # A configured table with a bad hash/file is a hard error, never a
            # silent built-in deployment.
            raise
        if self._draftslice_adapter_policy is not None and (
            self._draftslice_adapter_policy.uses_table
        ):
            self._draftslice_stats.profile_mode = (
                self._draftslice_adapter_policy.profile_mode
            )
            self._draftslice_stats.profile_table = (
                self._draftslice_adapter_policy.table_path
            )
            self._draftslice_stats.profile_hash = (
                self._draftslice_adapter_policy.table_hash
            )
            # The table replaces the blanket DRAFTSLICE_VLLM_SLICE switch. An
            # explicitly disabled env flag still wins (emergency stop).
            if _slicing_flag() is None and self._draftslice_enabled:
                self._draftslice_slicing = (
                    self._draftslice_allow_slicing
                    and self._draftslice_adapter_policy.profile_mode != "off"
                )
        self._draftslice_policy_hold: str | None = None
        self._draftslice_traces = make_registry()
        self._draftslice_trace_sampler = TraceSampler(trace_sample_from_env())
        self._draftslice_policy_identity = (
            self._policy_model_name() or "unknown",
            self._policy_method_name(),
            self._policy_quant(),
            self._tp_world_size(),
        )
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
        self._draftslice_draft_step: Any | None = None
        self._draftslice_route_full = False

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

    @staticmethod
    def _union_token_sets(valid_sets: Any) -> TokenSet | None:
        ids: set[int] = set()
        for token_set in valid_sets or ():
            if token_set is not None:
                ids.update(token_set.ids)
        return TokenSet(ids) if ids else None

    def _grammar_state_id(self) -> str | None:
        mapping = self._draftslice_idx_mapping_np
        if mapping is None or len(mapping) == 0:
            return None
        try:
            return str(int(mapping[0]))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _emit_contract(
        self,
        *,
        mode: str,
        valid_sets: Any = None,
        reason: str | None = None,
        logits_stats: Any = None,
        vocab_size: int = 0,
        execution_kind: str | None = None,
    ) -> None:
        """Make the current step's contract concrete and observable."""

        valid = self._union_token_sets(valid_sets)
        screener_active = (
            self._draftslice_screener is not None
            and self._draftslice_screener_config is not None
        )
        if mode == "full":
            guarantee = Guarantee.full()
            execution = Execution(kind="full")
            candidate_set = None
            row_share = None
        else:
            if screener_active:
                config = self._draftslice_screener_config
                guarantee = Guarantee.calibrated(
                    config.delta if config.delta is not None else 0.05,
                    universe="valid",
                    calibration_id=config.calibration_id,
                )
            elif mode == "mask_allpos":
                guarantee = Guarantee.exact_valid_set()
            else:
                guarantee = Guarantee.exact_valid_set()
            candidate_set = valid
            rows = int(getattr(logits_stats, "rows", 0))
            candidates = int(getattr(logits_stats, "candidate_total", 0))
            row_share = candidates / (rows * vocab_size) if rows and vocab_size else None
            execution = Execution(
                kind=execution_kind
                or ("slice_gemm" if self._draftslice_shard is not None else "slice_gemv"),
                rows=candidate_set,
            )
        source = "draftslice"
        if self._draftslice_shard is not None and self._draftslice_shard.tp_size > 1:
            source = f"draftslice_tp{self._draftslice_shard.tp_size}"
        elif screener_active:
            source = f"draftslice+{self._draftslice_screener_config.name}"
        contract = DraftContract(
            grammar_state_id=self._grammar_state_id(),
            valid_set=valid,
            candidate_set=candidate_set,
            mode=mode,
            execution=execution,
            guarantee=guarantee,
            serving=ServingPoint(
                k=int(getattr(self, "num_speculative_steps", 0) or 0),
                source=source,
                target_metric="throughput",
                row_share=row_share,
                measured_head_share=getattr(
                    self, "_draftslice_measured_head_share", None
                ),
            ),
            fallback_reason=reason,
        )
        self._draftslice_stats.last_contract = contract.to_dict()
        if reason is not None:
            self._draftslice_stats.note_fallback(reason)

    def _fallback(
        self,
        reason: str | None,
        *,
        valid_sets: Any = None,
        base_compute: Any,
        hidden_states: Any,
        spec_step_idx: int,
    ) -> Any:
        """Fail closed to the full head and record the exact reason."""

        if _debug_on():
            _tp_debug(
                f"fallback reason={reason!r} "
                f"step={getattr(self, '_draftslice_step', 0)} "
                f"rows={int(hidden_states.shape[0])}"
            )
        self._draftslice_stats.record(sliced=False)
        self._emit_contract(mode="full", valid_sets=valid_sets, reason=reason)
        return base_compute(hidden_states, spec_step_idx)

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

    def _init_graph_mask(self) -> None:
        """Allocate the static packed bitmask read by the captured graph."""

        try:
            import torch

            vocab = int(getattr(self, "vocab_size", 0) or 0)
            max_rows = int(getattr(self, "max_num_reqs", 0) or 0)
            if vocab <= 0 or max_rows <= 0:
                raise ValueError("missing vocab_size/max_num_reqs")
            device = getattr(self, "device", None) or torch.device("cuda")
            words = (vocab + 31) // 32
            self._draftslice_graph_vocab = vocab
            self._draftslice_graph_packed = torch.full(
                (max_rows, words), -1, dtype=torch.int32, device=device
            )
            self._draftslice_graph_rows = torch.arange(
                max_rows, dtype=torch.int64, device=device
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_slicing = False
            self._draftslice_stats.note_fallback("graph_mask_unavailable")
            self._draftslice_stats.record_error(f"graph mask init: {exc}")

    def _prepare_graph_mask(self, input_batch: Any) -> None:
        """Refresh the static bitmask from the exact step-0 shadow sets."""

        packed_buf = self._draftslice_graph_packed
        if packed_buf is None or input_batch is None:
            return
        shadow = self._draftslice_shadow
        if shadow is None or not shadow.entries:
            return
        mapping = getattr(input_batch, "idx_mapping_np", None)
        if mapping is None:
            return
        try:
            import torch

            packed_buf.fill_(-1)
            handles: dict[int, Any] = {}
            states = np.asarray(mapping).reshape(-1)
            for row, state in enumerate(states[: packed_buf.shape[0]]):
                state = int(state)
                if state < 0 or not shadow.has(state):
                    continue
                handle = shadow.begin_block(state)
                if handle is not None:
                    handles[row] = handle
            if handles:
                sub = shadow.batch_block_masks(
                    handles,
                    vocab_size=self._draftslice_graph_vocab,
                    device=packed_buf.device,
                )
                if sub is not None:
                    rows = torch.tensor(
                        sorted(handles), dtype=torch.int64, device=packed_buf.device
                    )
                    packed_buf[rows] = sub.reshape(len(handles), -1)
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"graph mask refresh: {exc}")

    def _disable_graph_slicing(self, *, reason: str = "full_cuda_graph") -> None:
        if not self._draftslice_slicing:
            return
        self._draftslice_slicing = False
        self._draftslice_stats.note_fallback(reason)
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
            if self._draftslice_graph_packed is None:
                self._init_graph_mask()
            else:
                self._draftslice_graph_packed.fill_(-1)
        return super().capture()

    def reset_state(self) -> None:
        self._draftslice_stats = DraftSliceVllmStats()
        adapter = self._draftslice_adapter_policy
        if adapter is not None and adapter.uses_table:
            self._draftslice_stats.profile_mode = adapter.profile_mode
            self._draftslice_stats.profile_table = adapter.table_path
            self._draftslice_stats.profile_hash = adapter.table_hash
        self._draftslice_shard = None
        self._draftslice_head_key = None
        self._draftslice_head_cache = None
        self._draftslice_quant_head = None
        self._draftslice_dequant_weight = None
        self._draftslice_dequant_filled = set()
        self._draftslice_snapshot = None
        self._draftslice_input_batch = None
        self._draftslice_idx_mapping = None
        self._draftslice_idx_mapping_np = None
        self._draftslice_step = 0
        self._draftslice_spec_step = 0
        self._draftslice_call_index = 0
        self._draftslice_req_by_state = {}
        self._draftslice_draft_step = None
        self._draftslice_route_full = False
        self._draftslice_policy_decision = None
        self._draftslice_policy_hold = None
        self._draftslice_row_workloads = None
        if self._draftslice_shadow is not None:
            self._draftslice_shadow.clear()
        return super().reset_state()

    def register_shadow(self, req_index: int, sampling_params: Any) -> None:
        """Register a worker-side grammar shadow for a newly added request."""

        if (
            not self._draftslice_slicing
            or not self._policy_allows_work()
            or sampling_params is None
        ):
            return
        if self._draftslice_shadow is None:
            try:
                from .vllm_shadow import GrammarShadowRegistry

                vocab_size = int(
                    getattr(self, "vocab_size", 0)
                    or self.vllm_config.model_config.get_vocab_size()
                )
                workload_default, workload_map = workload_config_from_env()
                self._draftslice_shadow = GrammarShadowRegistry(
                    self.vllm_config,
                    max_ids=_max_slice_rows(vocab_size),
                    workload_default=workload_default,
                    workload_map=workload_map,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._draftslice_stats.record_error(f"shadow init: {exc}")
                return
            if not self._draftslice_shadow.enabled:
                self._draftslice_stats.note_fallback("shadow_disabled")
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

        if self._draftslice_shadow is None or not self._policy_allows_work():
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

    def _batch_has_grammar(self, input_batch: Any) -> bool:
        """Cheap router predicate: does this step contain any grammar state?"""

        snapshot = current_grammar_snapshot()
        shadow = self._draftslice_shadow
        try:
            idx_np = getattr(input_batch, "idx_mapping_np", None)
            states = None if idx_np is None else np.asarray(idx_np).reshape(-1)
        except Exception:  # pragma: no cover - defensive
            return True
        if shadow is not None and shadow.entries:
            if states is None:
                return True
            if any(int(state) in shadow.entries for state in states):
                return True
        if snapshot is not None and snapshot.structured_req_ids:
            # vLLM captures a snapshot even for an all-free-form step, with an
            # empty structured set. Only a structured row counts as grammar.
            req_ids = set(getattr(input_batch, "req_ids", ()) or ())
            if not req_ids:
                return True
            return any(req_id in req_ids for req_id in snapshot.structured_req_ids)
        return False

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        if not self._draftslice_slicing:
            return super().propose(*args, **kwargs)
        adapter_policy = self._draftslice_adapter_policy
        hold = None if adapter_policy is None else adapter_policy.hold_reason
        if adapter_policy is not None and adapter_policy.uses_table and hold is not None:
            # observe/off/disabled-only execute the stock path with no shadow,
            # mapping, or valid-set work: the A/A gate needs byte-identical
            # built-in behaviour, not merely identical output.
            self._draftslice_stats.propose_calls += 1
            self._draftslice_stats.profile_table = adapter_policy.table_path
            self._draftslice_stats.profile_hash = adapter_policy.table_hash
            self._draftslice_policy_hold = hold.value
            self._record_policy(
                PolicyDecision(
                    action="builtin",
                    mode="builtin",
                    profile_mode=adapter_policy.profile_mode,
                    reason=hold,
                )
            )
            self._draftslice_stats.note_fallback(hold.value)
            self._draftslice_route_full = False
            return super().propose(*args, **kwargs)
        self._draftslice_stats.propose_calls += 1
        self._draftslice_policy_hold = None
        input_batch = args[0] if args else kwargs.get("input_batch")
        if self._draftslice_graph_replay:
            self._prepare_graph_mask(input_batch)
        if _router_enabled() and self._tp_world_size() == 1 and input_batch is not None:
            if not self._batch_has_grammar(input_batch):
                # No row can use a valid set: take the stock path without
                # shadow/mapping/logits bookkeeping.
                self._draftslice_route_full = True
                self._draftslice_stats.routed_full_steps += 1
                self._draftslice_input_batch = None
                self._draftslice_snapshot = None
                return super().propose(*args, **kwargs)
        self._draftslice_route_full = False
        self._draftslice_stats.routed_slice_steps += 1
        if self._draftslice_shadow is not None and self._draftslice_shadow.entries:
            self._draftslice_stats.propose_shadow_calls += 1
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
        if self._draftslice_route_full:
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
        self._draftslice_draft_step = draft_step
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
            and not self._draftslice_graph_replay
            and not self._draftslice_block
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

    def _block_valid_sets(self, hidden_states: Any) -> list[Any] | None:
        """Exact valid sets for DFlash block position 0 only.

        DFlash produces one logits tensor for every block position in a single
        forward, before any draft token is sampled. A later position's grammar
        state therefore does not exist yet. Position 0 is exact (the worker
        shadow is already advanced by the verified prefix); positions > 0 get
        ``None`` so they use the full head. This is deliberately *not* "use the
        position-0 set for the whole block", which would drop later-position
        valid tokens.

        Returns ``None`` when no row can be planned, so the caller takes the
        ordinary full-head fallback.
        """

        rows = int(hidden_states.shape[0])
        shadow = self._draftslice_shadow
        mapping = self._draftslice_idx_mapping
        steps = self._draftslice_draft_step
        if shadow is None or mapping is None or steps is None:
            return None
        try:
            states = mapping[:rows].reshape(-1).tolist()
            positions = steps[:rows].reshape(-1).tolist()
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"block mapping: {exc}")
            return None
        if len(states) != rows or len(positions) != rows:
            return None
        result: list[Any] = []
        planned = 0
        for state, position in zip(states, positions, strict=True):
            if int(position) == 0 and shadow.has(int(state)):
                result.append(shadow.valid_ids(int(state), step=0))
                planned += 1
            else:
                result.append(None)
        if not planned:
            return None
        self._draftslice_stats.block_step0_rows += planned
        return result

    def _valid_sets_for_rows(self, hidden_states: Any) -> list[Any] | None:
        if self._draftslice_block:
            return self._block_valid_sets(hidden_states)
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
            self._draftslice_row_workloads = [
                shadow.workload(int(state))
                if shadow.has(int(state))
                else "structured"
                for state in states
            ]
            return result

        if (
            self._draftslice_snapshot is None
            or not self._draftslice_req_by_state
        ):
            self._draftslice_row_workloads = None
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
        # The scheduler snapshot does not carry the structured request type;
        # label the rows "structured" so no cell can over-claim a workload.
        self._draftslice_row_workloads = ["structured"] * len(states)
        return [
            self._snapshot_valid_ids(int(state), max_ids=max_ids, position=position)
            for state in states
        ]

    def _ensure_screener(self) -> None:
        """Build the configured registry screener or load its artifact once."""

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
            if config.artifact:
                from ..screen import load_screener_bundle

                screener, bundle = load_screener_bundle(config.artifact)
                calibration = bundle.manifest.calibration or {}
                config = replace(
                    config,
                    name=screener.name,
                    calibration_id=calibration.get("calibration_id"),
                    delta=calibration.get("delta"),
                )
            else:
                from ..screen import get_screener

                screener = get_screener(config.name, **config.args)
        except Exception as exc:
            self._draftslice_stats.record_error(
                f"screener init {config.artifact or config.name!r}: {exc}"
            )
            return
        self._draftslice_screener = screener
        self._draftslice_screener_config = config
        if os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1":
            import sys

            print(
                f"[draftslice] screener={config.name!r} "
                f"artifact={config.artifact!r} universe={config.universe!r} "
                f"calibration={config.calibration_id!r}",
                file=sys.stderr,
                flush=True,
            )

    @staticmethod
    def _move_screener(screener: Any, device: Any) -> Any:
        updates: dict[str, Any] = {}
        for name in ("centroids", "w1", "w2", "vr", "a", "group_mass"):
            value = getattr(screener, name, None)
            if value is not None and hasattr(value, "to"):
                if getattr(value, "device", None) != device:
                    updates[name] = value.to(device)
        return replace(screener, **updates) if updates else screener

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
        self._draftslice_stats.screener_universe = config.universe
        self._draftslice_stats.screener_calibration_id = config.calibration_id
        self._draftslice_stats.screener_delta = config.delta
        device = getattr(hidden_states, "device", None)
        if device is not None:
            screener = self._move_screener(screener, device)
            self._draftslice_screener = screener
        t0 = time.perf_counter()
        try:
            candidates = screener.candidates(hidden_states)
            universe = config.universe
            if (
                self._draftslice_block
                and self._draftslice_shadow is not None
                and self._draftslice_shadow.entries
            ):
                # Structured DFlash: rows after block position 0 have no exact
                # state, so never let free-form candidates bypass the grammar.
                universe = "valid"
            result = _screen_valid_sets(
                valid_sets, candidates, universe=universe
            )
        except Exception as exc:
            self._draftslice_stats.screener_errors += 1
            self._draftslice_stats.record_error(f"screener {config.name!r}: {exc}")
            return valid_sets
        _TIMING["screen"] += time.perf_counter() - t0
        self._draftslice_stats.screener_calls += 1
        self._draftslice_stats.screener_rows += result.rows
        self._draftslice_stats.screener_dropped += result.dropped
        self._draftslice_stats.screener_empty_rows += result.empty_rows
        self._draftslice_stats.screener_freeform_rows += result.freeform_rows
        return result.sets

    def _lm_head_weight(self) -> Any | None:
        """Return the local, unquantized, CUDA LM-head weight or ``None``."""

        model = getattr(self, "model", None)
        lm_head = getattr(model, "lm_head", None) if model is not None else None
        if lm_head is None:
            self._draftslice_stats.note_fallback("no_lm_head")
            return None
        try:
            import torch
        except ImportError:  # pragma: no cover - vLLM requires torch
            return None
        weight = getattr(lm_head, "weight", None)
        if weight is None or weight.device.type != "cuda":
            self._draftslice_stats.note_fallback("no_cuda_weight")
            return None
        if weight.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ) and not is_quantized_weight(weight):
            self._draftslice_stats.note_fallback("unsupported_dtype")
            return None
        return weight

    def _slice_head(self) -> tuple[Any, HeadShard] | None:
        """Return the sliceable local ``(weight, shard)`` or fail closed.

        TP>1 is supported by slicing each rank's contiguous vocab row shard and
        scattering the result back to global token-id columns. Any condition we
        cannot prove safe returns ``None`` and records a fallback reason.
        """

        weight = self._lm_head_weight()
        if weight is None:
            return None
        lm_head = getattr(getattr(self, "model", None), "lm_head", None)
        if is_quantized_weight(weight):
            quant_head = describe_quantized_head(lm_head, weight)
            if quant_head is None:
                self._draftslice_stats.note_fallback("unsupported_quant_scheme")
                return None
            self._draftslice_quant_head = quant_head
            self._draftslice_stats.quantized_head = True
        else:
            self._draftslice_quant_head = None
        try:
            vocab_size = int(getattr(self, "vocab_size", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            vocab_size = 0
        shard = HeadShard.from_lm_head(
            lm_head,
            weight_rows=int(weight.shape[0]),
            vocab_size=vocab_size,
        )
        if shard is None:
            self._draftslice_stats.note_fallback("unsupported_head_layout")
            return None
        self._draftslice_shard = shard
        self._draftslice_stats.tp_size = shard.tp_size
        self._draftslice_stats.tp_rank = shard.tp_rank
        self._draftslice_stats.tp_shard_rows = shard.shard_rows
        key = (weight.data_ptr(), weight.dtype, tuple(weight.shape), shard)
        if self._draftslice_head_key == key and self._draftslice_head_cache is not None:
            return self._draftslice_head_cache
        self._draftslice_head_key = key
        self._draftslice_head_cache = (weight, shard)
        return weight, shard

    # -- production policy -------------------------------------------------
    def _policy_allows_work(self) -> bool:
        """Observe/off/disabled-only must not pay shadow bookkeeping."""

        adapter = self._draftslice_adapter_policy
        if adapter is None or not adapter.uses_table:
            return True
        return adapter.hold_reason is None

    def _policy_model_name(self) -> str:
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        raw = getattr(model_config, "model", None) or ""
        text = str(raw).rstrip("/")
        if not text:
            return ""
        # Cell keys use the model directory/name ("Qwen3.5-4B"), not the full
        # local path used by the engine.
        return text.rsplit("/", 1)[-1] if "/" in text else text

    def _policy_method_name(self) -> str:
        spec = getattr(getattr(self, "vllm_config", None), "speculative_config", None)
        method = str(getattr(spec, "method", "") or "")
        # vLLM resolves a registered speculative method to its base family
        # ("mtp"/"eagle3"/"dflash"). Cell keys name the DraftSlice profile, so
        # the DraftSlice adapter prefixes the family it runs.
        if method in ("mtp", "eagle3", "dflash"):
            return f"draftslice_{method}"
        return method

    def _policy_quant(self) -> str:
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        quant = getattr(model_config, "quantization", None)
        if quant:
            return str(quant).lower()
        dtype = getattr(model_config, "dtype", None)
        name = str(getattr(dtype, "name", dtype) or "").strip().lower()
        if name.startswith("torch."):
            name = name[len("torch.") :]
        return {
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "fp32",
            "float8_e4m3fn": "fp8",
            "float8_e5m2": "fp8",
            "int8": "int8",
        }.get(name, name or "auto")

    def _row_workloads(self, rows: int) -> list[str]:
        workloads = self._draftslice_row_workloads
        if workloads is not None and len(workloads) >= rows:
            return list(workloads[:rows])
        return ["structured"] * rows

    def _decide_policy(self, valid_sets: Any) -> PolicyDecision | None:
        adapter = self._draftslice_adapter_policy
        if adapter is None or adapter.policy is None:
            return None
        workloads = self._row_workloads(len(valid_sets))
        rows = tuple(
            RowFacts(
                workload=workload,
                valid_size=0 if token_set is None else len(token_set),
            )
            for workload, token_set in zip(workloads, valid_sets, strict=True)
        )
        model_name, method_name, quant, tp_size = getattr(
            self,
            "_draftslice_policy_identity",
            (
                self._policy_model_name(),
                self._policy_method_name(),
                self._policy_quant(),
                self._tp_world_size(),
            ),
        )
        if rows:
            bucket = batch_bucket(max(1, len(rows)))
            stats = self._draftslice_stats
            stats.batch_buckets[bucket] = stats.batch_buckets.get(bucket, 0) + 1
        facts = StepFacts(
            engine="vllm",
            model=model_name,
            method=method_name,
            batch=len(rows),
            k=int(getattr(self, "num_speculative_steps", 0) or 0),
            tp=tp_size,
            quant=quant,
            graph="full_cuda_graph"
            if getattr(self, "_draftslice_graph_replay", False)
            else "eager",
            rows=rows,
            capabilities={
                "grammar": any(row.valid_size > 0 for row in rows),
                "graph": not getattr(self, "_draftslice_graph_replay", False),
            },
        )
        decision = adapter.policy.decide(facts)
        if _debug_on():
            _tp_debug(
                "policy facts "
                f"model={facts.model!r} method={facts.method!r} "
                f"batch={facts.batch} k={facts.k} tp={facts.tp} "
                f"quant={facts.quant!r} graph={facts.graph!r} "
                f"rows={[(row.workload, row.valid_size) for row in rows][:4]} "
                f"-> {decision.to_dict()}"
            )
        self._record_policy(
            decision,
            rows=len(rows),
            candidates=sum(
                row.valid_size for row, hit in zip(rows, decision.row_mask, strict=True)
                if hit
            ),
        )
        return decision

    def _record_policy(
        self, decision: PolicyDecision, *, rows: int = 0, candidates: int = 0
    ) -> None:
        stats = self._draftslice_stats
        self._draftslice_policy_decision = decision
        stats.profile_mode = decision.profile_mode
        stats.cell_id = decision.cell_id
        stats.cell_hit = decision.cell_hit
        stats.action = decision.action
        if decision.executes_slice:
            stats.policy_slice_steps += 1
        else:
            stats.policy_full_steps += 1
        if (
            self._draftslice_traces is not None
            and self._draftslice_trace_sampler is not None
            and self._draftslice_trace_sampler()
        ):
            vocab = max(1, int(getattr(self, "vocab_size", 0) or 1))
            self._draftslice_traces.add(
                StepTrace(
                    identity=TraceIdentity(
                        engine="vllm",
                        model=self._policy_model_name() or "unknown",
                        tp=self._tp_world_size(),
                        quant=self._policy_quant(),
                        graph="full_cuda_graph"
                        if getattr(self, "_draftslice_graph_replay", False)
                        else "eager",
                        workload=decision.workload or "unknown",
                        config_hash=stats.profile_hash,
                    ),
                    decision=TraceDecision(
                        mode=decision.mode,
                        action=decision.action,
                        profile_mode=decision.profile_mode,
                        source="policy"
                        if self._draftslice_adapter_policy is not None
                        else "builtin",
                        k=int(getattr(self, "num_speculative_steps", 0) or 0),
                        cell_id=decision.cell_id,
                        cell_hit=decision.cell_hit,
                        reason=None
                        if decision.executes_slice
                        else decision.reason.value,
                        candidates=int(candidates),
                        rows=int(rows),
                        row_share=min(
                            1.0, float(candidates) / float(max(1, rows) * vocab)
                        ),
                    ),
                    timing=TraceTiming(),
                )
            )
            stats.trace_count = self._draftslice_traces.recorded

    def _tp_world_size(self) -> int:
        """Tensor-parallel size from the engine config (1 when unknown)."""

        config = getattr(self, "vllm_config", None)
        parallel = getattr(config, "parallel_config", None)
        try:
            return max(1, int(getattr(parallel, "tensor_parallel_size", 1) or 1))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 1

    def _tp_plan_callbacks(
        self, shard: HeadShard | None
    ) -> tuple[Any | None, Any | None]:
        """Return ``(plan_agreement, combine_logits)`` for a TP>1 head shard.

        The engine samples (and advances the worker grammar shadow) on every
        rank, so a rank-local sliced tensor lets the ranks diverge: different
        draft tokens fork the shadow differently, and the next step's full-head
        row sets stop matching. The engine's own full-head path all-gathers the
        logits for exactly this reason; the sliced path must do the same.

        ``combine_logits`` max-reduces the partial rows (each column is scored
        by one rank, all other contributions are ``-inf``, which is the max
        identity). ``plan_agreement`` unions
        the full-head row set with a tiny collective before the bigger one, so
        no rank can enter the engine's logits all-gather alone.
        """

        if shard is None or int(shard.tp_size) <= 1:
            return None, None
        cached = self._draftslice_tp_plan
        if cached is not None and cached[0] == int(shard.tp_size):
            return cached[1], cached[2]
        tp_size = int(shard.tp_size)
        try:
            import torch
            from vllm.distributed import tensor_model_parallel_all_gather
        except Exception as exc:  # pragma: no cover - engine import guard
            self._draftslice_stats.record_error(f"tp collectives: {exc}")
            # A rank that cannot join the collectives must not silently fall
            # back: the other ranks would deadlock in the engine all-gather.
            raise

        def plan_agreement(
            rows: int, full_rows: Any, device: Any
        ) -> list[int]:
            flags = torch.zeros(rows, dtype=torch.int32, device=device)
            local = list(full_rows)
            if local:
                index = torch.as_tensor(local, dtype=torch.int64, device=device)
                flags[index] = 1
            _tp_debug(f"agree enter rows={rows} local={local}")
            gathered = tensor_model_parallel_all_gather(flags, dim=0)
            agreed = gathered.view(tp_size, rows).amax(dim=0)
            result = agreed.nonzero(as_tuple=False).flatten().tolist()
            _tp_debug(f"agree exit rows={rows} agreed={result}")
            return result

        def combine_logits(logits: Any) -> Any:
            rows = int(logits.shape[0])
            _tp_debug(f"combine enter shape={tuple(logits.shape)}")
            gathered = tensor_model_parallel_all_gather(
                logits.contiguous(), dim=-1
            )
            result = gathered.view(rows, tp_size, -1).amax(dim=1)
            _tp_debug(f"combine exit shape={tuple(result.shape)}")
            return result

        self._draftslice_tp_plan = (tp_size, plan_agreement, combine_logits)
        return plan_agreement, combine_logits

    def _prepare_weight(
        self,
        weight: Any,
        valid_sets: Any,
        shard: HeadShard | None,
        local_vocab: int,
        vocab_size: int,
    ) -> Any:
        """Materialize the candidate weight rows (dequantizing if needed)."""

        quant_head = self._draftslice_quant_head
        if quant_head is None:
            return weight if shard is not None else weight[:local_vocab]

        import torch

        rows: set[int] = set()
        for token_set in valid_sets or ():
            if token_set is None:
                continue
            for token in token_set.ids:
                if token >= vocab_size:
                    continue
                row = token if shard is None else shard.local_row(token)
                if row is not None and 0 <= row < local_vocab:
                    rows.add(int(row))
        buffer = self._draftslice_dequant_weight
        if (
            buffer is None
            or buffer.shape[0] != local_vocab
            or buffer.shape[1] != quant_head.in_features
        ):
            buffer = torch.zeros(
                (local_vocab, quant_head.in_features),
                dtype=quant_head.out_dtype,
                device=weight.device,
            )
            self._draftslice_dequant_weight = buffer
            self._draftslice_dequant_filled = set()
        pending = sorted(rows - self._draftslice_dequant_filled)
        if pending:
            values = dequantize_rows(quant_head, pending)
            if values is None:
                self._draftslice_stats.note_fallback("unsupported_quant_scheme")
                return None
            with torch.no_grad():
                buffer[pending] = values
            self._draftslice_dequant_filled.update(pending)
            self._draftslice_stats.quantized_rows += len(pending)
        return buffer

    def _logits_postprocess(self) -> tuple[float, float | None, bool]:
        processor = getattr(getattr(self, "model", None), "logits_processor", None)
        if processor is None:
            return 1.0, None, True
        if getattr(processor, "logits_as_input", False):
            return 1.0, None, False
        scale = float(getattr(processor, "scale", 1.0))
        soft_cap = getattr(processor, "soft_cap", None)
        return scale, soft_cap, True

    def _compute_block_allpos(
        self,
        hidden_states: Any,
        spec_step_idx: int,
        base_compute: Any,
    ) -> Any | None:
        """DFlash: mask every block position with its own exact grammar set.

        This is a *mask-only* path: the full block head is computed once, then
        each position is masked in sequence. Greedy sampling makes the next
        token the argmax of the masked row, so that argmax can advance a forked
        matcher to build the next position's set before the engine samples.

        Returns ``None`` when the shape, the state mapping, or the sampling mode
        is unsupported; the caller then uses the ordinary (position-0) path.
        """

        shadow = self._draftslice_shadow
        mapping = self._draftslice_idx_mapping
        if shadow is None or mapping is None or not shadow.entries:
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None
        rows = int(hidden_states.shape[0])
        steps = int(getattr(self, "num_speculative_steps", 1) or 1)
        if steps <= 1 or rows <= 0 or rows % steps:
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None
        if rows // steps > _block_allpos_max_batch():
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None
        try:
            state_rows = [int(x) for x in mapping[:rows].reshape(-1).tolist()]
            step_tensor = self._draftslice_draft_step
            if step_tensor is not None:
                step_rows = [int(x) for x in step_tensor[:rows].reshape(-1).tolist()]
            else:
                step_rows = [index % steps for index in range(rows)]
        except Exception as exc:  # pragma: no cover - defensive
            self._draftslice_stats.record_error(f"block allpos mapping: {exc}")
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None
        temperature = getattr(self, "temperature", None)
        if temperature is not None:
            try:
                involved = sorted({state for state in state_rows if state >= 0})
                if involved and float(temperature[involved].max()) != 0.0:
                    # Sampling with temperature would not reproduce the argmax
                    # used to advance the grammar, so keep the exact fallback.
                    self._draftslice_stats.block_allpos_fallbacks += 1
                    return None
            except Exception:  # pragma: no cover - defensive
                self._draftslice_stats.block_allpos_fallbacks += 1
                return None

        logits = base_compute(hidden_states, spec_step_idx)
        if logits.ndim != 2 or int(logits.shape[0]) != rows:
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None

        groups = rows // steps
        # Position-major, batched state advance: one grammar mask per
        # (request, position), but only one argmax sync per position. The
        # per-row version serialized B*k argmaxes and stalled the stream.
        work: dict[int, Any] = {}
        for req in range(groups):
            base = req * steps
            state = state_rows[base]
            if state < 0 or not shadow.has(state):
                continue
            if step_rows[base : base + steps] != list(range(steps)):
                continue
            handle = shadow.begin_block(state)
            if handle is not None:
                work[req] = handle
        if not work:
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None

        depth = _block_allpos_depth()
        limit = steps if depth is None else min(steps, depth)
        view = logits.view(groups, steps, -1)
        masked_rows = 0
        fast = _block_allpos_fast()
        vocab = int(logits.shape[1])
        try:
            import torch

            for position in range(limit):
                if fast:
                    from .vllm_fastmask import apply_packed_mask

                    packed = shadow.batch_block_masks(
                        work, vocab_size=vocab, device=logits.device
                    )
                    if packed is None:
                        break
                    reqs = list(work.keys())
                    rows = upload_numpy(
                        np.asarray(reqs, dtype=np.int32), logits.device
                    )
                    apply_packed_mask(
                        logits,
                        base=position * vocab,
                        row_stride=steps * vocab,
                        packed=packed,
                        rows=rows,
                        vocab_size=vocab,
                    )
                    masked_rows += len(reqs)
                    target = view[:, position, :]
                    tokens = target.argmax(dim=-1).tolist()
                    for req in reqs:
                        if not shadow.block_advance(work[req], int(tokens[req])):
                            work.pop(req, None)
                    continue

                active: list[int] = []
                masks: list[Any] = []
                for req, handle in work.items():
                    mask = shadow.block_mask(
                        handle,
                        vocab_size=vocab,
                        device=logits.device,
                    )
                    if mask is None:
                        continue
                    active.append(req)
                    masks.append(mask)
                if not active:
                    break
                target = view[:, position, :]
                if len(active) == groups:
                    target.masked_fill_(~torch.stack(masks), float("-inf"))
                else:
                    for req, mask in zip(active, masks, strict=True):
                        target[req].masked_fill_(~mask, float("-inf"))
                tokens = target[active].argmax(dim=-1).tolist()
                masked_rows += len(active)
                for req, token in zip(active, tokens, strict=True):
                    if not shadow.block_advance(work[req], int(token)):
                        work.pop(req, None)
        except Exception as exc:  # pragma: no cover - defensive
            if os.environ.get("DRAFTSLICE_VLLM_DEBUG") == "1":
                import traceback

                traceback.print_exc()
            self._draftslice_stats.record_error(f"block allpos: {exc}")
            self._draftslice_stats.block_allpos_fallbacks += 1
            return None

        self._draftslice_stats.block_allpos_rows += masked_rows
        self._draftslice_stats.block_allpos_requests += groups
        self._draftslice_stats.calls += 1
        self._draftslice_stats.sliced_calls += 1
        return logits

    def get_draft_top_tokens(self, hidden_states: Any, spec_step_idx: int = 0) -> Any:
        """Greedy ids from the sliced head when local-argmax reduction is on.

        vLLM's ``sample_draft`` bypasses ``compute_draft_logits`` in this mode
        and calls this hook directly. We reuse the sliced path to produce
        ``[batch, vocab]`` logits (and its contract/fallback stats), then apply
        the same ``(value, id)`` TP reduction vLLM uses in
        ``LogitsProcessor.get_top_tokens``.
        """

        base = super().get_draft_top_tokens
        if not (
            self._draftslice_enabled
            and self._draftslice_slicing
            and self._draftslice_local_argmax
        ):
            try:
                return base(hidden_states, spec_step_idx)
            except TypeError:  # pragma: no cover - single-arg model hooks
                return base(hidden_states)

        import torch

        logits = self.compute_draft_logits(hidden_states, spec_step_idx, _combine=False)
        values, ids = logits.max(dim=-1)
        shard = self._draftslice_shard
        if shard is not None and shard.tp_size > 1:
            try:
                from vllm.distributed import tensor_model_parallel_all_gather

                pairs = torch.stack((values.float(), ids.float()), dim=-1)
                gathered = tensor_model_parallel_all_gather(pairs, dim=-1)
                gathered = gathered.view(logits.shape[0], shard.tp_size, 2)
                best = gathered[..., 0].argmax(dim=-1)
                rows = torch.arange(logits.shape[0], device=logits.device)
                ids = gathered[rows, best, 1]
            except Exception as exc:  # pragma: no cover - defensive
                self._draftslice_stats.record_error(f"local argmax reduce: {exc}")
        ids = ids.to(torch.int64)
        draft_to_target = getattr(
            getattr(self, "model", None), "draft_id_to_target_id", None
        )
        if draft_to_target is not None:
            ids = ids + draft_to_target[ids]
        return ids

    def compute_draft_logits(
        self, hidden_states: Any, spec_step_idx: int, *, _combine: bool = True
    ) -> Any:
        base_compute = super().compute_draft_logits
        if _debug_on():
            _tp_debug(
                f"compute enter step={getattr(self, '_draftslice_step', 0)} "
                f"rows={int(hidden_states.shape[0])} combine={_combine} "
                f"shadow={getattr(self, '_draftslice_shadow', None) is not None} "
                f"graph={getattr(self, '_draftslice_graph_replay', False)}"
            )

        def fallback(reason: str | None, valid_sets: Any = None) -> Any:
            return self._fallback(
                reason,
                valid_sets=valid_sets,
                base_compute=base_compute,
                hidden_states=hidden_states,
                spec_step_idx=spec_step_idx,
            )

        if self._draftslice_graph_replay and self._draftslice_slicing:
            logits = base_compute(hidden_states, spec_step_idx)
            vocab = min(self._draftslice_graph_vocab or int(logits.shape[1]), int(logits.shape[1]))
            if self._draftslice_spec_step == 0 and self._draftslice_graph_packed is not None:
                rows = min(hidden_states.shape[0], self._draftslice_graph_packed.shape[0])
                apply_packed_mask(
                    logits,
                    base=0,
                    row_stride=int(logits.shape[1]),
                    packed=self._draftslice_graph_packed[:rows],
                    rows=self._draftslice_graph_rows[:rows],
                    vocab_size=vocab,
                )
                self._draftslice_stats.calls += 1
                self._draftslice_stats.graph_mask_calls += 1
                self._emit_contract(
                    mode="mask", vocab_size=vocab, execution_kind="full"
                )
            else:
                self._draftslice_stats.graph_replay_calls += 1
                self._draftslice_stats.record(sliced=False)
                self._emit_contract(
                    mode="full", reason="graph_later_step", vocab_size=vocab
                )
            return logits
        if not self._draftslice_enabled or not self._draftslice_slicing:
            if self._draftslice_graph_replay:
                self._draftslice_stats.graph_replay_calls += 1
            else:
                self._draftslice_stats.slice_disabled_calls += int(self._draftslice_enabled)
            reason = "full_cuda_graph" if self._draftslice_graph_replay else "slicing_disabled"
            return fallback(reason)
        if self._draftslice_route_full:
            self._draftslice_stats.routed_full_calls += 1
            return fallback("router_freeform")
        if self._draftslice_dummy:
            return fallback("dummy_run")
        if self._draftslice_policy_hold is not None:
            # observe/off: stock path, no valid-set or shadow work at all.
            return fallback(self._draftslice_policy_hold)
        if self._draftslice_block and _block_allpos_enabled():
            allpos = self._compute_block_allpos(hidden_states, spec_step_idx, base_compute)
            if allpos is not None:
                # All-position masking computes the full block head once, then
                # masks each position, so its execution kind is "full".
                self._emit_contract(mode="mask_allpos", execution_kind="full")
                return allpos
        if self._draftslice_snapshot is None:
            self._draftslice_snapshot = current_grammar_snapshot()
        if self._draftslice_shadow is None and self._draftslice_spec_step != 0:
            self._draftslice_stats.nonzero_step_calls += 1
            return fallback("later_step")

        self._ensure_screener()
        screener_config = self._draftslice_screener_config
        screener_active = (
            self._draftslice_screener is not None and screener_config is not None
        )
        no_state = (
            (self._draftslice_snapshot is None and self._draftslice_shadow is None)
            or (
                self._draftslice_idx_mapping is None
                and self._draftslice_idx_mapping_np is None
            )
        )
        if not screener_active:
            if no_state:
                self._draftslice_stats.no_snapshot_calls += 1
                return fallback("no_grammar_state")
            valid_sets = self._valid_sets_for_rows(hidden_states)
            if valid_sets is None or not any(
                token_set is not None for token_set in valid_sets
            ):
                self._draftslice_stats.no_valid_sets_calls += 1
                return fallback("no_valid_sets")
        else:
            if screener_config.universe == "valid":
                if no_state:
                    self._draftslice_stats.no_snapshot_calls += 1
                    return fallback("no_grammar_state")
                valid_sets = self._valid_sets_for_rows(hidden_states)
                if valid_sets is None or not any(
                    token_set is not None for token_set in valid_sets
                ):
                    self._draftslice_stats.no_valid_sets_calls += 1
                    return fallback("no_valid_sets")
            else:
                valid_sets = self._valid_sets_for_rows(hidden_states)
                if valid_sets is None:
                    valid_sets = [None] * int(hidden_states.shape[0])
            valid_sets = self._apply_screener(hidden_states, valid_sets)
            if not any(token_set is not None for token_set in valid_sets):
                return fallback("screener_empty", valid_sets)

        policy_decision = self._decide_policy(valid_sets)
        contract_mode: str | None = None
        if policy_decision is not None:
            if not policy_decision.executes_slice:
                return fallback(policy_decision.reason.value, valid_sets)
            if policy_decision.action == "mask_allpos" and not self._draftslice_block:
                return fallback("capability_unsupported", valid_sets)
            # Only rows with an enabled exact cell keep their valid set; the
            # rest use the full head inside the same batch.
            valid_sets = filter_valid_sets(valid_sets, policy_decision.row_mask)
            contract_mode = policy_decision.mode

        if _full_mask_enabled() or (
            policy_decision is not None and policy_decision.action == "mask"
        ):
            vocab = min(
                int(getattr(self, "vocab_size", int(hidden_states.shape[1]))),
                int(hidden_states.shape[1]),
            )
            logits = mask_logits_to_valid(
                base_compute(hidden_states, spec_step_idx), valid_sets, vocab_size=vocab
            )
            self._draftslice_stats.calls += 1
            self._draftslice_stats.full_mask_calls += 1
            self._emit_contract(
                mode="mask",
                valid_sets=valid_sets,
                vocab_size=vocab,
                execution_kind="full",
            )
            return logits

        head_view = self._slice_head()
        if head_view is None:
            # _slice_head already recorded the specific reason.
            return fallback(None, valid_sets)
        weight, shard = head_view

        scale, soft_cap, supported = self._logits_postprocess()
        if not supported:
            return fallback("unsupported_logits_postprocess", valid_sets)

        vocab_size = min(
            int(getattr(self, "vocab_size", shard.org_vocab_size) or shard.org_vocab_size),
            shard.org_vocab_size,
        )
        if vocab_size <= 0:
            return fallback("invalid_vocab_size", valid_sets)
        deterministic = _deterministic_injection_enabled()
        if deterministic:
            self._draftslice_stats.deterministic_rows += sum(
                1
                for token_set in valid_sets
                if token_set is not None and len(token_set) == 1
            )
        t0 = time.perf_counter()
        plan_agreement, combine_logits = self._tp_plan_callbacks(shard)
        if not _combine:
            # The local-argmax hook reduces ``(value, id)`` pairs across ranks
            # itself, so it only needs this rank's columns.
            combine_logits = None
        try:
            slicing_weight = self._prepare_weight(
                weight,
                valid_sets,
                shard,
                int(shard.shard_rows if shard is not None else vocab_size),
                vocab_size,
            )
            if slicing_weight is None:
                # Full head + exact grammar mask keeps A subset of V even when
                # this rank cannot dequantize the candidate rows.
                logits = mask_logits_to_valid(
                    base_compute(hidden_states, spec_step_idx),
                    valid_sets,
                    vocab_size=vocab_size,
                )
                self._draftslice_stats.record(sliced=False)
                self._emit_contract(
                    mode="mask",
                    valid_sets=valid_sets,
                    vocab_size=vocab_size,
                    execution_kind="full",
                    reason="unsupported_quant_scheme",
                )
                return logits
            logits, stats = build_draft_logits(
                hidden_states,
                slicing_weight,
                valid_sets,
                vocab_size=vocab_size,
                full_logits=lambda rows: base_compute(rows, spec_step_idx),
                max_slice_rows=_max_slice_rows(vocab_size),
                scale=scale,
                soft_cap=soft_cap,
                deterministic=deterministic,
                shard=shard,
                plan_agreement=plan_agreement,
                combine_logits=combine_logits,
            )
        except Exception as exc:
            # A rank-local fallback would enter (or skip) the engine's TP logits
            # all-gather on this rank only and deadlock the others. Fail the
            # step loudly instead; the engine restarts the request.
            self._draftslice_stats.record_error(f"{type(exc).__name__}: {exc}")
            if shard is not None and int(shard.tp_size) > 1:
                raise
            return fallback("build_error", valid_sets)

        _TIMING["build"] += time.perf_counter() - t0
        self._draftslice_stats.record(sliced=True, stats=stats)
        self._draftslice_stats.row_share = compute_row_share(
            stats.candidate_total, stats.rows, vocab_size
        )
        self._emit_contract(
            mode=contract_mode
            or ("mask" if (self._draftslice_block or screener_active) else "strict"),
            valid_sets=valid_sets,
            logits_stats=stats,
            vocab_size=vocab_size,
        )
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
        # DFlash uses one forward pass for the whole block. Only block position
        # 0 has an exact grammar state before sampling; later positions use the
        # full head so no valid token is dropped.
        _draftslice_allow_slicing = True
        _draftslice_block = True

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
