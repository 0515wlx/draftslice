"""SGLang integration: grammar-aware draft-head slicing.

SGLang's EAGLE draft loop knows the per-request grammar state before the first
draft token of each proposal, and it already applies a grammar mask to the
target verify logits. DraftSlice adds the missing piece: on draft block
position 0 it slices the draft LM head to the grammar-valid rows instead of
computing the full vocabulary. Later positions keep the full head because their
grammar state depends on draft tokens that are not sampled yet.

The hook is opt-in (``DRAFTSLICE_SGLANG=1``) and installed as an SGLang general
plugin. It patches two narrow points:

- ``EagleDraftWorker.draft`` publishes the batch grammars and a step counter in a
  context variable;
- ``LogitsProcessor._compute_lm_head`` consumes that state on step 0 and calls
  the shared DraftSlice row kernel. Unsupported heads (TP > 1, quantized, no
  CUDA weight, no grammar, oversized valid set, or any error) fall back to the
  original full head.

The target verifier remains exact. No later-position valid token is dropped
because those positions never use the position-0 candidate set.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from ..constraints.xgrammar import bitmask_to_token_set
from ..plan.types import (
    DraftContract,
    Execution,
    Guarantee,
    GuaranteeKind,
    ServingPoint,
    TokenSet,
)
from ..runtime.policy import PolicyDecision, RowFacts, StepFacts
from ..runtime.trace import (
    StepTrace,
    TraceDecision,
    TraceIdentity,
    TraceRegistry,
    TraceTiming,
)
from ..screen.base import Screener
from .base import (
    AdapterPolicy,
    AdapterPolicyError,
    TraceSampler,
    filter_valid_sets,
    load_adapter_policy,
    make_registry,
    trace_sample_from_env,
)
from .head_shard import HeadShard
from .vllm_logits import build_draft_logits

_REGISTERED = False
_PATCHED = False
_DRAFT_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "draftslice_sglang_draft_state", default=None
)
_ORIGINAL_DRAFT: Any | None = None
_ORIGINAL_LM_HEAD: Any | None = None
_ORIGINAL_DRAFTS: dict[type, Any] = {}
_SCREENER: Screener | None = None
_SCREENER_READY = False
_STATS: dict[str, Any] = {
    "step0_calls": 0,
    "sliced_calls": 0,
    "candidate_rows": 0,
    "fallback_calls": 0,
    "fallback_reasons": {},
    "last_contract": None,
    "tp_size": 1,
    "tp_rank": 0,
    "profile_mode": "off",
    "profile_table": "",
    "profile_hash": "",
    "cell_id": None,
    "cell_hit": False,
    "action": "builtin",
    "row_share": 0.0,
    "measured_head_share": None,
    "trace_count": 0,
}


def _note_fallback(reason: str) -> None:
    reasons = _STATS["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _contract_reason(lm_head: Any) -> str:
    """Why this head cannot be sliced (used in the emitted full contract)."""

    weight = getattr(lm_head, "weight", None)
    if weight is None or getattr(weight, "device", None) is None:
        return "no_weight"
    if weight.device.type != "cuda":
        return "no_cuda_weight"
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None and "unquant" not in type(quant_method).__name__.lower():
        return "quantized_head"
    num_embeddings = getattr(lm_head, "num_embeddings", None)
    if num_embeddings is not None and int(num_embeddings) != int(weight.shape[0]):
        return "padded_or_sharded_head"
    return "unsupported_head"


def _emit_contract(
    *,
    mode: str,
    reason: str | None = None,
    valid_sets: Any = None,
    candidates: int = 0,
    rows: int = 0,
    vocab_size: int = 0,
) -> None:
    """Emit the stable contract for a step, following the production data model."""

    state = _DRAFT_STATE.get()
    k = int(state.get("topk", 0)) if state else 0
    ids: set[int] = set()
    for token_set in valid_sets or ():
        if token_set is not None:
            ids.update(token_set.ids)
    valid = TokenSet(ids) if ids else None
    if mode == "full":
        execution = Execution(kind="full")
        # SGLang's fallback is the engine's own full-head path, not a
        # DraftSlice-managed full-head mask.
        guarantee = Guarantee(
            kind=GuaranteeKind.FULL, universe="vocab", fallback_rule="built_in_method"
        )
        candidate_set = None
        row_share = None
    else:
        execution = Execution(kind="slice_gemv", rows=valid)
        guarantee = Guarantee.exact_valid_set()
        candidate_set = valid
        row_share = candidates / (rows * vocab_size) if rows and vocab_size else None
    contract = DraftContract(
        grammar_state_id=None,
        valid_set=valid,
        candidate_set=candidate_set,
        mode=mode,
        execution=execution,
        guarantee=guarantee,
        serving=ServingPoint(
            k=k,
            source="draftslice_sglang",
            target_metric="throughput",
            row_share=row_share,
        ),
        fallback_reason=reason,
    )
    _STATS["last_contract"] = contract.to_dict()
    if reason is not None:
        _note_fallback(reason)


class SglangIntegrationError(RuntimeError):
    """Raised when the SGLang integration cannot be installed."""


@dataclass(frozen=True, slots=True)
class SglangConfig:
    """Environment-level switches for the SGLang hook."""

    enabled: bool = False
    screener: str | None = None
    screener_args: dict[str, Any] | None = None
    screener_artifact: str | None = None
    max_rows: int = 32_768
    # Profile policy facts. SGLang does not expose the structured request type
    # at the head hook, so the operator labels the deployment workload/model.
    model: str = ""
    method: str = "draftslice_eagle3"
    workload: str = "structured"
    quant: str = "auto"
    graph: str = "eager"

    def __post_init__(self) -> None:
        if self.max_rows <= 0:
            raise SglangIntegrationError("max_rows must be positive")


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "False", "")


def _json_args(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SglangIntegrationError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SglangIntegrationError(f"{name} must be a JSON object")
    return value


def config_from_env() -> SglangConfig:
    raw_rows = os.environ.get("DRAFTSLICE_SGLANG_MAX_ROWS", "32768").strip()
    try:
        max_rows = int(raw_rows)
    except ValueError as exc:
        raise SglangIntegrationError(
            "DRAFTSLICE_SGLANG_MAX_ROWS must be an integer"
        ) from exc
    return SglangConfig(
        enabled=_flag("DRAFTSLICE_SGLANG"),
        screener=os.environ.get("DRAFTSLICE_SGLANG_SCREENER", "").strip() or None,
        screener_args=_json_args("DRAFTSLICE_SGLANG_SCREENER_ARGS"),
        screener_artifact=os.environ.get(
            "DRAFTSLICE_SGLANG_SCREENER_ARTIFACT", ""
        ).strip()
        or None,
        max_rows=max_rows,
        model=os.environ.get("DRAFTSLICE_MODEL", "").strip(),
        method=os.environ.get("DRAFTSLICE_METHOD", "").strip()
        or "draftslice_eagle3",
        workload=os.environ.get("DRAFTSLICE_WORKLOAD", "").strip() or "structured",
        quant=os.environ.get("DRAFTSLICE_QUANT", "").strip() or "auto",
        graph=os.environ.get("DRAFTSLICE_GRAPH", "").strip() or "eager",
    )


_POLICY: AdapterPolicy | None = None
_POLICY_READY = False
_TRACES: TraceRegistry | None = None
_TRACE_SAMPLER: TraceSampler | None = None


def _trace_sink() -> tuple[TraceRegistry, TraceSampler]:
    global _TRACES, _TRACE_SAMPLER
    if _TRACES is None or _TRACE_SAMPLER is None:
        try:
            _TRACES = make_registry(256)
            _TRACE_SAMPLER = TraceSampler(trace_sample_from_env())
        except AdapterPolicyError as exc:
            raise SglangIntegrationError(str(exc)) from exc
    return _TRACES, _TRACE_SAMPLER


def adapter_policy(config: SglangConfig) -> AdapterPolicy | None:
    """Load the production policy once; a bad table/hash raises at install."""

    global _POLICY, _POLICY_READY
    if _POLICY_READY:
        return _POLICY
    _POLICY_READY = True
    try:
        _POLICY = load_adapter_policy(
            "sglang", model=config.model or None, method=config.method
        )
    except AdapterPolicyError as exc:
        raise SglangIntegrationError(str(exc)) from exc
    if _POLICY is not None and _POLICY.uses_table:
        _STATS["profile_mode"] = _POLICY.profile_mode
        _STATS["profile_table"] = _POLICY.table_path
        _STATS["profile_hash"] = _POLICY.table_hash
    return _POLICY


def _policy_hold(config: SglangConfig) -> Any | None:
    """Reason to hold on the stock head for the whole run (observe/off)."""

    policy = adapter_policy(config)
    if policy is None:
        return None
    reason = policy.hold_reason
    if reason is not None:
        _STATS["profile_mode"] = policy.profile_mode
        _STATS["action"] = "builtin"
        _STATS["cell_hit"] = False
        _STATS["cell_id"] = None
    return reason


def _decide_policy(
    config: SglangConfig, state: dict[str, Any], candidates: list[Any], rows: int
) -> PolicyDecision | None:
    policy = adapter_policy(config)
    if policy is None or policy.policy is None:
        return None
    workload = config.workload or "structured"
    facts = StepFacts(
        engine="sglang",
        model=config.model or "<unset>",
        method=config.method,
        batch=int(state.get("batch", 1) or 1),
        k=int(state.get("topk", 1) or 1),
        tp=int(_STATS.get("tp_size", 1) or 1),
        quant=config.quant,
        graph=config.graph,
        rows=tuple(
            RowFacts(
                workload=workload,
                valid_size=0 if token_set is None else len(token_set),
            )
            for token_set in candidates[:rows]
        ),
        capabilities={"grammar": True, "graph": True},
    )
    decision = policy.policy.decide(facts)
    _STATS["profile_mode"] = decision.profile_mode
    _STATS["cell_id"] = decision.cell_id
    _STATS["cell_hit"] = decision.cell_hit
    _STATS["action"] = decision.action
    traces, sampler = _trace_sink()
    if sampler():
        row_facts = facts.rows
        candidates = sum(
            row.valid_size
            for row, hit in zip(row_facts, decision.row_mask, strict=True)
            if hit
        )
        traces.add(
            StepTrace(
                identity=TraceIdentity(
                    engine="sglang",
                    model=config.model or "unknown",
                    tp=int(_STATS.get("tp_size", 1) or 1),
                    quant=config.quant,
                    graph=config.graph,
                    workload=config.workload or "structured",
                    config_hash=str(_STATS.get("profile_hash", "")),
                ),
                decision=TraceDecision(
                    mode=decision.mode,
                    action=decision.action,
                    profile_mode=decision.profile_mode,
                    source="policy",
                    k=int(state.get("topk", 1) or 1),
                    cell_id=decision.cell_id,
                    cell_hit=decision.cell_hit,
                    reason=None if decision.executes_slice else decision.reason.value,
                    candidates=candidates,
                    rows=len(row_facts),
                    row_share=min(
                        1.0,
                        float(candidates) / float(max(1, len(row_facts)) * max(1, rows)),
                    ),
                ),
                timing=TraceTiming(),
            )
        )
        _STATS["trace_count"] = traces.recorded
    return decision


def stats() -> dict[str, Any]:
    """Copy of the integration counters plus the recent step traces."""

    out = dict(_STATS)
    out["fallback_reasons"] = dict(_STATS["fallback_reasons"])
    out["recent_traces"] = (
        [] if _TRACES is None else [t.to_dict() for t in _TRACES.recent(4)]
    )
    return out


def _ensure_screener(config: SglangConfig) -> Screener | None:
    """Load the optional screener once per process."""

    global _SCREENER, _SCREENER_READY
    if _SCREENER_READY:
        return _SCREENER
    _SCREENER_READY = True
    try:
        if config.screener_artifact:
            from ..screen import load_screener_bundle

            _SCREENER, _ = load_screener_bundle(config.screener_artifact)
        elif config.screener:
            from ..screen import get_screener

            _SCREENER = get_screener(
                config.screener, **(config.screener_args or {})
            )
    except Exception:
        _SCREENER = None
    return _SCREENER


def _move_screener(screener: Screener, device: Any) -> Screener:
    updates: dict[str, Any] = {}
    for name in ("centroids", "w1", "w2", "vr", "a", "group_mass"):
        value = getattr(screener, name, None)
        if value is not None and hasattr(value, "to"):
            if getattr(value, "device", None) != device:
                updates[name] = value.to(device)
    return replace(screener, **updates) if updates else screener


def _grammar_candidates(
    state: dict[str, Any],
    rows: int,
    vocab_size: int,
    *,
    max_rows: int,
    device: Any,
    screener: Screener | None,
    step: int = 0,
) -> list[TokenSet | None] | None:
    """Return one candidate set per draft row for block position 0."""

    if step != 0:
        return None
    grammars = state.get("grammars")
    if not grammars:
        return None
    batch = len(grammars)
    if batch <= 0 or rows % batch != 0:
        return None
    repeat = rows // batch
    try:
        mask = grammars[0].allocate_vocab_mask(vocab_size, batch, "cpu")
        for index, grammar in enumerate(grammars):
            if grammar is None or grammar.is_terminated():
                continue
            grammar.fill_vocab_mask(mask, index)
    except Exception:
        return None

    per_request: list[TokenSet | None] = []
    for index, grammar in enumerate(grammars):
        if grammar is None or grammar.is_terminated():
            per_request.append(None)
            continue
        try:
            tokens = bitmask_to_token_set(mask[index], vocab_size)
        except Exception:
            per_request.append(None)
            continue
        if tokens.is_empty or len(tokens) > max_rows:
            per_request.append(None)
            continue
        per_request.append(tokens)

    if not any(tokens is not None for tokens in per_request):
        return None

    if screener is not None:
        screener = _move_screener(screener, device)
        try:
            proposed = screener.candidates(state["hidden_states"])
        except Exception:
            return None
        if len(proposed) != batch:
            return None
        per_request = [
            None if valid is None else valid.intersection(candidate)
            for candidate, valid in zip(proposed, per_request, strict=True)
        ]
        per_request = [
            None if tokens is None or tokens.is_empty else tokens
            for tokens in per_request
        ]

    return [
        per_request[index // repeat] if repeat else None for index in range(rows)
    ]


def _sliceable_weight(lm_head: Any) -> Any | None:
    weight = getattr(lm_head, "weight", None)
    if weight is None or getattr(weight, "device", None) is None:
        return None
    if weight.device.type != "cuda":
        return None
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None and "unquant" not in type(quant_method).__name__.lower():
        return None
    if _head_shard(lm_head, weight) is None:
        return None
    return weight


def _head_shard(lm_head: Any, weight: Any) -> HeadShard | None:
    vocab_size = int(getattr(lm_head, "num_embeddings", 0) or weight.shape[0])
    return HeadShard.from_lm_head(
        lm_head, weight_rows=int(weight.shape[0]), vocab_size=vocab_size
    )


def _patched_compute_lm_head(
    self: Any,
    hidden_states: Any,
    lm_head: Any,
    embedding_bias: Any = None,
    *,
    out: Any = None,
) -> Any:
    state = _DRAFT_STATE.get()
    if state is None or _ORIGINAL_LM_HEAD is None:
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )

    step = int(state.get("step", 0))
    if step != 0:
        state["step"] = step + 1
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )

    config: SglangConfig = state["config"]
    state["step"] = step + 1
    _STATS["step0_calls"] += 1
    _STATS["tp_size"] = int(getattr(lm_head, "tp_size", 1) or 1)
    _STATS["tp_rank"] = int(getattr(lm_head, "tp_rank", 0) or 0)
    hold = _policy_hold(config)
    if hold is not None:
        # observe/off/disabled-only: stock head, no candidate work.
        _STATS["fallback_calls"] += 1
        _note_fallback(hold.value)
        _emit_contract(mode="full", reason=hold.value)
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )
    weight = _sliceable_weight(lm_head)
    if weight is None:
        _STATS["fallback_calls"] += 1
        _emit_contract(mode="full", reason=_contract_reason(lm_head))
        if os.environ.get("DRAFTSLICE_SGLANG_DEBUG") == "1":
            import sys

            raw_weight = getattr(lm_head, "weight", None)
            print(
                "[draftslice:sglang] step0 fallback: unsupported LM head "
                f"type={type(lm_head).__name__} "
                f"device={getattr(raw_weight, 'device', None)} "
                f"shape={getattr(raw_weight, 'shape', None)} "
                f"tp_size={getattr(lm_head, 'tp_size', None)} "
                f"num_embeddings={getattr(lm_head, 'num_embeddings', None)} "
                f"quant={getattr(lm_head, 'quant_method', None)}",
                file=sys.stderr,
                flush=True,
            )
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )
    rows = int(hidden_states.shape[0])
    shard = _head_shard(lm_head, weight)
    # The grammar mask is over global token ids; the kernel reads local rows.
    global_vocab = int(
        getattr(self, "vocab_size", getattr(lm_head, "num_embeddings", weight.shape[0]))
        or weight.shape[0]
    )
    kernel_vocab = int(weight.shape[0])
    state["hidden_states"] = hidden_states
    candidates = _grammar_candidates(
        state,
        rows,
        global_vocab,
        max_rows=config.max_rows,
        device=hidden_states.device,
        screener=_ensure_screener(config),
        step=step,
    )
    if candidates is None:
        _STATS["fallback_calls"] += 1
        _emit_contract(mode="full", reason="no_candidates")
        if os.environ.get("DRAFTSLICE_SGLANG_DEBUG") == "1":
            import sys

            print(
                "[draftslice:sglang] step0 fallback: no candidates",
                file=sys.stderr,
                flush=True,
            )
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )

    policy_decision = _decide_policy(config, state, candidates, rows)
    if policy_decision is not None:
        if not policy_decision.executes_slice:
            _STATS["fallback_calls"] += 1
            reason = policy_decision.reason.value
            _note_fallback(reason)
            _emit_contract(
                mode="full",
                reason=reason,
                valid_sets=candidates,
                rows=rows,
                vocab_size=kernel_vocab,
            )
            return _ORIGINAL_LM_HEAD(
                self, hidden_states, lm_head, embedding_bias, out=out
            )
        if policy_decision.action not in ("slice", "mask"):
            # SGLang currently slices block position 0; a mask_allpos cell is a
            # DFlash/vLLM operating point.
            _STATS["fallback_calls"] += 1
            _note_fallback("capability_unsupported")
            _emit_contract(mode="full", reason="capability_unsupported")
            return _ORIGINAL_LM_HEAD(
                self, hidden_states, lm_head, embedding_bias, out=out
            )
        candidates = filter_valid_sets(candidates, policy_decision.row_mask)
        if not any(token_set is not None for token_set in candidates):
            _STATS["fallback_calls"] += 1
            _note_fallback("no_candidates")
            _emit_contract(mode="full", reason="no_candidates")
            return _ORIGINAL_LM_HEAD(
                self, hidden_states, lm_head, embedding_bias, out=out
            )

    # _compute_lm_head returns rank-local logits; SGLang gathers them later.
    # A TP rank therefore slices its own shard with local row ids and returns
    # local logits, so no TP collective is needed inside this hook.
    if shard is not None and shard.tp_size > 1:
        local_candidates: list[Any] = []
        for token_set in candidates:
            if token_set is None:
                local_candidates.append(None)
                continue
            local_ids = tuple(
                row
                for row in (shard.local_row(token) for token in token_set.ids)
                if row is not None and 0 <= row < kernel_vocab
            )
            local_candidates.append(TokenSet(local_ids) if local_ids else None)
        candidates = local_candidates

    def full_logits(subset: Any) -> Any:
        return _ORIGINAL_LM_HEAD(self, subset, lm_head, embedding_bias, out=None)

    try:
        logits, stats = build_draft_logits(
            hidden_states,
            weight[:kernel_vocab],
            candidates,
            vocab_size=kernel_vocab,
            full_logits=full_logits,
            max_slice_rows=config.max_rows,
        )
    except Exception:
        _STATS["fallback_calls"] += 1
        _emit_contract(mode="full", reason="build_error")
        return _ORIGINAL_LM_HEAD(
            self, hidden_states, lm_head, embedding_bias, out=out
        )
    state["sliced_calls"] = int(state.get("sliced_calls", 0)) + 1
    _STATS["sliced_calls"] += 1
    candidate_rows = sum(0 if tokens is None else len(tokens) for tokens in candidates)
    _STATS["candidate_rows"] += candidate_rows
    _emit_contract(
        mode="mask",
        valid_sets=candidates,
        candidates=getattr(stats, "candidate_total", candidate_rows),
        rows=getattr(stats, "rows", rows),
        vocab_size=kernel_vocab,
    )
    if os.environ.get("DRAFTSLICE_SGLANG_DEBUG") == "1":
        import sys

        print(
            f"[draftslice:sglang] step0 sliced rows={rows} "
            f"candidates={_STATS['candidate_rows']} stats={_STATS}",
            file=sys.stderr,
            flush=True,
        )
    return logits


def _patched_draft(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
    original = None
    for cls in type(self).__mro__:
        if cls in _ORIGINAL_DRAFTS:
            original = _ORIGINAL_DRAFTS[cls]
            break
    if original is None:
        raise SglangIntegrationError("SGLang draft hook is not installed")
    config = config_from_env()
    grammars = (
        [getattr(req, "grammar", None) for req in getattr(batch, "reqs", ())]
        if config.enabled and getattr(batch, "has_grammar", False)
        else None
    )
    if not grammars:
        grammars = (
            getattr(getattr(batch, "sampling_info", None), "grammars", None)
            if config.enabled and getattr(batch, "has_grammar", False)
            else None
        )
    if os.environ.get("DRAFTSLICE_SGLANG_DEBUG") == "1":
        import sys

        print(
            f"[draftslice:sglang] draft hook enabled={config.enabled} "
            f"has_grammar={getattr(batch, 'has_grammar', False)} "
            f"grammars={0 if not grammars else len(grammars)}",
            file=sys.stderr,
            flush=True,
        )
    state = (
        {
            "grammars": grammars,
            "step": 0,
            "config": config,
            "topk": int(getattr(self, "topk", 1)),
            "batch": len(grammars),
            "sliced_calls": 0,
        }
        if grammars
        else None
    )
    token = _DRAFT_STATE.set(state)
    try:
        return original(self, batch, *args, **kwargs)
    finally:
        _DRAFT_STATE.reset(token)


def install() -> None:
    """Install the SGLang draft/head hooks (idempotent)."""

    global _PATCHED, _ORIGINAL_DRAFT, _ORIGINAL_LM_HEAD
    if _PATCHED:
        return
    # Fail at startup, not on the first request, when a table/hash is bad.
    adapter_policy(config_from_env())
    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
        from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
            FrozenKVMTPDraftWorker,
        )
    except Exception as exc:  # pragma: no cover - engine optional
        raise SglangIntegrationError(
            f"cannot install the SGLang hook: {exc}"
        ) from exc
    for cls in (EagleDraftWorker, FrozenKVMTPDraftWorker):
        if cls.draft is not _patched_draft:
            _ORIGINAL_DRAFTS[cls] = cls.draft
            cls.draft = _patched_draft
    _ORIGINAL_DRAFT = _ORIGINAL_DRAFTS.get(EagleDraftWorker)
    _ORIGINAL_LM_HEAD = LogitsProcessor._compute_lm_head
    LogitsProcessor._compute_lm_head = _patched_compute_lm_head
    _PATCHED = True


def _sglang_status() -> tuple[bool, str]:
    try:
        import sglang  # noqa: F401
    except ImportError as exc:
        return False, f"sglang is not installed: {exc}"
    try:
        import sglang.srt.speculative.spec_info  # noqa: F401
    except Exception as exc:
        return False, f"sglang speculative interfaces are unavailable: {exc}"
    for module in (
        "sglang.srt.speculative.eagle_worker_v2",
        "sglang.srt.speculative.eagle_worker",
    ):
        try:
            __import__(module)
            return True, ""
        except Exception:
            continue
    return False, "sglang has no importable EAGLE draft worker"


def screen_logits(
    logits: Any,
    hidden_states: Any,
    valid_sets: list[TokenSet | None] | None,
    screener: Screener,
    *,
    universe: str = "valid",
) -> Any:
    """Mask logits to ``A ∩ V`` (or ``A`` for free-form rows).

    This pure helper remains available for hosts that prefer a post-head mask.
    The plugin path above slices the head instead.
    """

    if universe not in ("valid", "vocab"):
        raise SglangIntegrationError("universe must be valid or vocab")
    if logits.ndim != 2 or hidden_states.ndim != 2:
        raise SglangIntegrationError("logits and hidden_states must be 2-D")
    if int(logits.shape[0]) != int(hidden_states.shape[0]):
        raise SglangIntegrationError("logits and hidden_states row counts differ")
    candidates = screener.candidates(hidden_states)
    if len(candidates) != int(logits.shape[0]):
        raise SglangIntegrationError(
            f"screener returned {len(candidates)} rows for {int(logits.shape[0])}"
        )
    if valid_sets is None:
        valid_sets = [None] * len(candidates)
    if len(valid_sets) != len(candidates):
        raise SglangIntegrationError("valid_sets length does not match the batch")

    import torch

    selected: list[TokenSet | None] = []
    if universe == "valid":
        selected = [
            None if valid is None else candidate.intersection(valid)
            for candidate, valid in zip(candidates, valid_sets, strict=True)
        ]
    else:
        for candidate, valid in zip(candidates, valid_sets, strict=True):
            selected.append(candidate if valid is None else candidate.intersection(valid))

    out = logits.clone()
    for row, tokens in enumerate(selected):
        if tokens is None or tokens.is_empty:
            continue
        keep_ids = [token for token in tokens.ids if token < int(out.shape[1])]
        if not keep_ids:
            continue
        keep = torch.tensor(keep_ids, dtype=torch.long, device=logits.device)
        mask = torch.full_like(out[row], float("-inf"))
        mask[keep] = out[row, keep]
        out[row] = mask
    return out


def register() -> None:
    """SGLang general-plugin entry point."""

    global _REGISTERED
    if _REGISTERED:
        return
    available, reason = _sglang_status()
    if not available:
        raise SglangIntegrationError(
            f"DraftSlice SGLang boundary is unavailable: {reason}. "
            "Install a complete SGLang build with an importable EAGLE worker."
        )
    if config_from_env().enabled:
        install()
    _REGISTERED = True


__all__ = [
    "SglangConfig",
    "SglangIntegrationError",
    "config_from_env",
    "install",
    "register",
    "screen_logits",
]
