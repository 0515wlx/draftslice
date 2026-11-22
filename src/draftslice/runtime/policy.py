"""Single-profile policy: the only place that decides slice vs built-in.

Decision order (``docs/production-architecture.md`` §4, must not be reordered):

1. no grammar in the step / free-form rows -> built-in;
2. ``profile_mode=off`` -> built-in; no table -> built-in;
3. capability failure -> full head + stable reason;
4. exact profile cell lookup (bucket crossing is a miss);
5. disabled or engine-mismatched cell -> built-in;
6. ``observe`` executes built-in but reports the matched cell;
7. ``enforce`` executes the cell's action.

A miss never falls back to a "nearby" cell. Rows that hit a cell are marked in
``PolicyDecision.row_mask``; the engine adapter passes only those rows an exact
valid set and leaves the rest on the full head, so a mixed batch is supported
without a second decision table.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .profile import ProfileTable, ProfitCell, batch_bucket, valid_bucket
from .reasons import FallbackReason

PROFILE_MODES = frozenset({"off", "observe", "enforce"})
ACTIONS = frozenset({"builtin", "full", "slice", "mask", "mask_allpos"})
MODES = frozenset({"builtin", "full", "strict", "mask", "mask_allpos"})

#: Capability probe -> reason code when it fails.
CAPABILITY_REASONS: dict[str, FallbackReason] = {
    "grammar": FallbackReason.NO_GRAMMAR_STATE,
    "head_layout": FallbackReason.UNSUPPORTED_HEAD_LAYOUT,
    "quant": FallbackReason.UNSUPPORTED_QUANT_SCHEME,
    "tp": FallbackReason.TP_COLLECTIVE_UNAVAILABLE,
    "graph": FallbackReason.FULL_CUDA_GRAPH,
    "temperature": FallbackReason.CAPABILITY_UNSUPPORTED,
    "local_argmax": FallbackReason.CAPABILITY_UNSUPPORTED,
}

_ACTION_TO_MODE = {
    "strict_slice": "strict",
    "mask": "mask",
    "mask_allpos": "mask_allpos",
}
_ACTION_TO_EXECUTION = {
    "strict_slice": "slice",
    "mask": "mask",
    "mask_allpos": "mask_allpos",
}


class PolicyError(ValueError):
    """The policy cannot be constructed safely."""


@dataclass(frozen=True, slots=True)
class RowFacts:
    """Per-row inputs for an exact cell lookup."""

    workload: str = "free"
    valid_size: int = 0

    def __post_init__(self) -> None:
        if self.valid_size < 0:
            raise PolicyError("valid_size must be non-negative")


@dataclass(frozen=True, slots=True)
class StepFacts:
    """Everything the policy is allowed to know about one draft step."""

    engine: str
    model: str
    method: str
    batch: int
    k: int
    tp: int = 1
    quant: str = "bf16"
    graph: str = "eager"
    rows: tuple[RowFacts, ...] = ()
    #: Capability probes; a missing key means "not checked / assume supported".
    capabilities: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in ("vllm", "sglang", "reference"):
            raise PolicyError(f"invalid engine: {self.engine!r}")
        if not self.model:
            raise PolicyError("model must not be empty")
        if not self.method:
            raise PolicyError("method must not be empty")
        if self.batch < 1:
            raise PolicyError("batch must be >= 1")
        if self.k < 0:
            raise PolicyError("k must be >= 0")
        if self.tp < 1:
            raise PolicyError("tp must be >= 1")

    @property
    def has_grammar(self) -> bool:
        return any(row.valid_size > 0 for row in self.rows)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """The step-level action plus the per-row hit mask."""

    action: str
    mode: str
    profile_mode: str
    reason: FallbackReason
    cell: ProfitCell | None = None
    row_mask: tuple[bool, ...] = ()
    workload: str = ""
    batch_bucket: str = ""
    valid_bucket: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise PolicyError(f"invalid action: {self.action!r}")
        if self.mode not in MODES:
            raise PolicyError(f"invalid mode: {self.mode!r}")
        if self.profile_mode not in PROFILE_MODES:
            raise PolicyError(f"invalid profile_mode: {self.profile_mode!r}")

    @property
    def cell_hit(self) -> bool:
        return self.cell is not None

    @property
    def cell_id(self) -> str | None:
        return None if self.cell is None else self.cell.cell_id

    @property
    def executes_slice(self) -> bool:
        return self.action not in ("builtin", "full")

    @property
    def sliced_rows(self) -> int:
        return sum(1 for hit in self.row_mask if hit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mode": self.mode,
            "profile_mode": self.profile_mode,
            "reason": self.reason.value,
            "cell_id": self.cell_id,
            "cell_hit": self.cell_hit,
            "sliced_rows": self.sliced_rows,
            "workload": self.workload,
            "batch_bucket": self.batch_bucket,
            "valid_bucket": self.valid_bucket,
        }


def _builtin(
    reason: FallbackReason,
    *,
    profile_mode: str,
    cell: ProfitCell | None = None,
    rows: int = 0,
) -> PolicyDecision:
    return PolicyDecision(
        action="builtin",
        mode="builtin",
        profile_mode=profile_mode,
        reason=reason,
        cell=cell,
        row_mask=(False,) * rows,
    )


def _full(
    reason: FallbackReason,
    *,
    profile_mode: str,
    cell: ProfitCell | None = None,
    rows: int = 0,
) -> PolicyDecision:
    return PolicyDecision(
        action="full",
        mode="full",
        profile_mode=profile_mode,
        reason=reason,
        cell=cell,
        row_mask=(False,) * rows,
    )


class ProfilePolicy:
    """Exact-cell policy over one :class:`ProfileTable`."""

    def __init__(
        self,
        table: ProfileTable | None,
        *,
        profile_mode: str = "observe",
        engine: str = "vllm",
        model: str | None = None,
        method: str | None = None,
    ) -> None:
        if profile_mode not in PROFILE_MODES:
            raise PolicyError(f"invalid profile_mode: {profile_mode!r}")
        if engine not in ("vllm", "sglang", "reference"):
            raise PolicyError(f"invalid engine: {engine!r}")
        if profile_mode == "enforce" and (table is None or not table.cells):
            raise PolicyError("profile_mode='enforce' requires a non-empty profile table")
        self.table = table
        self.profile_mode = profile_mode
        self.engine = engine
        self.model = model or (table.model if table is not None else "")
        self.method = method or ""
        # Row lookups repeat across the draft steps of one propose (same
        # workload/k/batch, usually the same valid bucket). Memoize the exact
        # match so the per-step policy does not re-parse bucket labels.
        self._lookup_cache: dict[tuple[Any, ...], Any] = {}
        self._lookup_cache_max = 256

    @property
    def enabled(self) -> bool:
        """Whether any cell could execute DraftSlice."""

        return bool(self.table is not None and self.table.enabled_cells)

    # -- decisions --------------------------------------------------------
    def decide(self, facts: StepFacts) -> PolicyDecision:
        rows = len(facts.rows)
        if self.profile_mode == "off":
            return _builtin(FallbackReason.MODE_OFF, profile_mode=self.profile_mode)
        if self.table is None or not self.table.cells:
            return _builtin(
                FallbackReason.PROFILES_UNAVAILABLE, profile_mode=self.profile_mode
            )
        if not facts.has_grammar:
            # No row can be constrained: the stock path is the correct one.
            return _builtin(
                FallbackReason.ROUTER_FREEFORM, profile_mode=self.profile_mode
            )

        capability_reason = self._capability_failure(facts)
        if capability_reason is not None:
            return _full(
                capability_reason, profile_mode=self.profile_mode, rows=rows
            )

        row_mask: list[bool] = []
        matched: list[ProfitCell] = []
        disabled: ProfitCell | None = None
        first_reason = FallbackReason.NO_CELL
        for row in facts.rows:
            if row.workload == "free" or row.valid_size <= 0:
                row_mask.append(False)
                continue
            cache_key = (
                facts.model,
                facts.engine,
                facts.method,
                row.workload,
                facts.batch,
                facts.k,
                facts.tp,
                facts.quant,
                facts.graph,
                row.valid_size,
            )
            match = self._lookup_cache.get(cache_key)
            if match is None:
                match = self.table.lookup(
                    model=facts.model,
                    engine=facts.engine,
                    method=facts.method,
                    workload=row.workload,
                    batch=facts.batch,
                    k=facts.k,
                    tp=facts.tp,
                    quant=facts.quant,
                    graph=facts.graph,
                    valid=row.valid_size,
                )
                if len(self._lookup_cache) >= self._lookup_cache_max:
                    self._lookup_cache.clear()
                self._lookup_cache[cache_key] = match
            if match.cell is not None and not match.hit:
                disabled = match.cell
            if not match.hit:
                if first_reason is FallbackReason.NO_CELL:
                    first_reason = match.reason
                row_mask.append(False)
                continue
            assert match.cell is not None
            matched.append(match.cell)
            row_mask.append(True)

        if not matched:
            return _builtin(
                FallbackReason.PROFILE_DISABLED
                if disabled is not None
                else first_reason,
                profile_mode=self.profile_mode,
                cell=disabled,
                rows=rows,
            )

        actions = {cell.action for cell in matched}
        if len(actions) > 1:
            return _builtin(
                FallbackReason.MIXED_CELL_ACTIONS,
                profile_mode=self.profile_mode,
                cell=matched[0],
                rows=rows,
            )
        action = actions.pop()
        cell = matched[0]
        if self.profile_mode == "observe":
            # Observe must be byte-for-byte the built-in path.
            return _builtin(
                FallbackReason.MODE_OBSERVE,
                profile_mode=self.profile_mode,
                cell=cell,
                rows=rows,
            )
        return PolicyDecision(
            action=_ACTION_TO_EXECUTION[action],
            mode=_ACTION_TO_MODE[action],
            profile_mode=self.profile_mode,
            reason=FallbackReason.NO_CELL,
            cell=cell,
            row_mask=tuple(row_mask),
            workload=cell.workload,
            batch_bucket=cell.batch_bucket,
            valid_bucket=cell.valid_bucket,
        )

    def _capability_failure(self, facts: StepFacts) -> FallbackReason | None:
        for name, reason in CAPABILITY_REASONS.items():
            if facts.capabilities.get(name, True):
                continue
            # Abstract capabilities that are not meaningful for an engine are
            # still checked only when the adapter explicitly reports them.
            return reason
        return None

    def classify(
        self, *, workload: str, valid_size: int, batch: int
    ) -> tuple[str, str]:
        """Bucket labels for logging (no lookup)."""

        return batch_bucket(batch), valid_bucket(valid_size)


def policy_from_env(
    engine: str,
    *,
    model: str | None = None,
    method: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProfilePolicy | None:
    """Build a policy from ``DRAFTSLICE_PROFILE_TABLE``/``_MODE``.

    Returns ``None`` when no table is configured: the caller keeps the legacy
    research override (``DRAFTSLICE_*_SLICE``) and the built-in default. An
    invalid table or hash raises instead of silently serving built-in.
    """

    env = os.environ if environ is None else environ
    table_path = env.get("DRAFTSLICE_PROFILE_TABLE")
    mode = env.get("DRAFTSLICE_PROFILE_MODE", "observe")
    if mode not in PROFILE_MODES:
        raise PolicyError(f"invalid DRAFTSLICE_PROFILE_MODE: {mode!r}")
    if not table_path:
        return None
    table = ProfileTable.load(table_path)
    return ProfilePolicy(
        table,
        profile_mode=mode,
        engine=engine,
        model=model,
        method=method,
    )


__all__ = [
    "ACTIONS",
    "CAPABILITY_REASONS",
    "MODES",
    "PROFILE_MODES",
    "PolicyDecision",
    "PolicyError",
    "ProfilePolicy",
    "RowFacts",
    "StepFacts",
    "policy_from_env",
]
