"""SGLang integration boundary.

SGLang's EAGLE worker captures draft logits in
``EagleWorker.capture_for_decode`` and then keeps the top-k probabilities. The
DraftSlice boundary is deliberately smaller than the vLLM adapter: it provides
the candidate-set transform and a guarded registration entry point.

Runtime status: the boundary and imports are validated against SGLang 0.5.23
(``eagle_worker_v2``). The engine hook still needs a per-request grammar-state
provider: it must run the draft LM head on a ``HeadPlan`` row set (not mask
logits after the full head), and keep the full head when no state exists.
``register()`` only claims availability after the speculative worker imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet
from ..screen.base import Screener

_REGISTERED = False


class SglangIntegrationError(RuntimeError):
    """Raised when the SGLang integration cannot be installed."""


@dataclass(frozen=True, slots=True)
class SglangConfig:
    """Environment-level switches for the boundary."""

    enabled: bool = True
    screener: str | None = None
    universe: str = "valid"


def config_from_env() -> SglangConfig:
    def flag(name: str, default: str = "1") -> bool:
        return os.environ.get(name, default) not in ("0", "false", "False")

    universe = os.environ.get("DRAFTSLICE_SGLANG_UNIVERSE", "valid").strip().lower()
    if universe not in ("valid", "vocab"):
        raise SglangIntegrationError("DRAFTSLICE_SGLANG_UNIVERSE must be valid or vocab")
    screener = os.environ.get("DRAFTSLICE_SGLANG_SCREENER", "").strip() or None
    return SglangConfig(
        enabled=flag("DRAFTSLICE_SGLANG"),
        screener=screener,
        universe=universe,
    )


def screen_logits(
    logits: Any,
    hidden_states: Any,
    valid_sets: list[TokenSet | None] | None,
    screener: Screener,
    *,
    universe: str = "valid",
) -> Any:
    """Mask logits to ``A ∩ V`` (or ``A`` for free-form rows).

    ``logits`` and ``hidden_states`` have the same batch rows. Rows without a
    plan (free-form with ``universe="valid"``, empty intersections, or screener
    errors handled by the caller) are returned unchanged. The input tensor is
    not mutated; a new tensor is returned.
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


def register() -> None:
    """Register the SGLang boundary.

    There is no stable public registration API across SGLang releases, so this
    function only checks the engine surface. The host integration must wrap the
    draft head with :func:`screen_logits` (or a plan-aware equivalent) and keep
    the full head when per-request grammar state is absent.
    """

    global _REGISTERED
    if _REGISTERED:
        return
    available, reason = _sglang_status()
    if not available:
        raise SglangIntegrationError(
            f"DraftSlice SGLang boundary is unavailable: {reason}. "
            "Install a complete SGLang build with an importable EAGLE worker "
            "before enabling the engine hook."
        )
    _REGISTERED = True


__all__ = [
    "SglangConfig",
    "SglangIntegrationError",
    "config_from_env",
    "register",
    "screen_logits",
]
