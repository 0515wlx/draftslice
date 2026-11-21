"""Reference candidate sources (screeners).

Every policy implements the :class:`draftslice.screen.base.Screener` protocol:
given draft ``hidden_states`` it returns one :class:`TokenSet` per batch row.
The plan compiler decides the universe (valid set or vocabulary), intersects the
candidates for constrained-calibrated mode, and keeps scoring inside the
candidate set exact.

These are reference implementations of the static, low-rank, cluster/group, and
router families (plus the non-deployable oracle upper bound). They are labelled
as reference baselines in reports, not as claims about another paper's exact
kernel or training recipe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet
from .base import ScreenError

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise ScreenError("torch is not installed")
    return torch


@dataclass(frozen=True, slots=True)
class FullPolicy:
    """Every vocabulary row; the pass-through / reference point."""

    name: str = "full"
    vocab_size: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ScreenError("vocab_size must be positive")

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        batch = int(hidden_states.shape[0])
        full = TokenSet(range(self.vocab_size))
        return [full for _ in range(batch)]


@dataclass(frozen=True, slots=True)
class StaticPolicy:
    """A fixed retained vocabulary (VocabTrim-style)."""

    rows: tuple[int, ...]
    name: str = "static"

    def __post_init__(self) -> None:
        if not self.rows:
            raise ScreenError("static policy needs at least one row")
        if any(row < 0 for row in self.rows):
            raise ScreenError("static rows must be non-negative")
        object.__setattr__(self, "rows", tuple(sorted(set(self.rows))))

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        batch = int(hidden_states.shape[0])
        tokens = TokenSet(self.rows)
        return [tokens for _ in range(batch)]


@dataclass(frozen=True, slots=True)
class GroupPolicy:
    """Score vocabulary groups by centroid dot product and take the top C.

    This is the reference cluster-shortlist baseline (FlashHead-like /
    DynaSpec-like depending on how the groups and centroids were produced).
    """

    groups: tuple[tuple[int, ...], ...]
    centroids: Any
    top_c: int
    name: str = "group_dot"

    def __post_init__(self) -> None:
        if not self.groups:
            raise ScreenError("group policy needs groups")
        if self.top_c <= 0:
            raise ScreenError("top_c must be positive")
        if len(self.centroids) != len(self.groups):
            raise ScreenError("centroids and groups must have equal length")
        if any(len(group) == 0 for group in self.groups):
            raise ScreenError("groups must not be empty")

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        scores = hidden_states @ self.centroids.T
        top = scores.topk(min(self.top_c, len(self.groups)), dim=-1).indices
        cache: dict[tuple[int, ...], TokenSet] = {}
        result: list[TokenSet] = []
        for row in top.tolist():
            key = tuple(sorted(row))
            tokens = cache.get(key)
            if tokens is None:
                ids: list[int] = []
                for group_id in key:
                    ids.extend(self.groups[group_id])
                tokens = TokenSet(ids)
                cache[key] = tokens
            result.append(tokens)
        return result


@dataclass(frozen=True, slots=True)
class RouterPolicy:
    """Score groups with an MLP router and take the top C (DynaSpec-like)."""

    groups: tuple[tuple[int, ...], ...]
    w1: Any
    w2: Any
    top_c: int
    name: str = "mlp_router"

    def __post_init__(self) -> None:
        if not self.groups:
            raise ScreenError("router policy needs groups")
        if self.top_c <= 0:
            raise ScreenError("top_c must be positive")
        if int(self.w2.shape[0]) != len(self.groups):
            raise ScreenError("w2 output size must match the group count")

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        scores = F.gelu(hidden_states @ self.w1.T) @ self.w2.T
        top = scores.topk(min(self.top_c, len(self.groups)), dim=-1).indices
        cache: dict[tuple[int, ...], TokenSet] = {}
        result: list[TokenSet] = []
        for row in top.tolist():
            key = tuple(sorted(row))
            tokens = cache.get(key)
            if tokens is None:
                ids: list[int] = []
                for group_id in key:
                    ids.extend(self.groups[group_id])
                tokens = TokenSet(ids)
                cache[key] = tokens
            result.append(tokens)
        return result


@dataclass(frozen=True, slots=True)
class LowRankPolicy:
    """Top-m tokens by a low-rank score (SVD or trained low-rank)."""

    vr: Any  # [D, r]
    a: Any  # [V, r]
    top_m: int
    name: str = "lowrank"

    def __post_init__(self) -> None:
        if self.top_m <= 0:
            raise ScreenError("top_m must be positive")
        if int(self.vr.shape[1]) != int(self.a.shape[1]):
            raise ScreenError("vr and a have incompatible ranks")

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        scores = (hidden_states @ self.vr) @ self.a.T
        top = scores.topk(min(self.top_m, scores.shape[-1]), dim=-1).indices
        return [TokenSet(row) for row in top.tolist()]

    @classmethod
    def from_weight(
        cls,
        weight: Any,
        *,
        rank: int,
        top_m: int,
        name: str = "svd_lowrank",
    ) -> LowRankPolicy:
        t = _require_torch()
        if rank <= 0:
            raise ScreenError("rank must be positive")
        original_dtype = weight.dtype
        matrix = weight.float()
        _, _, v = t.pca_lowrank(matrix, q=min(rank, min(matrix.shape)), niter=2)
        vr = v[:, :rank].contiguous()
        a = (matrix @ vr).contiguous()
        return cls(
            vr=vr.to(dtype=original_dtype),
            a=a.to(dtype=original_dtype),
            top_m=top_m,
            name=name,
        )

    @classmethod
    def from_trained(
        cls,
        *,
        vt: Any,
        u: Any,
        top_m: int,
        name: str = "trained_lowrank",
    ) -> LowRankPolicy:
        if vt.ndim != 2 or u.ndim != 2:
            raise ScreenError("vt and u must be 2-D")
        if int(vt.shape[0]) != int(u.shape[1]):
            raise ScreenError("vt and u have incompatible ranks")
        return cls(vr=vt.T.contiguous(), a=u, top_m=top_m, name=name)


@dataclass(frozen=True, slots=True)
class OracleGroupPolicy:
    """Top-C by true group mass; non-deployable upper bound."""

    groups: tuple[tuple[int, ...], ...]
    group_mass: Any  # [batch, groups]
    top_c: int
    name: str = "oracle_group_mass"

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        batch = int(hidden_states.shape[0])
        if int(self.group_mass.shape[0]) < batch:
            raise ScreenError("group_mass has fewer rows than the batch")
        top = self.group_mass[:batch].topk(min(self.top_c, len(self.groups)), dim=-1).indices
        result = []
        for row in top.tolist():
            ids: list[int] = []
            for group_id in row:
                ids.extend(self.groups[group_id])
            result.append(TokenSet(ids))
        return result


def group_centroids(weight: Any, groups: Sequence[Sequence[int]]) -> Any:
    """Mean row of each group, in the weight dtype."""

    t = _require_torch()
    if not groups:
        raise ScreenError("groups must not be empty")
    centroids = []
    for group in groups:
        if not group:
            raise ScreenError("groups must not be empty")
        index = t.tensor(list(group), dtype=t.long, device=weight.device)
        centroids.append(weight.index_select(0, index).float().mean(0))
    return t.stack(centroids).to(dtype=weight.dtype)


def make_group_policy(
    weight: Any,
    groups: Sequence[Sequence[int]],
    *,
    top_c: int,
    name: str = "group_dot",
) -> GroupPolicy:
    return GroupPolicy(
        groups=tuple(tuple(group) for group in groups),
        centroids=group_centroids(weight, groups),
        top_c=top_c,
        name=name,
    )


__all__ = [
    "FullPolicy",
    "GroupPolicy",
    "LowRankPolicy",
    "OracleGroupPolicy",
    "RouterPolicy",
    "StaticPolicy",
    "group_centroids",
    "make_group_policy",
]
