"""Reference CFHead/FlashHead/VocabTrim-style candidate policies.

The algorithms follow the official CFHead repository
(https://github.com/HigherIdeal/CFHead, IEEE TC 2026). This is a reimplementation
for the shared ``CandidatePolicy`` interface, not a copy of the official
evaluation harness. It is a reference baseline and must be labelled as such.

CFHead:
    split the hidden dimension into tiles; k-means each tile independently;
    store a codebook and an index map; score every token with LUT lookup and
    reduction; take top-k candidates.

FlashHead:
    balanced spherical k-means over vocabulary rows; score cluster centroids;
    probe the top-p clusters; union their token blocks.

VocabTrim is a static retained-vocabulary baseline; use ``StaticPolicy`` with
``name="vocabtrim"`` or the helper below.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..plan.types import TokenSet
from .baselines import BaselineError, StaticPolicy

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> Any:
    if torch is None:
        raise BaselineError("torch is not installed")
    return torch


def _validate_weight(weight: Any) -> tuple[int, int]:
    t = _require_torch()
    if not isinstance(weight, t.Tensor) or weight.ndim != 2:
        raise BaselineError("weight must be a 2-D torch tensor")
    return int(weight.shape[0]), int(weight.shape[1])


@dataclass(frozen=True, slots=True)
class CFHeadPolicy:
    """CFHead-style tile-wise coarse screening policy."""

    weight: Any
    tile_size: int = 8
    cluster_size: int = 4
    top_k: int = 8192
    kmeans_iters: int = 10
    seed: int = 99
    name: str = "cfhead"
    bias: Any | None = None
    _vocab_size: int = field(init=False, repr=False, compare=False)
    _hidden_size: int = field(init=False, repr=False, compare=False)
    _num_tiles: int = field(init=False, repr=False, compare=False)
    _codebook: Any = field(init=False, repr=False, compare=False)
    _index_map: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        t = _require_torch()
        vocab, hidden = _validate_weight(self.weight)
        if self.tile_size <= 0 or hidden % self.tile_size != 0:
            raise BaselineError("tile_size must divide the hidden dimension")
        if self.cluster_size <= 1 or self.cluster_size > 256:
            raise BaselineError("cluster_size must be in 2..256")
        if self.top_k <= 0 or self.top_k > vocab:
            raise BaselineError("top_k must be in 1..vocab_size")
        if self.kmeans_iters <= 0:
            raise BaselineError("kmeans_iters must be positive")
        codebook, index_map = self._build(t)
        object.__setattr__(self, "_vocab_size", vocab)
        object.__setattr__(self, "_hidden_size", hidden)
        object.__setattr__(self, "_num_tiles", hidden // self.tile_size)
        object.__setattr__(self, "_codebook", codebook)
        object.__setattr__(self, "_index_map", index_map)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def storage_bytes(self) -> int:
        codebook = self._codebook.numel() * self._codebook.element_size()
        index = self._index_map.numel() * self._index_map.element_size()
        return codebook + index

    def _build(self, t: Any) -> tuple[Any, Any]:
        weight = self.weight
        vocab, hidden = _validate_weight(weight)
        tiles = hidden // self.tile_size
        codebook = t.empty(
            (tiles, self.cluster_size, self.tile_size),
            device=weight.device,
            dtype=t.float32,
        )
        index_map = t.empty((vocab, tiles), device=weight.device, dtype=t.uint8)
        generator = t.Generator(device="cpu")
        generator.manual_seed(self.seed)

        for tile in range(tiles):
            start = tile * self.tile_size
            values = weight[:, start : start + self.tile_size].float()
            init = t.randperm(vocab, generator=generator)[: self.cluster_size]
            centroids = values.index_select(0, init.to(weight.device)).clone()
            assignment = None
            for _ in range(self.kmeans_iters):
                distances = t.cdist(values, centroids, p=2)
                assignment = t.argmin(distances, dim=1)
                counts = t.bincount(assignment, minlength=self.cluster_size).to(
                    device=weight.device, dtype=t.float32
                )
                updated = t.zeros_like(centroids)
                updated.index_add_(0, assignment, values)
                nonempty = counts > 0
                updated[nonempty] /= counts[nonempty].unsqueeze(1)
                updated[~nonempty] = centroids[~nonempty]
                centroids = updated
            assert assignment is not None
            codebook[tile] = centroids
            index_map[:, tile] = assignment.to(t.uint8)

        return codebook.to(dtype=weight.dtype), index_map

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        t = _require_torch()
        hidden = hidden_states
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 2 or hidden.shape[1] != self._hidden_size:
            raise BaselineError("hidden shape does not match the policy")
        batch = hidden.shape[0]
        tiles = hidden.reshape(batch, self._num_tiles, self.tile_size).float()
        codebook = self._codebook.to(device=hidden.device)
        lut = t.einsum("bnt,nct->bnc", tiles, codebook.float())
        index = self._index_map.to(device=hidden.device, dtype=t.long).t().contiguous()
        coarse = t.zeros((batch, self._vocab_size), device=hidden.device, dtype=t.float32)
        chunk = 32
        for start in range(0, self._num_tiles, chunk):
            end = min(start + chunk, self._num_tiles)
            part = lut[:, start:end, :]
            idx = index[start:end].unsqueeze(0).expand(batch, end - start, self._vocab_size)
            coarse += t.gather(part, dim=2, index=idx).sum(dim=1)
        if self.bias is not None:
            coarse += self.bias.to(device=coarse.device, dtype=coarse.dtype).unsqueeze(0)
        top = coarse.topk(self.top_k, dim=-1, largest=True, sorted=False).indices
        return [TokenSet(row) for row in top.tolist()]


@dataclass(frozen=True, slots=True)
class FlashHeadPolicy:
    """FlashHead-style balanced-cluster retrieval policy."""

    weight: Any
    num_clusters: int = 8016
    num_probes: int = 512
    cluster_iters: int = 5
    top_choices: int = 16
    seed: int = 99
    name: str = "flashhead"
    bias: Any | None = None
    _vocab_size: int = field(init=False, repr=False, compare=False)
    _hidden_size: int = field(init=False, repr=False, compare=False)
    _tokens_per_cluster: int = field(init=False, repr=False, compare=False)
    _centroids: Any = field(init=False, repr=False, compare=False)
    _c2t: Any = field(init=False, repr=False, compare=False)
    _valid: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        t = _require_torch()
        vocab, hidden = _validate_weight(self.weight)
        if self.num_clusters <= 1 or self.num_clusters > vocab:
            raise BaselineError("num_clusters must be in 2..vocab_size")
        if self.num_probes <= 0 or self.num_probes > self.num_clusters:
            raise BaselineError("num_probes must be in 1..num_clusters")
        if self.cluster_iters <= 0 or self.top_choices <= 0:
            raise BaselineError("cluster_iters and top_choices must be positive")
        tokens_per_cluster = math.ceil(vocab / self.num_clusters)
        centroids, c2t, valid = self._build(t, vocab, hidden, tokens_per_cluster)
        object.__setattr__(self, "_vocab_size", vocab)
        object.__setattr__(self, "_hidden_size", hidden)
        object.__setattr__(self, "_tokens_per_cluster", tokens_per_cluster)
        object.__setattr__(self, "_centroids", centroids)
        object.__setattr__(self, "_c2t", c2t)
        object.__setattr__(self, "_valid", valid)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def candidate_tokens(self) -> int:
        return self.num_probes * self._tokens_per_cluster

    def _build(self, t: Any, vocab: int, hidden: int, capacity: int) -> tuple[Any, Any, Any]:
        weight = self.weight
        x = F.normalize(weight.float(), dim=1).contiguous()
        generator = t.Generator(device="cpu")
        generator.manual_seed(self.seed)
        perm = t.randperm(vocab, generator=generator)
        assignment = t.empty((vocab,), dtype=t.long)
        for position, token_id in enumerate(perm.tolist()):
            assignment[token_id] = min(position // capacity, self.num_clusters - 1)
        centroids = t.empty((self.num_clusters, hidden), device=weight.device, dtype=t.float32)
        for iteration in range(self.cluster_iters):
            centroids.zero_()
            device_assignment = assignment.to(weight.device)
            centroids.index_add_(0, device_assignment, x)
            counts = (
                t.bincount(device_assignment, minlength=self.num_clusters)
                .to(device=weight.device, dtype=t.float32)
                .clamp_min_(1.0)
            )
            centroids = F.normalize(centroids / counts.unsqueeze(1), dim=1)
            if iteration < self.cluster_iters - 1:
                assignment = self._balanced_assign(
                    t, x, centroids, capacity, seed=self.seed + iteration + 1
                )

        c2t = t.full((self.num_clusters, capacity), -1, dtype=t.long)
        valid = t.zeros((self.num_clusters, capacity), dtype=t.bool)
        fill = [0] * self.num_clusters
        for token_id, cluster_id in enumerate(assignment.tolist()):
            position = fill[cluster_id]
            if position < capacity:
                c2t[cluster_id, position] = token_id
                valid[cluster_id, position] = True
                fill[cluster_id] += 1
        c2t[~valid] = 0
        return centroids.to(dtype=weight.dtype), c2t, valid

    def _balanced_assign(
        self,
        t: Any,
        x: Any,
        centroids: Any,
        capacity: int,
        *,
        seed: int,
    ) -> Any:
        vocab = int(x.shape[0])
        generator = t.Generator(device="cpu")
        generator.manual_seed(seed)
        order = t.randperm(vocab, generator=generator)
        capacities = [capacity] * self.num_clusters
        assignment = t.empty((vocab,), dtype=t.long)
        top_choices = min(self.top_choices, self.num_clusters)
        chunk = 512
        for start in range(0, vocab, chunk):
            token_cpu = order[start : start + chunk]
            token_dev = token_cpu.to(x.device)
            scores = x.index_select(0, token_dev) @ centroids.t()
            choices = scores.topk(top_choices, dim=1).indices.cpu()
            for row, token_id in enumerate(token_cpu.tolist()):
                chosen = -1
                for cluster_id in choices[row].tolist():
                    if capacities[cluster_id] > 0:
                        chosen = cluster_id
                        break
                if chosen < 0:
                    chosen = max(range(self.num_clusters), key=lambda cid: capacities[cid])
                assignment[token_id] = chosen
                capacities[chosen] -= 1
        return assignment

    def candidates(self, hidden_states: Any) -> list[TokenSet]:
        _require_torch()
        hidden = hidden_states
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 2 or hidden.shape[1] != self._hidden_size:
            raise BaselineError("hidden shape does not match the policy")
        centroids = self._centroids.to(device=hidden.device)
        scores = hidden.float() @ centroids.float().t()
        selected = scores.topk(self.num_probes, dim=-1).indices
        candidates = self._c2t.to(hidden.device)[selected]
        valid = self._valid.to(hidden.device)[selected]
        result = []
        for row in range(hidden.shape[0]):
            ids = candidates[row][valid[row]].tolist()
            result.append(TokenSet(ids))
        return result


def make_vocabtrim_policy(rows: Sequence[int]) -> StaticPolicy:
    """Static retained-vocabulary baseline under the official method name."""

    return StaticPolicy(tuple(rows), name="vocabtrim")


__all__ = [
    "CFHeadPolicy",
    "FlashHeadPolicy",
    "make_vocabtrim_policy",
]
