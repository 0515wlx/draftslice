"""Vocabulary partitions and leaf-contiguous layout helpers.

The partition is a deterministic k-means over normalized weight rows. It is not
claimed to be the best partition; it is a reproducible artifact builder that the
router/low-rank experiments and layout tests can share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet


class PartitionError(ValueError):
    """Raised when a partition cannot be built."""


@dataclass(frozen=True, slots=True)
class VocabularyPartition:
    groups: tuple[tuple[int, ...], ...]
    assignment: tuple[int, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.groups:
            raise PartitionError("partition needs at least one group")
        seen: set[int] = set()
        for group_id, group in enumerate(self.groups):
            if not group:
                raise PartitionError(f"group {group_id} is empty")
            for token_id in group:
                if token_id < 0:
                    raise PartitionError("token ids must be non-negative")
                if token_id in seen:
                    raise PartitionError(f"token {token_id} appears in two groups")
                seen.add(token_id)
        if len(self.assignment) != len(seen):
            raise PartitionError("assignment length does not match the token count")
        if any(group < 0 or group >= len(self.groups) for group in self.assignment):
            raise PartitionError("assignment points outside the group table")

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_tokens(self) -> int:
        return len(self.assignment)

    def group_of(self, token_id: int) -> int:
        if token_id < 0 or token_id >= len(self.assignment):
            raise IndexError(token_id)
        return self.assignment[token_id]

    def layout(self) -> tuple[int, ...]:
        """Return a leaf-contiguous permutation ``row -> token id``."""

        order: list[int] = []
        for group in self.groups:
            order.extend(group)
        return tuple(order)

    def token_to_row(self) -> tuple[int, ...]:
        order = self.layout()
        reverse = [0] * len(order)
        for row, token in enumerate(order):
            reverse[token] = row
        return tuple(reverse)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PartitionError("torch is required to build a partition") from exc
    return torch


def build_partition(
    weight: Any,
    num_groups: int,
    *,
    iters: int = 8,
    seed: int = 0,
) -> VocabularyPartition:
    """K-means partition over normalized weight rows."""

    t = _require_torch()
    if weight.ndim != 2:
        raise PartitionError("weight must be 2-D")
    vocab = int(weight.shape[0])
    if num_groups <= 0 or num_groups > vocab:
        raise PartitionError("num_groups must be in 1..vocab")
    if iters <= 0:
        raise PartitionError("iters must be positive")

    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    x = weight.detach().float()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
    init = t.randperm(vocab, generator=generator)[:num_groups]
    centroids = x.index_select(0, init.to(weight.device)).clone()
    assignment = t.zeros(vocab, dtype=t.long, device=weight.device)

    for _ in range(iters):
        # Chunk the score matrix so a 248k x 1024 partition fits comfortably.
        chunk = 262_144
        for start in range(0, vocab, chunk):
            end = min(start + chunk, vocab)
            scores = x[start:end] @ centroids.T
            assignment[start:end] = scores.argmax(dim=1)
        counts = t.bincount(assignment, minlength=num_groups).to(
            device=weight.device, dtype=x.dtype
        )
        updated = t.zeros_like(centroids)
        updated.index_add_(0, assignment, x)
        nonempty = counts > 0
        updated[nonempty] /= counts[nonempty].unsqueeze(1)
        updated[~nonempty] = centroids[~nonempty]
        centroids = updated

    # Make every group non-empty without changing most assignments: move the
    # farthest token from the largest group into each empty group.
    assignment_cpu = assignment.tolist()
    groups: list[list[int]] = [[] for _ in range(num_groups)]
    for token, group in enumerate(assignment_cpu):
        groups[group].append(token)
    for group_id, group in enumerate(groups):
        if group:
            continue
        donor = max(range(num_groups), key=lambda index: len(groups[index]))
        if len(groups[donor]) <= 1:
            raise PartitionError("cannot fill empty partition groups")
        token = groups[donor].pop()
        groups[group_id].append(token)
        assignment_cpu[token] = group_id

    return VocabularyPartition(
        groups=tuple(tuple(sorted(group)) for group in groups),
        assignment=tuple(assignment_cpu),
        seed=seed,
    )


def partition_tokens_for_groups(
    partition: VocabularyPartition,
    group_ids: list[int] | tuple[int, ...],
) -> TokenSet:
    tokens: list[int] = []
    for group_id in group_ids:
        if group_id < 0 or group_id >= partition.num_groups:
            raise PartitionError(f"group id {group_id} is out of range")
        tokens.extend(partition.groups[group_id])
    return TokenSet(tokens)


__all__ = [
    "PartitionError",
    "VocabularyPartition",
    "build_partition",
    "partition_tokens_for_groups",
]
