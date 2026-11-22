"""Engine-agnostic vocabulary-parallel head shard detection and row mapping.

Under tensor parallelism an LM head owns only a contiguous slice of the global
vocabulary. Its weight rows are laid out as ``[base slice][added slice]`` (the
layout vLLM's ``VocabParallelEmbedding`` uses; SGLang uses the same convention).
Slicing the head per rank therefore means:

1. detect the rank's global token-id ranges from the head object;
2. map a global grammar valid set to the rank's *local* weight rows;
3. scatter the resulting logits back to *global* token-id columns.

When the layout cannot be proven, detection returns ``None`` and the caller
fails closed to the full head (with a recorded reason). This module never
imports an engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet


@dataclass(frozen=True, slots=True)
class HeadShard:
    """One rank's view of a vocabulary-parallel LM head.

    ``vocab_start``/``vocab_end`` cover the rank's share of the original
    vocabulary. ``added_*`` describe the optional LoRA/added vocabulary that
    lives after the padded base slice in the same weight tensor.
    """

    tp_size: int
    tp_rank: int
    vocab_start: int
    vocab_end: int
    shard_rows: int
    org_vocab_size: int
    padded_vocab_size: int
    # Padded size of the *base* vocabulary across all ranks. Used to reproduce
    # every rank's id range locally, so collective decisions are identical.
    org_vocab_padded: int = 0
    added_vocab_start: int | None = None
    added_vocab_end: int | None = None
    added_local_start: int | None = None

    def local_row(self, token_id: int) -> int | None:
        """Local weight row for a *global* token id, or ``None`` if not owned."""

        if self.vocab_start <= token_id < self.vocab_end:
            return token_id - self.vocab_start
        if (
            self.added_vocab_start is not None
            and self.added_vocab_end is not None
            and self.added_local_start is not None
            and self.added_vocab_start <= token_id < self.added_vocab_end
        ):
            return self.added_local_start + (token_id - self.added_vocab_start)
        return None

    def owns(self, token_id: int) -> bool:
        return self.local_row(token_id) is not None

    def rank_ranges(self) -> list[tuple[int, int]]:
        """All ranks' original-vocab id ranges, computable on every rank.

        The caller uses this to make a full-head fallback decision that is the
        same on every rank. Otherwise only some ranks would enter vLLM's logits
        all-gather and the engine would deadlock.
        """

        padded = self.org_vocab_padded or self.shard_rows * self.tp_size
        per = max(1, padded // self.tp_size)
        return [
            (min(rank * per, self.org_vocab_size), min((rank + 1) * per, self.org_vocab_size))
            for rank in range(self.tp_size)
        ]

    def local_counts(self, token_set: TokenSet) -> list[int]:
        """How many of ``token_set``'s ids each rank owns (base vocab only)."""

        ranges = self.rank_ranges()
        counts = [0] * self.tp_size
        for token_id in token_set.ids:
            for rank, (start, end) in enumerate(ranges):
                if start <= token_id < end:
                    counts[rank] += 1
                    break
        return counts

    def global_id(self, local_row: int) -> int | None:
        """Inverse of :meth:`local_row` for real (non-padding) rows."""

        if 0 <= local_row < self.vocab_end - self.vocab_start:
            return self.vocab_start + local_row
        if (
            self.added_local_start is not None
            and self.added_vocab_start is not None
            and self.added_local_start <= local_row < self.shard_rows
        ):
            return self.added_vocab_start + (local_row - self.added_local_start)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "vocab_start": self.vocab_start,
            "vocab_end": self.vocab_end,
            "shard_rows": self.shard_rows,
            "org_vocab_size": self.org_vocab_size,
            "padded_vocab_size": self.padded_vocab_size,
            "added_vocab_start": self.added_vocab_start,
            "added_vocab_end": self.added_vocab_end,
        }

    @classmethod
    def unsharded(cls, weight_rows: int, vocab_size: int) -> HeadShard | None:
        """Trivial shard for TP=1; ``None`` if the head cannot cover the vocab."""

        if weight_rows < vocab_size or vocab_size <= 0:
            return None
        return cls(
            tp_size=1,
            tp_rank=0,
            vocab_start=0,
            vocab_end=vocab_size,
            shard_rows=weight_rows,
            org_vocab_size=vocab_size,
            padded_vocab_size=weight_rows,
            org_vocab_padded=weight_rows,
        )

    @classmethod
    def from_lm_head(
        cls,
        lm_head: Any,
        *,
        weight_rows: int,
        vocab_size: int,
    ) -> HeadShard | None:
        """Detect the shard layout of a vLLM/SGLang LM head.

        Returns ``None`` (never raises) for quantized heads, missing metadata,
        or inconsistent layouts so the adapter can fail closed.
        """

        # A quant config or quant method on the module does not mean the weight
        # itself is quantized (vLLM keeps bf16 heads under `--quantization fp8`).
        # Dtype is validated by the adapter; shard layout is independent of it.
        try:
            tp_size = int(getattr(lm_head, "tp_size", 1) or 1)
            tp_rank = getattr(lm_head, "tp_rank", None)
            tp_rank = None if tp_rank is None else int(tp_rank)
        except (TypeError, ValueError):
            return None
        if tp_size < 1:
            return None
        if weight_rows <= 0 or vocab_size <= 0:
            return None
        if tp_size == 1:
            return cls.unsharded(weight_rows, vocab_size)

        indices = getattr(lm_head, "shard_indices", None)
        if indices is None and tp_rank is None:
            return None
        if tp_rank is None:
            # SGLang's VocabParallelEmbedding does not expose tp_rank directly;
            # derive it from the padded base-vocab start of this shard.
            try:
                per = int(indices.num_org_elements_padded)
                tp_rank = int(indices.padded_org_vocab_start_index) // per
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                return None
        if not 0 <= tp_rank < tp_size:
            return None
        per_partition = getattr(lm_head, "num_embeddings_per_partition", None)
        padded = getattr(lm_head, "num_embeddings_padded", None)
        org_vocab = getattr(lm_head, "org_vocab_size", None)
        if indices is None or per_partition is None or padded is None or org_vocab is None:
            return None
        try:
            per_partition = int(per_partition)
            padded = int(padded)
            org_vocab = int(org_vocab)
            vocab_start = int(indices.org_vocab_start_index)
            vocab_end = int(indices.org_vocab_end_index)
            added_start = int(indices.added_vocab_start_index)
            added_end = int(indices.added_vocab_end_index)
            org_elements_padded = int(indices.num_org_elements_padded)
        except (AttributeError, TypeError, ValueError):
            return None

        if per_partition != weight_rows:
            return None
        if padded != tp_size * per_partition:
            return None
        if not (0 <= vocab_start <= vocab_end <= org_vocab):
            return None
        if org_elements_padded < vocab_end - vocab_start:
            return None
        if not (0 <= added_start <= added_end <= vocab_size):
            return None
        if org_elements_padded > per_partition:
            return None
        added_local_start = org_elements_padded if added_end > added_start else None
        if (
            added_local_start is not None
            and added_local_start + (added_end - added_start) > per_partition
        ):
            return None
        return cls(
            tp_size=tp_size,
            tp_rank=tp_rank,
            vocab_start=vocab_start,
            vocab_end=vocab_end,
            shard_rows=per_partition,
            org_vocab_size=org_vocab,
            padded_vocab_size=padded,
            org_vocab_padded=org_elements_padded * tp_size,
            added_vocab_start=added_start if added_end > added_start else None,
            added_vocab_end=added_end if added_end > added_start else None,
            added_local_start=added_local_start,
        )


@dataclass(frozen=True, slots=True)
class RankCandidates:
    """Local weight rows and the global token ids they score."""

    local: TokenSet
    global_ids: tuple[int, ...]


def restrict_to_rank(
    valid_sets: Sequence[TokenSet | None],
    shard: HeadShard,
    *,
    max_rows: int,
) -> list[RankCandidates | None]:
    """Map each global valid set to this rank's local rows.

    ``None`` means this rank cannot slice the row; the caller must use the full
    head for it. No valid token is dropped: tokens owned by other ranks are
    simply scored by those ranks.
    """

    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    out: list[RankCandidates | None] = []
    for token_set in valid_sets:
        if token_set is None or token_set.is_empty or len(token_set) > max_rows:
            out.append(None)
            continue
        local: list[int] = []
        ids: list[int] = []
        for token_id in token_set.ids:
            row = shard.local_row(token_id)
            if row is None or row >= shard.shard_rows:
                continue
            local.append(row)
            ids.append(token_id)
        if not local:
            out.append(None)
            continue
        out.append(RankCandidates(TokenSet._from_sorted_unique(local), tuple(ids)))
    return out


__all__ = [
    "HeadShard",
    "RankCandidates",
    "restrict_to_rank",
]
