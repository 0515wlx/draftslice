from __future__ import annotations

from types import SimpleNamespace

import pytest

from draftslice.integrations.head_shard import (
    HeadShard,
    restrict_to_rank,
)
from draftslice.plan import TokenSet


def _head(tp_rank: int, *, tp_size: int = 2, org_vocab: int = 16) -> SimpleNamespace:
    per_partition = org_vocab // tp_size
    start = tp_rank * per_partition
    return SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        org_vocab_size=org_vocab,
        num_embeddings_padded=org_vocab,
        num_embeddings_per_partition=per_partition,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=start + per_partition,
            added_vocab_start_index=org_vocab,
            added_vocab_end_index=org_vocab,
            num_org_elements_padded=per_partition,
        ),
    )


def test_detect_rank_shard_and_map_ids() -> None:
    rank0 = HeadShard.from_lm_head(_head(0), weight_rows=8, vocab_size=16)
    rank1 = HeadShard.from_lm_head(_head(1), weight_rows=8, vocab_size=16)
    assert rank0 is not None and rank1 is not None
    assert (rank0.tp_size, rank0.tp_rank) == (2, 0)
    assert (rank1.tp_size, rank1.tp_rank) == (2, 1)
    assert rank0.local_row(3) == 3
    assert rank0.local_row(9) is None
    assert rank1.local_row(9) == 1
    assert rank1.local_row(3) is None
    assert rank1.global_id(1) == 9
    assert rank0.to_dict()["shard_rows"] == 8


def test_unsharded_head_is_trivial_shard() -> None:
    shard = HeadShard.from_lm_head(
        SimpleNamespace(tp_size=1), weight_rows=16, vocab_size=16
    )
    assert shard is not None
    assert shard.tp_size == 1
    assert shard.local_row(7) == 7
    assert HeadShard.from_lm_head(
        SimpleNamespace(tp_size=1), weight_rows=8, vocab_size=16
    ) is None


def test_quant_config_alone_does_not_reject_a_bf16_head() -> None:
    # A module can carry a quant config while its LM-head weight stays bf16.
    # Shard detection is layout-only; dtype is checked by the adapter.
    head = _head(0)
    head.quant_config = object()
    assert HeadShard.from_lm_head(head, weight_rows=8, vocab_size=16) is not None


def test_inconsistent_head_is_rejected() -> None:
    head = _head(0)
    head.num_embeddings_per_partition = 7
    assert HeadShard.from_lm_head(head, weight_rows=8, vocab_size=16) is None


def test_restrict_to_rank_keeps_global_ids_aligned() -> None:
    rank0 = HeadShard.from_lm_head(_head(0), weight_rows=8, vocab_size=16)
    rank1 = HeadShard.from_lm_head(_head(1), weight_rows=8, vocab_size=16)
    assert rank0 is not None and rank1 is not None
    valid_sets = [
        TokenSet([1, 3, 9, 14]),
        TokenSet([2]),
        None,
        TokenSet([10, 11]),
    ]
    local0 = restrict_to_rank(valid_sets, rank0, max_rows=64)
    local1 = restrict_to_rank(valid_sets, rank1, max_rows=64)
    assert local0[0] is not None and local0[0].local == TokenSet([1, 3])
    assert local0[0].global_ids == (1, 3)
    assert local1[0] is not None and local1[0].local == TokenSet([1, 6])
    assert local1[0].global_ids == (9, 14)
    # A rank with no owned token must fall back (and get masked to V upstream).
    assert local1[1] is None
    assert local0[2] is None and local1[2] is None
    assert local0[3] is None
    assert local1[3] is not None and local1[3].global_ids == (10, 11)


def test_full_head_decision_is_rank_independent() -> None:
    """All ranks derive the same fallback rows, so TP collectives stay balanced."""

    rank0 = HeadShard.from_lm_head(_head(0), weight_rows=8, vocab_size=16)
    rank1 = HeadShard.from_lm_head(_head(1), weight_rows=8, vocab_size=16)
    assert rank0 is not None and rank1 is not None
    assert rank0.rank_ranges() == rank1.rank_ranges() == [(0, 8), (8, 16)]
    for token_set in (
        TokenSet([1, 3, 9, 14]),  # every rank owns something
        TokenSet([2]),  # only rank 0
        TokenSet([10, 11]),  # only rank 1
    ):
        assert rank0.local_counts(token_set) == rank1.local_counts(token_set)
        assert rank0.local_counts(token_set) == [
            sum(0 <= t < 8 for t in token_set.ids),
            sum(8 <= t < 16 for t in token_set.ids),
        ]


def test_restrict_to_rank_respects_max_rows() -> None:
    shard = HeadShard.unsharded(16, 16)
    assert shard is not None
    assert restrict_to_rank([TokenSet([1, 2, 3])], shard, max_rows=2) == [None]


def test_added_vocab_rows_follow_the_base_slice() -> None:
    head = _head(0, org_vocab=16)
    head.num_embeddings_padded = 24
    head.num_embeddings_per_partition = 12
    head.shard_indices.added_vocab_start_index = 16
    head.shard_indices.added_vocab_end_index = 20
    head.shard_indices.num_org_elements_padded = 8
    shard = HeadShard.from_lm_head(head, weight_rows=12, vocab_size=20)
    assert shard is not None
    assert shard.local_row(17) == 9
    assert shard.global_id(9) == 17
    with pytest.raises(ValueError):
        restrict_to_rank([TokenSet([1])], shard, max_rows=0)
