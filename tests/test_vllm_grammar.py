from __future__ import annotations

import numpy as np
import pytest

from draftslice.integrations.vllm_grammar import (
    GrammarBridgeError,
    GrammarSnapshot,
    capture_grammar_snapshot,
    clear_valid_set_cache,
    count_valid_ids,
    current_grammar_snapshot,
    unpack_valid_ids,
)
from draftslice.plan import TokenSet


def row_with_bits(bits: set[int], words: int) -> np.ndarray:
    row = np.zeros(words, dtype=np.int32)
    for bit in bits:
        row[bit // 32] |= 1 << (bit % 32)
    return row


def test_unpack_valid_ids() -> None:
    row = row_with_bits({1, 3, 33, 65}, 4)
    assert unpack_valid_ids(row, 128).ids == (1, 3, 33, 65)
    assert unpack_valid_ids(row, 10).ids == (1, 3)
    assert unpack_valid_ids(np.zeros(4, dtype=np.int32), 128).ids == ()


def test_count_valid_ids_handles_vocab_padding() -> None:
    row = row_with_bits({1, 3, 33, 65}, 4)
    assert count_valid_ids(row, 128) == 4
    # Bit 65 and above are outside a 64-token vocabulary.
    assert count_valid_ids(row, 64) == 3
    assert count_valid_ids(np.zeros(4, dtype=np.int32), 128) == 0
    assert count_valid_ids(np.array([0xFFFFFFFF], dtype=np.uint32), 32) == 32


def test_valid_ids_cache_returns_shared_object() -> None:
    clear_valid_set_cache()
    row = row_with_bits({1, 2, 3}, 2)
    snapshot = GrammarSnapshot(
        req_ids=("a",),
        structured_req_ids=("a",),
        bitmask=np.stack([row]),
        mask_stride=1,
        vocab_size=64,
    )
    first = snapshot.valid_ids("a")
    second = snapshot.valid_ids("a")
    assert first == TokenSet([1, 2, 3])
    assert first is second


def test_max_ids_returns_full_fallback() -> None:
    snapshot = GrammarSnapshot(
        req_ids=("a",),
        structured_req_ids=("a",),
        bitmask=np.stack([row_with_bits({1, 2, 3}, 2)]),
        mask_stride=1,
        vocab_size=64,
    )
    assert snapshot.valid_ids("a", max_ids=2) is None
    assert snapshot.valid_ids("a", max_ids=3) == TokenSet([1, 2, 3])


def test_grammar_snapshot_mapping() -> None:
    bitmask = np.stack(
        [
            row_with_bits({1, 2}, 2),
            row_with_bits({3}, 2),
            row_with_bits({5}, 2),
        ]
    )
    snapshot = GrammarSnapshot(
        req_ids=("a", "b", "c"),
        structured_req_ids=("a", "c"),
        bitmask=bitmask,
        mask_stride=1,
        vocab_size=64,
    )
    assert snapshot.valid_ids("a") == TokenSet([1, 2])
    assert snapshot.valid_ids("b") is None
    assert snapshot.valid_ids("c") == TokenSet([3])


def test_grammar_snapshot_positions() -> None:
    bitmask = np.stack(
        [
            row_with_bits({1}, 2),
            row_with_bits({2}, 2),
        ]
    )
    snapshot = GrammarSnapshot(
        req_ids=("a",),
        structured_req_ids=("a",),
        bitmask=bitmask,
        mask_stride=2,
        vocab_size=64,
    )
    assert snapshot.valid_ids("a", 0) == TokenSet([1])
    assert snapshot.valid_ids("a", 1) == TokenSet([2])
    with pytest.raises(GrammarBridgeError):
        snapshot.valid_ids("a", 2)


def test_grammar_snapshot_validation() -> None:
    with pytest.raises(GrammarBridgeError):
        GrammarSnapshot(
            req_ids=("a",),
            structured_req_ids=("a",),
            bitmask=np.zeros((0, 1), dtype=np.int32),
            mask_stride=1,
            vocab_size=64,
        )


def test_capture_snapshot() -> None:
    snapshot = GrammarSnapshot(
        req_ids=("a",),
        structured_req_ids=("a",),
        bitmask=np.stack([row_with_bits({7}, 1)]),
        mask_stride=1,
        vocab_size=32,
    )
    capture_grammar_snapshot(snapshot)
    assert current_grammar_snapshot() is snapshot
    capture_grammar_snapshot(None)
    assert current_grammar_snapshot() is None
