from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.constraints.xgrammar import bitmask_to_token_set  # noqa: E402


def test_bitmask_decoding_respects_vocab_and_bit_order() -> None:
    bitmask = torch.zeros((1, 2), dtype=torch.int32)
    # word 0: bits 1 and 31; word 1: bit 1
    bitmask[0, 0] = -2_147_483_646  # bits 1 and 31 as signed int32
    bitmask[0, 1] = 1 << 1
    tokens = bitmask_to_token_set(bitmask, vocab_size=64)
    assert tokens.ids == (1, 31, 33)

    assert bitmask_to_token_set(bitmask, vocab_size=32).ids == (1, 31)
    assert bitmask_to_token_set(torch.zeros(2, dtype=torch.int32), 64).is_empty
