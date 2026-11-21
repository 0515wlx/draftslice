from __future__ import annotations

import pytest

from draftslice.layout import LayoutError, VocabularyLayout
from draftslice.plan import TokenSet


def test_native_layout_round_trip() -> None:
    layout = VocabularyLayout.native(5)
    assert layout.vocab_size == 5
    assert layout.row_of(3) == 3
    assert layout.token_of(3) == 3
    packed = layout.pack([3, 1, 3])
    assert packed.token_ids == TokenSet([1, 3])
    assert packed.rows == (1, 3)
    assert packed.tokens_by_row == (1, 3)
    assert layout.row_ranges([0, 1, 3]) == ((0, 2), (3, 4))


def test_grouped_layout_is_a_bijection() -> None:
    layout = VocabularyLayout.from_groups([[0, 2], [1, 3, 4]])
    assert layout.row_to_token == (0, 2, 1, 3, 4)
    assert layout.row_of(0) == 0
    assert layout.row_of(1) == 2
    assert layout.token_of(2) == 1
    assert layout.group_of_token == (0, 0, 1, 1, 1)
    assert layout.group_rows(0) == (0, 2)
    assert layout.group_rows(1) == (2, 5)


def test_grouped_packing_and_ranges() -> None:
    layout = VocabularyLayout.from_groups([[0, 2], [1, 3, 4]])
    packed = layout.pack([0, 1, 2])
    assert packed.rows == (0, 1, 2)
    assert packed.tokens_by_row == (0, 2, 1)
    assert layout.row_ranges([0, 1, 2]) == ((0, 3),)
    assert layout.row_ranges([0, 4]) == ((0, 1), (4, 5))
    assert layout.row_ranges([]) == ()


def test_layout_rejects_bad_groups() -> None:
    with pytest.raises(LayoutError):
        VocabularyLayout.from_groups([])
    with pytest.raises(LayoutError):
        VocabularyLayout.from_groups([[0, 1], [1, 2]])
    with pytest.raises(LayoutError):
        VocabularyLayout.from_groups([[0, 2]])
    with pytest.raises(LayoutError):
        VocabularyLayout.from_groups([[0, 1], []])


def test_layout_rejects_out_of_range_access() -> None:
    layout = VocabularyLayout.native(3)
    with pytest.raises(LayoutError):
        layout.row_of(-1)
    with pytest.raises(LayoutError):
        layout.row_of(3)
    with pytest.raises(LayoutError):
        layout.token_of(-1)
    with pytest.raises(LayoutError):
        layout.pack([4])
