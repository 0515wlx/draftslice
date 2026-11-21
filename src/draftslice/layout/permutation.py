"""Vocabulary row layouts.

A layout is a bijection between native token ids and storage rows. Native layout
keeps ``row == token``. A grouped layout stores the rows of each vocabulary group
contiguously, which can turn a candidate set into a small number of row ranges.
Whether that actually helps is a kernel measurement, not an assumption.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from ..plan.types import LayoutKind, TokenSet


class LayoutError(ValueError):
    """Raised when a layout is malformed."""


@dataclass(frozen=True, slots=True)
class PackedRows:
    """Storage rows for a token set.

    ``rows`` is sorted by storage row. ``tokens_by_row[i]`` is the native token id
    stored at ``rows[i]``; use it to map kernel output back to token ids.
    ``token_ids`` is the logical set and is sorted independently.
    """

    token_ids: TokenSet
    rows: tuple[int, ...]
    tokens_by_row: tuple[int, ...]
    layout: LayoutKind

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.rows):
            raise LayoutError("packed rows and token ids must have the same length")
        if len(self.tokens_by_row) != len(self.rows):
            raise LayoutError("tokens_by_row must align with rows")
        if set(self.tokens_by_row) != set(self.token_ids.ids):
            raise LayoutError("tokens_by_row must match the logical token set")
        if any(row < 0 for row in self.rows):
            raise LayoutError("rows must be non-negative")


@dataclass(frozen=True, slots=True)
class VocabularyLayout:
    """A bijection between token ids and storage rows."""

    token_to_row: tuple[int, ...]
    row_to_token: tuple[int, ...]
    kind: LayoutKind = "native"
    group_of_token: tuple[int, ...] | None = None
    group_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.token_to_row) != len(self.row_to_token):
            raise LayoutError("token_to_row and row_to_token must have equal length")
        if len(set(self.row_to_token)) != len(self.row_to_token):
            raise LayoutError("row_to_token must be a bijection")
        if sorted(self.row_to_token) != list(range(len(self.row_to_token))):
            raise LayoutError("row_to_token must contain every row exactly once")
        for token_id, row in enumerate(self.token_to_row):
            if row < 0 or row >= len(self.row_to_token):
                raise LayoutError(f"token {token_id} has invalid row {row}")
            if self.row_to_token[row] != token_id:
                raise LayoutError(f"row {row} does not map back to token {token_id}")
        if self.group_of_token is not None:
            if len(self.group_of_token) != len(self.token_to_row):
                raise LayoutError("group_of_token must cover every token")
            if self.group_offsets:
                if len(self.group_offsets) < 2:
                    raise LayoutError("group_offsets must contain at least two entries")
                if self.group_offsets[0] != 0:
                    raise LayoutError("group_offsets must start at zero")
                if self.group_offsets[-1] != len(self.row_to_token):
                    raise LayoutError("group_offsets must end at the vocabulary size")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_row)

    @classmethod
    def native(cls, vocab_size: int) -> VocabularyLayout:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rows = tuple(range(vocab_size))
        return cls(token_to_row=rows, row_to_token=rows, kind="native")

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[Sequence[int]],
        *,
        kind: Literal["leaf_contiguous", "packed"] = "leaf_contiguous",
    ) -> VocabularyLayout:
        if not groups:
            raise LayoutError("layout needs at least one group")
        seen: set[int] = set()
        order: list[int] = []
        group_of: list[int] = []
        offsets = [0]
        for group_id, group in enumerate(groups):
            if not group:
                raise LayoutError(f"group {group_id} is empty")
            for token_id in group:
                if not isinstance(token_id, int) or isinstance(token_id, bool):
                    raise LayoutError("token ids must be int")
                if token_id < 0:
                    raise LayoutError(f"negative token id: {token_id}")
                if token_id in seen:
                    raise LayoutError(f"duplicate token id: {token_id}")
                seen.add(token_id)
                order.append(token_id)
                group_of.append(group_id)
            offsets.append(len(order))
        if seen != set(range(len(order))):
            missing = sorted(set(range(len(order))) - seen)
            raise LayoutError(f"groups do not cover the vocabulary; missing {missing[:8]}")

        row_to_token = tuple(order)
        token_to_row = [0] * len(order)
        for row, token_id in enumerate(row_to_token):
            token_to_row[token_id] = row
        return cls(
            token_to_row=tuple(token_to_row),
            row_to_token=row_to_token,
            kind=kind,
            group_of_token=tuple(group_of),
            group_offsets=tuple(offsets),
        )

    def row_of(self, token_id: int) -> int:
        if token_id < 0:
            raise LayoutError(f"token {token_id} is outside the layout")
        try:
            return self.token_to_row[token_id]
        except IndexError as exc:
            raise LayoutError(f"token {token_id} is outside the layout") from exc

    def token_of(self, row: int) -> int:
        if row < 0:
            raise LayoutError(f"row {row} is outside the layout")
        try:
            return self.row_to_token[row]
        except IndexError as exc:
            raise LayoutError(f"row {row} is outside the layout") from exc

    def pack(self, token_ids: Iterable[int]) -> PackedRows:
        tokens = TokenSet(token_ids)
        for token_id in tokens:
            if token_id >= self.vocab_size:
                raise LayoutError(f"token {token_id} is outside the layout")
        rows = tuple(sorted(self.row_of(token_id) for token_id in tokens))
        tokens_by_row = tuple(self.token_of(row) for row in rows)
        return PackedRows(
            token_ids=tokens,
            rows=rows,
            tokens_by_row=tokens_by_row,
            layout=self.kind,
        )

    def row_ranges(self, token_ids: Iterable[int]) -> tuple[tuple[int, int], ...]:
        """Return contiguous row ranges as ``(start, end)`` with exclusive end."""

        packed = self.pack(token_ids)
        if not packed.rows:
            return ()
        ranges: list[tuple[int, int]] = []
        start = previous = packed.rows[0]
        for row in packed.rows[1:]:
            if row == previous + 1:
                previous = row
                continue
            ranges.append((start, previous + 1))
            start = previous = row
        ranges.append((start, previous + 1))
        return tuple(ranges)

    def group_rows(self, group_id: int) -> tuple[int, int]:
        if not self.group_offsets:
            raise LayoutError("layout has no group metadata")
        if group_id < 0 or group_id + 1 >= len(self.group_offsets):
            raise LayoutError(f"group {group_id} is outside the layout")
        return self.group_offsets[group_id], self.group_offsets[group_id + 1]


__all__ = ["LayoutError", "PackedRows", "VocabularyLayout"]
