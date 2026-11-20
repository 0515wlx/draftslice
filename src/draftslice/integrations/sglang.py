"""SGLang integration boundary (planned for M3)."""

from __future__ import annotations


def register() -> None:
    """Register DraftSlice with SGLang.

    The integration will attach to the speculative-decoding draft path and the
    draft LM head. It is intentionally not implemented in the skeleton.
    """
    raise NotImplementedError("SGLang integration lands in M3.")
