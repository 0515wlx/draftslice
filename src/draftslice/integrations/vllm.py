"""vLLM integration boundary (planned for M3)."""

from __future__ import annotations


def register() -> None:
    """Register DraftSlice with vLLM.

    The integration will attach to the speculative-decoding proposer and the
    draft LM head. It is intentionally not implemented in the skeleton.
    """
    raise NotImplementedError("vLLM integration lands in M3.")
