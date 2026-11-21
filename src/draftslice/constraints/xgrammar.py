"""xgrammar constraint adapter.

The adapter is optional and lazy: the core package never imports xgrammar or
transformers. A session owns a ``GrammarMatcher`` and converts the next-token
bitmask into a :class:`TokenSet`. State ids are local to the session; stable
cross-request ids require a backend with a serializable state hash.
"""

from __future__ import annotations

from typing import Any

from ..plan.types import TokenSet
from .base import ConstraintSession, ConstraintState


class XGrammarError(RuntimeError):
    """Raised when the xgrammar backend cannot build or advance a matcher."""


def _require_xgrammar() -> Any:
    try:
        import xgrammar
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise XGrammarError("xgrammar is not installed") from exc
    return xgrammar


def bitmask_to_token_set(bitmask: Any, vocab_size: int) -> TokenSet:
    """Decode an xgrammar int32 bitmask row into a sorted TokenSet."""

    if vocab_size <= 0:
        raise XGrammarError("vocab_size must be positive")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise XGrammarError("torch is required for bitmask decoding") from exc
    if not isinstance(bitmask, torch.Tensor):
        bitmask = torch.as_tensor(bitmask)
    flat = bitmask.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    values = flat.to(torch.int64)
    tokens: list[int] = []
    for bit in range(32):
        selected = torch.nonzero((values >> bit) & 1, as_tuple=False).flatten()
        if selected.numel():
            tokens.extend((selected * 32 + bit).tolist())
    return TokenSet(token for token in tokens if token < vocab_size)


class XGrammarSession(ConstraintSession):
    """One grammar cursor backed by ``xgrammar.GrammarMatcher``."""

    def __init__(
        self,
        matcher: Any,
        tokenizer_info: Any,
        *,
        state_id: int = 0,
        vocab_size: int | None = None,
    ) -> None:
        if state_id < 0:
            raise XGrammarError("state_id must be non-negative")
        self.matcher = matcher
        self.tokenizer_info = tokenizer_info
        self.vocab_size = int(vocab_size or tokenizer_info.vocab_size)
        self._state_id = state_id
        self._state = self._read_state()

    def _bitmask(self) -> Any:
        xgrammar = _require_xgrammar()
        return xgrammar.allocate_token_bitmask(1, self.vocab_size)

    def _read_state(self) -> ConstraintState:
        if self.matcher.is_terminated():
            return ConstraintState(
                state_id=self._state_id,
                valid_ids=TokenSet(self.matcher.stop_token_ids or ()),
                final=True,
            )
        xgrammar = _require_xgrammar()
        bitmask = self._bitmask()
        xgrammar.reset_token_bitmask(bitmask)
        self.matcher.fill_next_token_bitmask(bitmask, 0)
        valid = bitmask_to_token_set(bitmask, self.vocab_size)
        return ConstraintState(
            state_id=self._state_id,
            valid_ids=valid,
            final=self.matcher.is_completed(),
        )

    def current(self) -> ConstraintState:
        return self._state

    def valid_ids(self) -> TokenSet:
        return self._state.valid_ids

    def advance(self, token_id: int) -> None:
        if not self.matcher.accept_token(int(token_id)):
            raise XGrammarError(f"token {token_id} is not accepted by the grammar")
        self._state_id += 1
        self._state = self._read_state()

    def fork(self) -> XGrammarSession:
        return XGrammarSession(
            self.matcher.fork(),
            self.tokenizer_info,
            state_id=self._state_id,
            vocab_size=self.vocab_size,
        )

    def is_final(self) -> bool:
        return self.matcher.is_completed()

    @classmethod
    def from_json_schema(
        cls,
        schema: dict[str, Any] | str,
        *,
        tokenizer: str | Any,
        vocab_size: int | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> XGrammarSession:
        xgrammar = _require_xgrammar()
        if isinstance(tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise XGrammarError("transformers is required for a tokenizer path") from exc
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        info = xgrammar.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size or len(tokenizer),
            stop_token_ids=stop_token_ids,
        )
        compiler = xgrammar.GrammarCompiler(info)
        grammar = xgrammar.Grammar.from_json_schema(schema)
        matcher = xgrammar.GrammarMatcher(compiler.compile_grammar(grammar))
        return cls(matcher, info, vocab_size=info.vocab_size)

    @classmethod
    def from_regex(
        cls,
        pattern: str,
        *,
        tokenizer: str | Any,
        vocab_size: int | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> XGrammarSession:
        xgrammar = _require_xgrammar()
        if isinstance(tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise XGrammarError("transformers is required for a tokenizer path") from exc
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        info = xgrammar.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=vocab_size or len(tokenizer),
            stop_token_ids=stop_token_ids,
        )
        compiler = xgrammar.GrammarCompiler(info)
        grammar = xgrammar.Grammar.from_regex(pattern)
        matcher = xgrammar.GrammarMatcher(compiler.compile_grammar(grammar))
        return cls(matcher, info, vocab_size=info.vocab_size)


def record_trajectory(
    session: XGrammarSession,
    token_ids: list[int] | None = None,
    *,
    max_tokens: int = 64,
) -> list[dict[str, Any]]:
    """Record the valid set before each accepted token.

    If ``token_ids`` is supplied, those tokens are accepted. Otherwise the first
    valid token is chosen at each step, which gives a deterministic replay.
    """

    records: list[dict[str, Any]] = []
    limit = len(token_ids) if token_ids is not None else max_tokens
    for index in range(limit):
        state = session.current()
        records.append(
            {
                "state_id": state.state_id,
                "valid_ids": list(state.valid_ids.ids),
                "final": state.final,
            }
        )
        if state.final or not state.valid_ids:
            break
        if token_ids is not None:
            token = int(token_ids[index])
        else:
            token = state.valid_ids.ids[0]
        session.advance(token)
    return records


__all__ = [
    "XGrammarError",
    "XGrammarSession",
    "bitmask_to_token_set",
    "record_trajectory",
]
