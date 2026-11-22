from __future__ import annotations

import numpy as np
import torch

from draftslice.integrations.vllm_shadow import (
    GrammarShadowRegistry,
    _clear_state_cache,
    _mix_key,
    _ShadowEntry,
)
from draftslice.plan import TokenSet


class FakeConfig:
    class ModelConfig:
        def get_vocab_size(self) -> int:
            return 64

    class StructuredOutputsConfig:
        reasoning_parser = ""

    model_config = ModelConfig()
    structured_outputs_config = StructuredOutputsConfig()


class FakeGrammar:
    def __init__(self, allowed: set[int], terminated: bool = False) -> None:
        self.allowed = allowed
        self.terminated = terminated
        self.filled = 0

    def is_terminated(self) -> bool:
        return self.terminated

    def fill_bitmask(self, bitmask, idx: int) -> None:
        self.filled += 1
        bitmask[idx].zero_()
        for token in self.allowed:
            bitmask[idx, token // 32] |= 1 << (token % 32)

    def accept_tokens(self, req_id: str, tokens: list[int]) -> bool:
        self.accepted = list(tokens)
        return True


def make_registry() -> GrammarShadowRegistry:
    registry = GrammarShadowRegistry(FakeConfig(), max_ids=32)
    registry._bitmask = torch.zeros((1, 2), dtype=torch.int32)
    return registry


class FakeReasoningEnd:
    def __init__(self, offset: int, implicit: bool = False) -> None:
        self.offset = offset
        self.implicit = implicit


class FakeReasoner:
    reasoning_end_token_ids = [13]

    def __init__(self) -> None:
        self.calls = 0

    def find_reasoning_end(self, tokens):
        self.calls += 1
        if 13 in tokens:
            index = tokens.index(13)
            return FakeReasoningEnd(index)
        return FakeReasoningEnd(len(tokens))


def test_reasoning_parser_no_longer_disables_the_shadow() -> None:
    class Config(FakeConfig):
        class StructuredOutputsConfig(FakeConfig.StructuredOutputsConfig):
            reasoning_parser = "deepseek_r1"

    registry = GrammarShadowRegistry(Config(), max_ids=32)
    assert registry.enabled


def test_reasoning_boundary_gates_the_grammar() -> None:
    registry = make_registry()
    grammar = FakeGrammar({1, 2})
    reasoner = FakeReasoner()
    entry = _ShadowEntry(
        req_id="a",
        grammar=grammar,
        state_key=b"a",
        reasoner=reasoner,
        grammar_active=False,
    )
    registry.entries[0] = entry
    # Before the end token the grammar is not exposed at all.
    assert registry.valid_ids(0) is None
    assert registry.row_mask(0, vocab_size=64) is None
    assert registry._advance_reasoning(entry, [10, 11]) == []
    assert not entry.grammar_active
    # The explicit end token is skipped; only the following token is grammar.
    assert registry._advance_reasoning(entry, [13, 1]) == [1]
    assert entry.grammar_active
    assert registry.stats["reasoning_activated"] == 1
    assert registry.valid_ids(0) == TokenSet([1, 2])


def test_implicit_reasoning_end_token_is_grammar_content() -> None:
    class Implicit(FakeReasoner):
        def find_reasoning_end(self, tokens):
            if 13 in tokens:
                return FakeReasoningEnd(tokens.index(13), implicit=True)
            return FakeReasoningEnd(len(tokens))

    registry = make_registry()
    entry = _ShadowEntry(
        req_id="a",
        grammar=FakeGrammar({1}),
        state_key=b"a",
        reasoner=Implicit(),
        grammar_active=False,
    )
    assert registry._advance_reasoning(entry, [13, 1]) == [13, 1]


def test_mix_key_is_prefix_and_token_sensitive() -> None:
    base = b"schema"
    assert _mix_key(base, [1, 2]) == _mix_key(base, [1, 2])
    assert _mix_key(base, [1, 2]) != _mix_key(base, [2, 1])
    assert _mix_key(base, [1]) != _mix_key(base, [1, 2])


def test_row_mask_unpacks_bitmask_without_tokenset() -> None:
    registry = make_registry()
    grammar = FakeGrammar({0, 1, 33, 63})
    registry.entries[0] = _ShadowEntry(req_id="a", grammar=grammar, state_key=b"a")
    mask = registry.row_mask(0, vocab_size=64)
    assert mask is not None
    expected = torch.zeros(64, dtype=torch.bool)
    expected[[0, 1, 33, 63]] = True
    assert torch.equal(mask, expected)
    assert grammar.filled == 1


def test_row_mask_falls_back_for_terminated_or_missing_step() -> None:
    registry = make_registry()
    registry.entries[0] = _ShadowEntry(
        req_id="a", grammar=FakeGrammar({1}, terminated=True), state_key=b"a"
    )
    assert registry.row_mask(0, vocab_size=64) is None
    registry.entries[1] = _ShadowEntry(
        req_id="b", grammar=FakeGrammar({1}), state_key=b"b"
    )
    assert registry.row_mask(1, step=2, vocab_size=64) is None


def test_valid_ids_caches_by_state_key() -> None:
    _clear_state_cache()
    registry = make_registry()
    grammar = FakeGrammar({1, 3})
    registry.entries[0] = _ShadowEntry(req_id="a", grammar=grammar, state_key=b"a")
    first = registry.valid_ids(0)
    second = registry.valid_ids(0)
    assert first == TokenSet([1, 3])
    assert second is first
    assert grammar.filled == 1
    assert registry.stats["valid_state_hits"] == 1


def test_state_cache_does_not_leak_between_grammars() -> None:
    _clear_state_cache()
    registry = make_registry()
    grammar_a = FakeGrammar({1})
    grammar_b = FakeGrammar({2})
    registry.entries[0] = _ShadowEntry(req_id="a", grammar=grammar_a, state_key=b"tag-a")
    registry.entries[1] = _ShadowEntry(req_id="b", grammar=grammar_b, state_key=b"tag-b")
    assert registry.valid_ids(0) == TokenSet([1])
    assert registry.valid_ids(1) == TokenSet([2])
    assert grammar_a.filled == 1
    assert grammar_b.filled == 1


def test_too_wide_and_terminated_return_none() -> None:
    _clear_state_cache()
    registry = make_registry()
    registry.entries[0] = _ShadowEntry(
        req_id="a", grammar=FakeGrammar(set(range(40))), state_key=b"wide"
    )
    registry.entries[1] = _ShadowEntry(
        req_id="b", grammar=FakeGrammar(set(), terminated=True), state_key=b"done"
    )
    assert registry.valid_ids(0) is None
    assert registry.valid_ids(1) is None
    assert registry.stats["valid_too_wide"] == 1
    assert registry.stats["valid_terminated"] == 1


def test_advance_sampled_mixes_tokens_and_clears_steps() -> None:
    registry = make_registry()
    grammar = FakeGrammar({1})
    entry = _ShadowEntry(req_id="a", grammar=grammar, state_key=b"tag")
    entry.step_grammars[1] = object()
    entry.step_keys[1] = b"step"
    registry.entries[0] = entry
    before = entry.state_key
    registry.advance_sampled(
        np.array([0]), np.array([[5, -1]]), np.array([1])
    )
    assert entry.state_key != before
    assert entry.step_grammars == {}
    assert entry.step_keys == {}


def test_drop_removes_entry() -> None:
    registry = make_registry()
    registry.entries[0] = _ShadowEntry(req_id="a", grammar=FakeGrammar({1}))
    registry.drop({"a"})
    assert not registry.has(0)
    assert registry.stats["dropped"] == 1


def test_registry_is_noop_without_structured_output() -> None:
    registry = make_registry()
    registry.register(0, None)
    assert not registry.has(0)
    assert registry.stats["registered"] == 0
