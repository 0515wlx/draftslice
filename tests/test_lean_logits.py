"""Unit tests for the lean exact-candidate logits path (torch, no GPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.vllm_logits import (  # noqa: E402
    _LEAN_PAYLOADS,
    build_draft_logits_lean,
    clear_lean_payload_cache,
)
from draftslice.plan.types import TokenSet  # noqa: E402


class _FakeRowsKernel:
    """Minimal ``run_rows_into`` that writes candidate columns in torch."""

    supports_rows_into = True

    def run_rows_into(
        self,
        hidden,
        weight,
        tokens,
        out_rows,
        out,
        *,
        width=None,
        scale=1.0,
        soft_cap=None,
    ):
        for row, ids in zip(out_rows.tolist(), tokens.tolist(), strict=True):
            keep = [token for token in ids if token >= 0]
            values = hidden[row] @ weight[keep].T
            if soft_cap is not None:
                values = torch.tanh(values / soft_cap) * soft_cap
            if scale != 1.0:
                values = values * scale
            out[row, keep] = values.float()
        return out


def _reference(hidden, weight, valid_sets, vocab_size):
    out = torch.full((hidden.shape[0], vocab_size), float("-inf"))
    for row, token_set in enumerate(valid_sets):
        ids = list(range(vocab_size)) if token_set is None else list(token_set.ids)
        out[row, ids] = (hidden[row] @ weight[ids].T).float()
    return out


def test_lean_matches_full_and_caches_payloads():
    torch.manual_seed(0)
    clear_lean_payload_cache()
    hidden = torch.randn(2, 8)
    weight = torch.randn(64, 8)
    sliced = TokenSet((1, 3, 5, 7, 9))

    out, stats = build_draft_logits_lean(
        hidden,
        weight,
        [sliced, None],
        vocab_size=64,
        full_logits=lambda h: h @ weight.T,
        max_slice_rows=32,
        kernel=_FakeRowsKernel(),
    )
    assert stats.sliced_rows == 1
    assert stats.full_rows == 1
    assert torch.allclose(out, _reference(hidden, weight, [sliced, None], 64))
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert len(_LEAN_PAYLOADS) == 1

    # A second call with the same TokenSet object must reuse the payload.
    out2, _ = build_draft_logits_lean(
        hidden,
        weight,
        [sliced, None],
        vocab_size=64,
        full_logits=lambda h: h @ weight.T,
        max_slice_rows=32,
        kernel=_FakeRowsKernel(),
    )
    assert len(_LEAN_PAYLOADS) == 1
    assert torch.allclose(out2, out)


def test_lean_reuses_output_buffer():
    clear_lean_payload_cache()
    hidden = torch.randn(1, 4)
    weight = torch.randn(16, 4)
    token_set = TokenSet((2, 4, 6))
    buffer = torch.empty(1, 16)
    first, _ = build_draft_logits_lean(
        hidden,
        weight,
        [token_set],
        vocab_size=16,
        full_logits=lambda h: h @ weight.T,
        max_slice_rows=16,
        kernel=_FakeRowsKernel(),
        out=buffer,
    )
    assert first is buffer
    second, _ = build_draft_logits_lean(
        hidden,
        weight,
        [token_set],
        vocab_size=16,
        full_logits=lambda h: h @ weight.T,
        max_slice_rows=16,
        kernel=_FakeRowsKernel(),
        out=buffer,
    )
    assert second is buffer
    assert torch.allclose(second, first)
