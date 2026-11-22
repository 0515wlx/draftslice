from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations import vllm_fastmask  # noqa: E402
from draftslice.integrations.vllm_fastmask import apply_packed_mask  # noqa: E402


def test_packed_mask_torch_fallback(monkeypatch) -> None:
    # Force the portable path even when Triton is installed.
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setattr(vllm_fastmask, "_KERNEL", None)
    vocab, steps, groups = 64, 2, 2
    logits = torch.zeros(groups * steps, vocab)
    packed = torch.zeros((groups, (vocab + 31) // 32), dtype=torch.int32)
    packed[0, 0] = 0b11  # only tokens 0 and 1 are valid for group 0
    packed[1].fill_(-1)  # all-ones row: keep the full head
    rows = torch.tensor([0, 1], dtype=torch.int32)

    used_triton = apply_packed_mask(
        logits,
        base=0,
        row_stride=steps * vocab,
        packed=packed,
        rows=rows,
        vocab_size=vocab,
    )
    assert used_triton is False
    assert torch.isneginf(logits[0, 2:]).all()
    assert (logits[0, :2] == 0).all()
    assert (logits[steps] == 0).all()
