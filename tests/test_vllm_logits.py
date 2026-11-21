from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.vllm_logits import (  # noqa: E402
    VllmLogitsError,
    build_draft_logits,
    candidate_bucket,
)
from draftslice.kernels.torch import TorchGatherKernel  # noqa: E402
from draftslice.plan import TokenSet  # noqa: E402


def test_sliced_logits_match_full_head() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(4, 8)
    weight = torch.randn(16, 8)
    valid_sets = [
        TokenSet([1, 3, 5]),
        TokenSet([2]),
        None,
        TokenSet(),
    ]
    logits, stats = build_draft_logits(
        hidden,
        weight,
        valid_sets,
        vocab_size=16,
        full_logits=lambda rows: rows @ weight.T,
        kernel=TorchGatherKernel(),
    )
    full = hidden @ weight.T
    for row in (0, 1):
        valid = valid_sets[row]
        expected = torch.full((16,), float("-inf"))
        expected[list(valid.ids)] = full[row, list(valid.ids)]
        torch.testing.assert_close(logits[row], expected)
    for row in (2, 3):
        torch.testing.assert_close(logits[row], full[row])
    assert stats.rows == 4
    assert stats.sliced_rows == 2
    assert stats.full_rows == 2
    assert stats.distinct_sets == 2
    assert stats.candidate_total == (3 + 1) * 1


def test_max_slice_rows_falls_back() -> None:
    hidden = torch.randn(2, 4)
    weight = torch.randn(8, 4)
    logits, stats = build_draft_logits(
        hidden,
        weight,
        [TokenSet([1, 2, 3]), TokenSet([1])],
        vocab_size=8,
        full_logits=lambda rows: rows @ weight.T,
        max_slice_rows=2,
        kernel=TorchGatherKernel(),
    )
    assert stats.sliced_rows == 1
    assert stats.full_rows == 1
    torch.testing.assert_close(logits[0], hidden[0] @ weight.T)


def test_postprocess_matches_full_path() -> None:
    hidden = torch.randn(2, 4)
    weight = torch.randn(8, 4)
    valid_sets = [TokenSet([1, 2]), TokenSet([3])]

    def full(rows):
        values = rows @ weight.T
        return torch.tanh(values / 2.0) * 2.0

    logits, _ = build_draft_logits(
        hidden,
        weight,
        valid_sets,
        vocab_size=8,
        full_logits=full,
        scale=1.0,
        soft_cap=2.0,
        kernel=TorchGatherKernel(),
    )
    expected = full(hidden)
    for row, valid in enumerate(valid_sets):
        torch.testing.assert_close(logits[row, list(valid.ids)], expected[row, list(valid.ids)])
        assert torch.isneginf(logits[row, 0])


class FakeFusedKernel:
    """Torch stand-in for the Triton fused row kernel (CPU-testable)."""

    name = "fake_fused"
    supports_rows_into = True

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def run_rows_into(
        self, hidden, weight, tokens, out_rows, out, *, width=None, scale=1.0, soft_cap=None
    ):
        rows = out_rows.numel()
        matrix = tokens.reshape(rows, width) if tokens.ndim == 1 else tokens
        for position in range(rows):
            row = int(out_rows[position])
            valid = matrix[position][matrix[position] >= 0]
            ids = valid.long()
            values = hidden[row].float() @ weight[ids].float().T
            if soft_cap is not None:
                values = torch.tanh(values / soft_cap) * soft_cap
            if scale != 1.0:
                values = values * scale
            out[row, ids] = values
        self.calls.append(tuple(matrix.shape))
        return out


def test_fused_rows_match_full_head_and_bucket_shapes() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(4, 8)
    weight = torch.randn(16, 8)
    # counts 3 and 4 share bucket 4; count 5 needs bucket 8.
    valid_sets = [TokenSet([1, 3, 5]), TokenSet([2, 4, 6, 7]), TokenSet([8, 9, 10, 11, 12]), None]
    kernel = FakeFusedKernel()
    logits, stats = build_draft_logits(
        hidden,
        weight,
        valid_sets,
        vocab_size=16,
        full_logits=lambda rows: rows @ weight.T,
        kernel=kernel,
    )
    full = hidden @ weight.T
    for row, valid in enumerate(valid_sets):
        if valid is None:
            torch.testing.assert_close(logits[row], full[row])
            continue
        expected = torch.full((16,), float("-inf"))
        expected[list(valid.ids)] = full[row, list(valid.ids)]
        torch.testing.assert_close(logits[row], expected)
    assert stats.sliced_rows == 3
    assert stats.full_rows == 1
    assert stats.candidate_total == 3 + 4 + 5
    assert stats.distinct_sets == 3
    # One launch per width bucket, with padding written as -1.
    assert kernel.calls == [(2, 4), (1, 8)]


def test_fused_rows_apply_logit_postprocess() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 4)
    weight = torch.randn(8, 4)
    kernel = FakeFusedKernel()
    logits, _ = build_draft_logits(
        hidden,
        weight,
        [TokenSet([1, 2]), TokenSet([3])],
        vocab_size=8,
        full_logits=lambda rows: rows @ weight.T,
        kernel=kernel,
        scale=0.5,
        soft_cap=2.0,
    )
    reference = torch.tanh(hidden @ weight.T / 2.0) * 2.0 * 0.5
    for row, valid in enumerate([TokenSet([1, 2]), TokenSet([3])]):
        ids = list(valid.ids)
        torch.testing.assert_close(logits[row, ids], reference[row, ids])
        assert torch.isneginf(logits[row, 0])


def test_candidate_bucket_rounds_up_and_caps() -> None:
    assert candidate_bucket(0) == 0
    assert candidate_bucket(1) == 1
    assert candidate_bucket(3) == 4
    assert candidate_bucket(5) == 8
    assert candidate_bucket(166) == 256
    assert candidate_bucket(166, max_count=128) == 128
    assert candidate_bucket(200, max_count=32768) == 256


def test_validation_errors() -> None:
    hidden = torch.randn(2, 4)
    weight = torch.randn(8, 4)
    with pytest.raises(VllmLogitsError):
        build_draft_logits(
            hidden,
            weight,
            [TokenSet([1])],
            vocab_size=8,
            full_logits=lambda rows: rows @ weight.T,
        )
    with pytest.raises(VllmLogitsError):
        build_draft_logits(
            hidden,
            weight[:4],
            [TokenSet([1]), TokenSet([2])],
            vocab_size=8,
            full_logits=lambda rows: rows @ weight[:4].T,
        )
