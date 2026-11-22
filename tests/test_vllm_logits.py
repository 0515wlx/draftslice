from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.head_shard import HeadShard  # noqa: E402
from draftslice.integrations.vllm_logits import (  # noqa: E402
    VllmLogitsError,
    build_draft_logits,
    candidate_bucket,
    mask_logits_to_valid,
)
from draftslice.kernels.torch import TorchGatherKernel  # noqa: E402
from draftslice.plan import TokenSet  # noqa: E402


def test_mask_logits_to_valid_is_a_full_head_control() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [2.0, 2.1, 2.2, 2.3, 2.4],
        ]
    )
    original = logits.clone()
    out = mask_logits_to_valid(
        logits,
        [TokenSet([1, 3]), None, TokenSet()],
        vocab_size=5,
    )
    assert torch.isneginf(out[0, 0])
    assert torch.isneginf(out[0, 2])
    assert torch.isneginf(out[0, 4])
    torch.testing.assert_close(out[0, [1, 3]], original[0, [1, 3]])
    # A row without a set, and an empty set, keep the full-head values.
    torch.testing.assert_close(out[1], original[1])
    torch.testing.assert_close(out[2], original[2])
    # The helper must not mutate its input.
    torch.testing.assert_close(logits, original)


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


def test_deterministic_injection_skips_head() -> None:
    torch.manual_seed(3)
    hidden = torch.randn(3, 4)
    weight = torch.randn(8, 4)
    valid_sets = [TokenSet([2]), TokenSet([1, 3]), TokenSet([0])]
    calls: list[int] = []

    def full_logits(rows):
        calls.append(int(rows.shape[0]))
        return rows @ weight.T

    logits, stats = build_draft_logits(
        hidden,
        weight,
        valid_sets,
        vocab_size=8,
        full_logits=full_logits,
        kernel=TorchGatherKernel(),
        deterministic=True,
    )
    for row, token in ((0, 2), (2, 0)):
        assert float(logits[row, token]) == 0.0
        expected = torch.full((8,), float("-inf"))
        expected[token] = 0.0
        torch.testing.assert_close(logits[row], expected)
    assert float(logits[1, 1]) != float("-inf")
    assert stats.sliced_rows == 1
    assert stats.full_rows == 0
    assert calls == []


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


def _rank_shard(rank: int, vocab: int = 16) -> HeadShard:
    rows = vocab // 2
    return HeadShard(
        tp_size=2,
        tp_rank=rank,
        vocab_start=rank * rows,
        vocab_end=(rank + 1) * rows,
        shard_rows=rows,
        org_vocab_size=vocab,
        padded_vocab_size=vocab,
    )


def test_sharded_logits_combine_to_full_head_slice() -> None:
    """Two ranks slice their own shard and combine to the exact global result."""

    torch.manual_seed(7)
    hidden = torch.randn(4, 4)
    weight = torch.randn(16, 4)
    valid_sets = [
        TokenSet([1, 3, 9, 14]),
        TokenSet([2]),
        None,
        TokenSet([10, 11]),
    ]
    outs = []
    stats = []
    for rank in range(2):
        out, stat = build_draft_logits(
            hidden,
            weight[rank * 8 : (rank + 1) * 8],
            valid_sets,
            vocab_size=16,
            full_logits=lambda rows: rows @ weight.T,
            kernel=TorchGatherKernel(),
            shard=_rank_shard(rank),
        )
        outs.append(out)
        stats.append(stat)

    combined = torch.where(torch.isneginf(outs[0]), outs[1], outs[0])
    full = hidden @ weight.T
    for row, valid in enumerate(valid_sets):
        if valid is None:
            torch.testing.assert_close(combined[row], full[row])
            continue
        expected = torch.full((16,), float("-inf"))
        expected[list(valid.ids)] = full[row, list(valid.ids)]
        torch.testing.assert_close(combined[row], expected)

    # Rows 1 and 3 have all their tokens on a single rank. Both ranks take the
    # collective full-head fallback for them (masked to V), so the TP logits
    # all-gather stays balanced.
    assert stats[0].masked_full_rows == 2
    assert stats[1].masked_full_rows == 2
    row3 = outs[0][3]
    # The rank-local fallback cannot propose a grammar-invalid token.
    assert all(torch.isneginf(row3[i]) for i in range(16) if i not in (10, 11))
    assert not torch.isneginf(row3[10])


def test_sharded_deterministic_rows_use_global_ids() -> None:
    hidden = torch.randn(2, 4)
    weight = torch.randn(16, 4)
    valid_sets = [TokenSet([9]), TokenSet([2])]
    outs = []
    for rank in range(2):
        out, _ = build_draft_logits(
            hidden,
            weight[rank * 8 : (rank + 1) * 8],
            valid_sets,
            vocab_size=16,
            full_logits=lambda rows: rows @ weight.T,
            kernel=TorchGatherKernel(),
            deterministic=True,
            shard=_rank_shard(rank),
        )
        outs.append(out)
    combined = torch.where(torch.isneginf(outs[0]), outs[1], outs[0])
    for row, token in ((0, 9), (1, 2)):
        expected = torch.full((16,), float("-inf"))
        expected[token] = 0.0
        torch.testing.assert_close(combined[row], expected)


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
