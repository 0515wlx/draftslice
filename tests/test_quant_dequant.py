from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.quant_dequant import (  # noqa: E402
    dequantize_rows,
    describe_quantized_head,
    is_quantized_weight,
)


def test_bf16_head_is_not_quantized() -> None:
    assert not is_quantized_weight(torch.zeros(4, 8, dtype=torch.bfloat16))


def test_row_major_fp8_head_dequantizes_with_per_tensor_scale() -> None:
    head_module = SimpleNamespace(embedding_dim=8, weight_scale=0.5)
    q = (torch.arange(32, dtype=torch.float32).reshape(4, 8) / 4).to(
        torch.float8_e4m3fn
    )
    described = describe_quantized_head(head_module, q)
    assert described is not None
    assert not described.transposed
    rows = dequantize_rows(described, [1, 3])
    expected = (q[[1, 3]].to(torch.float32) * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(rows, expected)


def test_transposed_fp8_head_gathers_columns() -> None:
    head_module = SimpleNamespace(embedding_dim=8, weight_scale=0.25)
    q = (torch.arange(32, dtype=torch.float32).reshape(8, 4) / 4).to(
        torch.float8_e4m3fn
    )
    described = describe_quantized_head(head_module, q)
    assert described is not None
    assert described.transposed
    rows = dequantize_rows(described, [0, 2])
    expected = (q[:, [0, 2]].t().to(torch.float32) * 0.25).to(torch.bfloat16)
    torch.testing.assert_close(rows, expected)


def test_per_channel_scale_uses_local_rows() -> None:
    head_module = SimpleNamespace(
        embedding_dim=8, weight_scale=torch.tensor([0.5, 1.0, 2.0, 4.0])
    )
    q = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    described = describe_quantized_head(head_module, q)
    assert described is not None
    rows = dequantize_rows(described, [1, 3])
    expected = torch.tensor([[1.0] * 8, [4.0] * 8], dtype=torch.bfloat16)
    torch.testing.assert_close(rows, expected)


def test_scale_inverse_is_divided() -> None:
    head_module = SimpleNamespace(embedding_dim=8, weight_scale_inv=2.0)
    q = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    described = describe_quantized_head(head_module, q)
    assert described is not None
    torch.testing.assert_close(
        dequantize_rows(described, [0]), torch.full((1, 8), 0.5, dtype=torch.bfloat16)
    )


def test_prepare_weight_materializes_only_candidate_rows() -> None:
    from types import SimpleNamespace

    from draftslice.integrations.vllm import DraftSliceSpeculatorMixin, DraftSliceVllmStats
    from draftslice.plan import TokenSet

    class Speculator(DraftSliceSpeculatorMixin):
        pass

    speculator = Speculator.__new__(Speculator)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_dequant_weight = None
    speculator._draftslice_dequant_filled = set()
    speculator._draftslice_quant_head = describe_quantized_head(
        SimpleNamespace(embedding_dim=4, weight_scale=0.5),
        torch.full((8, 4), 2.0, dtype=torch.float8_e4m3fn),
    )
    assert speculator._draftslice_quant_head is not None
    prepared = speculator._prepare_weight(
        speculator._draftslice_quant_head.weight, [TokenSet([1, 3])], None, 8, 8
    )
    torch.testing.assert_close(prepared[1], torch.full((4,), 1.0, dtype=torch.bfloat16))
    torch.testing.assert_close(prepared[3], torch.full((4,), 1.0, dtype=torch.bfloat16))
    assert torch.count_nonzero(prepared[0]) == 0
    assert speculator._draftslice_stats.quantized_rows == 2
    # A second call reuses the cached rows.
    speculator._prepare_weight(
        speculator._draftslice_quant_head.weight, [TokenSet([1, 3])], None, 8, 8
    )
    assert speculator._draftslice_stats.quantized_rows == 2


def test_unsupported_scheme_is_rejected() -> None:
    q = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    assert describe_quantized_head(SimpleNamespace(embedding_dim=8), q) is None
    described = describe_quantized_head(
        SimpleNamespace(embedding_dim=8, weight_scale=torch.ones(2, 4)), q
    )
    assert described is not None
    # A scale whose shape cannot be mapped to rows is rejected at gather time.
    assert dequantize_rows(described, [0]) is None
