from __future__ import annotations

import pytest

from draftslice.kernels.tuning import KernelConfig, gemm_config, gemv_config


def test_kernel_config_rejects_bad_blocks() -> None:
    with pytest.raises(ValueError):
        KernelConfig(block_m=3, block_d=128, block_b=16, num_warps=4, num_stages=2)
    with pytest.raises(ValueError):
        KernelConfig(block_m=16, block_d=128, block_b=16, num_warps=0, num_stages=2)


@pytest.mark.parametrize("rows", [1, 8, 128, 1_024, 4_096, 8_192, 32_768])
def test_gemv_config_covers_all_row_regimes(rows: int) -> None:
    config = gemv_config(rows, 2_560)
    assert config.block_m >= 1
    assert config.block_d <= 4_096
    assert config.block_b == 1
    assert config.block_d & (config.block_d - 1) == 0


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("rows", [1, 128, 1_024, 4_096, 8_192, 32_768])
def test_gemm_config_covers_all_shapes(batch: int, rows: int) -> None:
    config = gemm_config(rows, batch, 2_560)
    assert config.block_b >= batch
    assert config.block_b & (config.block_b - 1) == 0
    assert config.block_m & (config.block_m - 1) == 0
    assert config.block_d & (config.block_d - 1) == 0
    assert config.num_warps > 0
    assert config.num_stages > 0


def test_gemm_config_tracks_batch_and_width() -> None:
    small = gemm_config(128, 4, 2_560)
    large = gemm_config(32_768, 32, 2_560)
    assert small.block_b == 16
    assert large.block_b == 32
    assert gemm_config(128, 4, 64).block_d <= 64


def test_autotune_candidates_are_valid_and_deduplicated() -> None:
    from draftslice.kernels.autotune import gemm_candidates, gemv_candidates

    for config in gemv_candidates(1_024, 2_560):
        assert config.block_b == 1
        assert config.block_d & (config.block_d - 1) == 0
    candidates = gemm_candidates(4_096, 8, 2_560)
    assert 1 <= len(candidates) <= 8
    keys = {(c.block_m, c.block_d, c.block_b, c.num_warps, c.num_stages) for c in candidates}
    assert len(keys) == len(candidates)
    assert all(candidate.block_b >= 16 for candidate in candidates)
