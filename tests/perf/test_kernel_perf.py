from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.perf

if os.environ.get("DRAFTSLICE_RUN_PERF") != "1":
    pytest.skip(
        "set DRAFTSLICE_RUN_PERF=1 to run GPU performance tests",
        allow_module_level=True,
    )

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from draftslice.bench.gpu import GpuClockController, query_gpu  # noqa: E402
from draftslice.bench.kernel import (  # noqa: E402
    KernelBenchConfig,
    run_kernel_bench,
    write_kernel_result,
)

ROWS = (32, 128, 1_024, 4_096, 8_192, 32_768)
BATCHES = (1, 8, 32, 64)


def test_fused_slice_under_fixed_sm_clock(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    gpu = int(os.environ.get("DRAFTSLICE_GPU", "0"))
    iterations = int(os.environ.get("DRAFTSLICE_PERF_ITERATIONS", "30"))
    config = KernelBenchConfig(
        batch_sizes=BATCHES,
        row_counts=ROWS,
        warmup=20,
        iterations=iterations,
        device=f"cuda:{gpu}",
    )

    with GpuClockController(gpu) as clock:
        before = clock.snapshot
        assert before.sm_clock_mhz == clock.target_sm_clock_mhz
        assert before.sm_clock_mhz == before.max_sm_clock_mhz

        result = run_kernel_bench(config, clock=before)
        write_kernel_result(tmp_path / "kernel.json", result)

        after = query_gpu(gpu)
        assert after.sm_clock_mhz == clock.target_sm_clock_mhz

    assert result["clock_fixed"] is True
    assert result["clock"]["sm_clock_mhz"] == clock.target_sm_clock_mhz
    assert result["triton_available"] is True
    assert result["graph"]["available"] is True
    assert result["graph"]["captures"] > 0
    assert result["graph"]["replays"] > 0
    assert result["graph"]["fallbacks"] == 0

    rows = result["results"]
    assert rows
    for row in rows:
        assert row["median_us"] > 0
        assert row["cv"] < 1.00, f"unstable measurement: {row}"

    by_key = {(row["kernel"], row["mode"], row["batch"], row["rows"]): row for row in rows}
    for batch in config.batch_sizes:
        full = by_key[("full", "eager", batch, config.vocab_size)]
        full_graph = by_key[("full_graph", "graph", batch, config.vocab_size)]
        assert 400.0 < full["median_us"] < 2000.0
        assert full_graph["median_us"] <= 1.05 * full["median_us"]
        for m in config.row_counts:
            fused = by_key[("triton_slice", "eager", batch, m)]
            gather = by_key[("torch_gather", "eager", batch, m)]
            fused_graph = by_key[("triton_slice_graph", "graph", batch, m)]
            gather_graph = by_key[("torch_gather_graph", "graph", batch, m)]
            assert fused["median_us"] < full["median_us"]
            assert fused["numerical_max_abs_diff"] < 2.0
            assert fused_graph["numerical_max_abs_diff"] < 2.0
            assert fused_graph["median_us"] < fused["median_us"]
            if m >= 8_192:
                assert fused["median_us"] < 0.90 * gather["median_us"]
                assert fused_graph["median_us"] < 0.75 * gather_graph["median_us"]
            if m <= 1_024:
                # Launch-bound regime: graph replay is the only way to reach the
                # target on every supported batch. Use a relative bound so a
                # temporarily throttled GPU does not fail the correctness gate.
                assert fused_graph["median_us"] < 0.60 * fused["median_us"]

    saved = json.loads((tmp_path / "kernel.json").read_text())
    assert saved["schema_version"] == "kernel_bench/1"
    assert saved["clock_fixed"] is True
