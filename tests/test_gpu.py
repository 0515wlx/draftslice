from __future__ import annotations

import pytest

from draftslice.bench.gpu import GpuSnapshot, query_gpu


def test_snapshot_to_dict() -> None:
    snapshot = GpuSnapshot(
        index=0,
        name="test-gpu",
        sm_clock_mhz=1200,
        max_sm_clock_mhz=1410,
        mem_clock_mhz=1593,
        max_mem_clock_mhz=1593,
        driver_version="580.0",
        temperature_c=35.0,
        power_w=80.0,
        pstate="P0",
    )
    value = snapshot.to_dict()
    assert value["sm_clock_mhz"] == 1200
    assert value["max_sm_clock_mhz"] == 1410
    assert value["pstate"] == "P0"


def test_query_gpu_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        query_gpu(-1)
