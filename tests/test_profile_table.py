from __future__ import annotations

import json

import pytest

from draftslice.runtime.profile import (
    ProfileHashError,
    ProfileParseError,
    ProfileTable,
    ProfileValidationError,
    ProfitCell,
    batch_bucket,
    build_table,
    valid_bucket,
)
from draftslice.runtime.reasons import FallbackReason


def _cell(**overrides) -> ProfitCell:
    data = {
        "model": "Qwen3.5-4B",
        "engine": "vllm",
        "method": "draftslice_mtp",
        "workload": "tool_call",
        "batch_bucket": "1",
        "k": 8,
        "tp": 1,
        "quant": "bf16",
        "graph": "eager",
        "valid_bucket": "65-512",
        "action": "strict_slice",
        "expected_gain": 1.24,
        "measured_p99_ratio": 1.02,
        "manifest_id": "b1-tool-k8",
        "enabled": True,
    }
    data.update(overrides)
    return ProfitCell(**data)


def _lookup(table: ProfileTable, **overrides):
    data = {
        "model": "Qwen3.5-4B",
        "engine": "vllm",
        "method": "draftslice_mtp",
        "workload": "tool_call",
        "batch": 1,
        "k": 8,
        "tp": 1,
        "quant": "bf16",
        "graph": "eager",
        "valid": 200,
    }
    data.update(overrides)
    return table.lookup(**data)


def test_buckets_are_stable() -> None:
    assert [batch_bucket(b) for b in (1, 2, 8, 9, 32, 33, 64, 65)] == [
        "1",
        "2-8",
        "2-8",
        "9-32",
        "9-32",
        "33-64",
        "33-64",
        ">64",
    ]
    assert [valid_bucket(v) for v in (0, 1, 64, 65, 512, 513, 4096, 4097)] == [
        "0-64",
        "0-64",
        "0-64",
        "65-512",
        "65-512",
        "513-4096",
        "513-4096",
        ">4096",
    ]


def test_cell_id_is_deterministic_and_unique_per_bucket() -> None:
    cell = _cell()
    assert cell.cell_id == (
        "Qwen3.5-4B|vllm|draftslice_mtp|tool_call|1|k8|tp1|bf16|eager|65-512"
    )
    assert _cell(batch_bucket="2-8").cell_id != cell.cell_id
    assert _cell(tp=2).cell_id != cell.cell_id


def test_cell_validation() -> None:
    with pytest.raises(ProfileValidationError):
        _cell(engine="trtllm")
    with pytest.raises(ProfileValidationError):
        _cell(batch_bucket="0-1")
    with pytest.raises(ProfileValidationError):
        _cell(k=-1)
    with pytest.raises(ProfileValidationError):
        _cell(expected_gain=0.0)
    with pytest.raises(ProfileValidationError):
        _cell(action="slice")
    with pytest.raises(ProfileValidationError):
        _cell(enabled="yes")  # type: ignore[arg-type]


def test_empty_table_means_builtin() -> None:
    table = ProfileTable.empty()
    match = _lookup(table)
    assert not match.hit
    assert match.reason is FallbackReason.PROFILES_UNAVAILABLE
    assert match.cell is None


def test_exact_cell_hit_and_miss_boundaries() -> None:
    table = build_table([_cell()], table_id="t", engine="vllm", model="Qwen3.5-4B")
    hit = _lookup(table)
    assert hit.hit
    assert hit.cell is not None and hit.cell.cell_id == _cell().cell_id

    # A different bucket is a miss, never a nearest-cell fallback.
    for override in (
        {"batch": 2},
        {"valid": 32},
        {"valid": 5000},
        {"k": 4},
        {"workload": "free"},
        {"quant": "fp8"},
    ):
        miss = _lookup(table, **override)
        assert not miss.hit, override
        assert miss.reason is FallbackReason.NO_CELL


def test_disabled_cell_and_engine_mismatch_fall_back() -> None:
    table = build_table([_cell(enabled=False)], table_id="t")
    match = _lookup(table)
    assert not match.hit
    assert match.reason is FallbackReason.PROFILE_DISABLED
    assert match.cell is not None

    other_engine = build_table([_cell()], table_id="t", engine="vllm")
    mismatch = _lookup(other_engine, engine="sglang")
    assert not mismatch.hit
    assert mismatch.reason is FallbackReason.ENGINE_MISMATCH


def test_builtin_action_does_not_execute_draftslice() -> None:
    table = build_table([_cell(action="builtin")], table_id="t")
    match = _lookup(table)
    assert not match.hit
    assert match.reason is FallbackReason.PROFILE_DISABLED
    assert not table.enabled_cells


def test_duplicate_cells_are_rejected() -> None:
    with pytest.raises(ProfileValidationError):
        ProfileTable(cells=(_cell(), _cell()), table_id="t")


def test_table_hash_roundtrip_and_tamper(tmp_path) -> None:
    table = build_table([_cell()], table_id="t", notes="gate 2026-11-22")
    assert table.hash.startswith("sha256:")
    path = tmp_path / "profiles.json"
    saved = table.save(path)
    assert saved == table.hash
    loaded = ProfileTable.load(path)
    assert loaded.hash == table.hash
    assert loaded.digest() == table.hash
    assert loaded.cells[0].cell_id == _cell().cell_id

    # Any content edit invalidates the artifact: hard error, never a silent use.
    document = json.loads(path.read_text())
    document["cells"][0]["expected_gain"] = 9.99
    path.write_text(json.dumps(document))
    with pytest.raises(ProfileHashError):
        ProfileTable.load(path)

    # A missing hash is a parse error unless the caller explicitly allows it.
    path.write_text(json.dumps(table.canonical_dict()))
    with pytest.raises(ProfileParseError):
        ProfileTable.load(path)
    loose = ProfileTable.load(path, require_hash=False)
    assert loose.digest() == table.hash


def test_unknown_schema_and_bad_json(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"schema": "profiles/2", "cells": []}))
    with pytest.raises(ProfileParseError):
        ProfileTable.load(path)
    path.write_text("{not json")
    with pytest.raises(ProfileParseError):
        ProfileTable.load(path)


def test_table_engine_and_model_must_match_every_cell() -> None:
    with pytest.raises(ProfileValidationError):
        build_table([_cell()], table_id="t", engine="sglang")
    with pytest.raises(ProfileValidationError):
        build_table([_cell()], table_id="t", model="Qwen3.5-35B-A3B")
