from __future__ import annotations

import json

import pytest

from draftslice.bench.release import (
    ReleaseError,
    ReleaseManifest,
    build_manifest,
    build_release,
    compare_manifests,
    file_hash,
    load_result,
    render_report,
    validate_result,
    verify_manifest,
)
from draftslice.cli import main


def _bench_result(name: str = "head") -> dict:
    return {
        "schema_version": "bench/1",
        "name": name,
        "metrics": {"top1": 0.5, "set_recall@5": 0.75},
        "warnings": ["example warning"],
    }


def _factorial_result() -> dict:
    return {
        "schema_version": "factorial/1",
        "k": 5,
        "dtype_bytes": 2,
        "screeners": ["static"],
        "cases": ["case-a"],
        "cells": [
            {
                "screener": "static",
                "case": "case-a",
                "coverage": 0.9,
                "mean_candidates": 32.0,
                "candidate_bytes": 128,
            }
        ],
    }


def test_validate_result_accepts_implemented_schemas() -> None:
    assert validate_result(_bench_result()) == "bench/1"
    assert validate_result(_factorial_result()) == "factorial/1"
    assert (
        validate_result(
            {"schema_version": "baseline_matrix/1", "policies": {}}
        )
        == "baseline_matrix/1"
    )
    assert (
        validate_result({"schema_version": "e2e/1", "methods": {}}) == "e2e/1"
    )


def test_validate_result_rejects_unknown_or_incomplete() -> None:
    with pytest.raises(ReleaseError):
        validate_result({"schema_version": "nope/1"})
    with pytest.raises(ReleaseError):
        validate_result({"schema_version": "bench/1"})
    with pytest.raises(ReleaseError):
        validate_result({"schema_version": "factorial/1"})


def test_build_release_report_and_manifest(tmp_path) -> None:
    bench = tmp_path / "bench.json"
    factorial = tmp_path / "factorial.json"
    bench.write_text(json.dumps(_bench_result()))
    factorial.write_text(json.dumps(_factorial_result()))
    out = tmp_path / "report.md"
    manifest, report = build_release(
        "test-release",
        [bench, factorial],
        out=out,
        metadata={"model": "tiny"},
        root=tmp_path,
    )
    assert out.exists()
    assert "[test-release]" not in report
    assert "Release report: test-release" in report
    assert "| top1 | 0.5000 |" in report
    assert "| static | case-a | 0.9000 | 32.0000 | 128 |" in report
    assert "example warning" in report
    manifest_path = out.with_suffix(".manifest.json")
    assert manifest_path.exists()
    loaded = ReleaseManifest.load(manifest_path)
    assert [item.schema for item in loaded.files] == ["bench/1", "factorial/1"]
    assert loaded.metadata == {"model": "tiny"}
    assert verify_manifest(loaded) == []


def test_manifest_detects_tampering(tmp_path) -> None:
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(_bench_result()))
    manifest = build_manifest("r", [path], root=tmp_path)
    assert manifest.files[0].sha256 == file_hash(path)
    path.write_text(json.dumps({**_bench_result(), "name": "changed"}))
    problems = verify_manifest(manifest)
    assert any("hash mismatch" in problem for problem in problems)


def test_manifest_rejects_unknown_schema_input(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "bad/1"}))
    with pytest.raises(ReleaseError):
        build_manifest("r", [path], root=tmp_path)
    with pytest.raises(ReleaseError):
        load_result(path)


def test_compare_reports_missing_kinds(tmp_path) -> None:
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(_bench_result()))
    base = build_manifest("base", [bench], root=tmp_path)
    factorial = tmp_path / "factorial.json"
    factorial.write_text(json.dumps(_factorial_result()))
    candidate = build_manifest("candidate", [factorial], root=tmp_path)
    report = compare_manifests(base, candidate)
    assert "Missing in candidate: head_accuracy" in report
    assert "Added in candidate: factorial" in report


def test_report_rejects_length_mismatch() -> None:
    with pytest.raises(ReleaseError):
        render_report(ReleaseManifest(name="r"), [_bench_result()])


def test_cli_report_validate_and_compare(tmp_path, capsys) -> None:
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(_bench_result()))
    out = tmp_path / "report.md"
    assert (
        main(
            [
                "bench",
                "report",
                "--input",
                str(bench),
                "--out",
                str(out),
                "--name",
                "cli-release",
            ]
        )
        == 0
    )
    assert out.exists()
    capsys.readouterr()
    assert main(["bench", "validate", "--manifest", str(out.with_suffix(".manifest.json"))]) == 0
    assert "ok:" in capsys.readouterr().out

    other = tmp_path / "other.json"
    other.write_text(json.dumps(_factorial_result()))
    out2 = tmp_path / "report2.md"
    assert (
        main(
            [
                "bench",
                "report",
                "--input",
                str(other),
                "--out",
                str(out2),
                "--name",
                "cli-release-2",
            ]
        )
        == 0
    )
    compare_out = tmp_path / "compare.md"
    assert (
        main(
            [
                "bench",
                "compare",
                "--base",
                str(out.with_suffix(".manifest.json")),
                "--candidate",
                str(out2.with_suffix(".manifest.json")),
                "--out",
                str(compare_out),
            ]
        )
        == 0
    )
    assert "B5 comparison" in compare_out.read_text()
