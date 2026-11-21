from __future__ import annotations

from pathlib import Path

import pytest

from draftslice.cli import main
from draftslice.plan.serialize import read_plan

FIXTURES = Path(__file__).parent / "fixtures"


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "draftslice" in capsys.readouterr().out


def test_version() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_compile_from_valid_ids(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "plan.json"
    assert (
        main(
            [
                "compile",
                "--valid-ids",
                "5,1,3",
                "--state-id",
                "2",
                "--vocab-size",
                "8",
                "--hidden-size",
                "4",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    plan = read_plan(out)
    assert plan.state_id == 2
    assert plan.candidates.ids == (1, 3, 5)
    assert plan.execution.kind == "slice_gemv"
    assert "wrote" in capsys.readouterr().out


def test_compile_from_fixture(tmp_path) -> None:
    out = tmp_path / "plan.json"
    assert (
        main(
            [
                "compile",
                "--schema",
                str(FIXTURES / "token_set.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    plan = read_plan(out)
    assert plan.state_id == 2
    assert plan.candidates.ids == (1, 3, 5)


def test_compile_selects_state(tmp_path) -> None:
    out = tmp_path / "plan.json"
    assert (
        main(
            [
                "compile",
                "--schema",
                str(FIXTURES / "states.json"),
                "--state-id",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert read_plan(out).candidates.ids == (2, 3, 7)


def test_compile_falls_back_when_over_budget(tmp_path) -> None:
    out = tmp_path / "plan.json"
    assert (
        main(
            [
                "compile",
                "--valid-ids",
                "1,3,5",
                "--vocab-size",
                "8",
                "--hidden-size",
                "4",
                "--max-candidates",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    plan = read_plan(out)
    assert plan.execution.kind == "full"
    assert plan.source == "fallback"
    assert plan.candidates.ids == (1, 3, 5)


def test_compile_can_forbid_fallback(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "plan.json"
    assert (
        main(
            [
                "compile",
                "--valid-ids",
                "1,3,5",
                "--vocab-size",
                "8",
                "--hidden-size",
                "4",
                "--max-candidates",
                "2",
                "--no-full-fallback",
                "--out",
                str(out),
            ]
        )
        == 2
    )
    assert "over budget" in capsys.readouterr().out


def test_compile_requires_a_source(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compile", "--out", "plan.json"]) == 2
    assert "provide --schema or --valid-ids" in capsys.readouterr().out


def test_compile_rejects_bad_fixture(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"type": "object"}')
    assert main(["compile", "--schema", str(bad), "--out", str(tmp_path / "plan.json")]) == 2
    assert "unsupported constraint fixture" in capsys.readouterr().out


def test_bench_accuracy(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"candidates": [1, 2, 3], "truth": [1, 2, 3, 4, 5], '
                '"valid": [1, 2, 3, 4, 5], "domain": "a"}',
                '{"candidates": [4], "truth": [4, 5, 6, 7, 8], '
                '"valid": [4, 5, 6, 7, 8], "domain": "b"}',
            ]
        )
        + "\n"
    )
    out = tmp_path / "result.json"
    assert main(["bench", "accuracy", "--input", str(source), "--out", str(out)]) == 0
    assert out.exists()
    assert "wrote" in capsys.readouterr().out


def test_bench_constraint(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "states.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"state_id": 0, "valid_ids": [1, 3, 5]}',
                '{"state_id": 0, "valid_ids": [1, 3, 5]}',
                '{"state_id": 1, "valid_ids": [2, 4]}',
            ]
        )
        + "\n"
    )
    out = tmp_path / "b3.json"
    assert main(["bench", "constraint", "--input", str(source), "--out", str(out)]) == 0
    assert out.exists()
    assert "states=3" in capsys.readouterr().out


def test_bench_config_is_still_a_placeholder(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bench", "--config", "config.toml"]) == 2
    assert "M2" in capsys.readouterr().out
