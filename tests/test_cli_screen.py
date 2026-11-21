from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from draftslice.artifact import ArtifactBundle  # noqa: E402
from draftslice.cli import main  # noqa: E402
from draftslice.screen import (  # noqa: E402
    StaticPolicy,
    build_screener_bundle,
    screener_from_bundle,
)


def test_bench_factorial_runs_artifacts(tmp_path, capsys) -> None:
    bundle = build_screener_bundle(
        StaticPolicy(rows=(1, 3, 5)), vocab_size=8, hidden_size=4
    )
    bundle.write(tmp_path / "static")
    case_file = tmp_path / "cases.pt"
    torch.save(
        {
            "cases": [
                {
                    "name": "case-a",
                    "hidden": torch.randn(2, 4),
                    "truth": [[1, 3], [1, 5]],
                }
            ]
        },
        case_file,
    )
    out = tmp_path / "factorial.json"
    assert (
        main(
            [
                "bench",
                "factorial",
                "--input",
                str(case_file),
                "--artifact",
                str(tmp_path / "static"),
                "--out",
                str(out),
                "--k",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(out.read_text())
    assert result["schema_version"] == "factorial/1"
    assert result["screeners"] == ["static"]
    assert result["cells"][0]["coverage"] == 1.0
    assert "case-a" in capsys.readouterr().out


def test_bench_factorial_requires_a_screener(tmp_path, capsys) -> None:
    case_file = tmp_path / "cases.pt"
    torch.save(
        {"cases": [{"name": "c", "hidden": torch.randn(1, 4), "truth": [[1]]}]},
        case_file,
    )
    assert (
        main(
            [
                "bench",
                "factorial",
                "--input",
                str(case_file),
                "--out",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    assert "provide --artifact" in capsys.readouterr().out


def test_screen_train_router_writes_artifact(tmp_path, capsys) -> None:
    torch.manual_seed(0)
    hidden = torch.eye(4).repeat(2, 1)
    group_mass = torch.eye(2).repeat(4, 1)
    input_path = tmp_path / "train.pt"
    torch.save(
        {
            "hidden": hidden,
            "group_mass": group_mass,
            "groups": [[0, 1], [2, 3]],
            "vocab_size": 4,
        },
        input_path,
    )
    out = tmp_path / "router"
    assert (
        main(
            [
                "screen",
                "train-router",
                "--input",
                str(input_path),
                "--out",
                str(out),
                "--top-c",
                "1",
                "--epochs",
                "5",
                "--hidden-dim",
                "8",
                "--seed",
                "1",
            ]
        )
        == 0
    )
    bundle = ArtifactBundle.load(out)
    policy = screener_from_bundle(bundle)
    assert policy.name == "mlp_router"
    assert policy.top_c == 1
    report = json.loads((out / "training_report.json").read_text())
    assert report["n_train"] + report["n_val"] == 8
    assert report["vocab_size"] == 4
    assert "wrote" in capsys.readouterr().out
