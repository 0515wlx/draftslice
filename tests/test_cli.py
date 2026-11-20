from __future__ import annotations

import pytest

from draftslice.cli import main


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "draftslice" in capsys.readouterr().out


def test_compile_is_a_placeholder(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compile", "--schema", "schema.json", "--out", "plan.json"]) == 2
    assert "M1" in capsys.readouterr().out


def test_version() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
