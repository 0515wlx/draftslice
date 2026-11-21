from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from draftslice.bench.baseline_runner import (  # noqa: E402
    BaselineRunnerError,
    HeadQualityCase,
    run_matrix,
    run_policy,
    write_matrix,
)
from draftslice.bench.baselines import FullPolicy, StaticPolicy  # noqa: E402
from draftslice.plan import TokenSet  # noqa: E402


def case() -> HeadQualityCase:
    return HeadQualityCase(
        hidden=torch.zeros(2, 4),
        truth=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10)),
        domains=("a", "b"),
        valid_sets=(TokenSet(range(1, 11)), TokenSet(range(1, 11))),
    )


def test_run_policy_and_matrix() -> None:
    full = run_policy(FullPolicy(vocab_size=12), case(), k=5)
    assert full.metrics["top1"] == 1.0
    static = run_policy(StaticPolicy((1, 2, 3, 4, 5, 6, 7, 8, 9, 10)), case(), k=5)
    assert static.metrics["top1"] == 1.0
    assert static.metrics["m_mean"] == 10.0
    matrix = run_matrix(
        [FullPolicy(vocab_size=12), StaticPolicy((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))],
        case(),
        k=5,
    )
    assert set(matrix) == {"full", "static"}


def test_intersect_valid() -> None:
    case_ = HeadQualityCase(
        hidden=torch.zeros(1, 4),
        truth=((1, 2, 3, 4, 5),),
        valid_sets=(TokenSet([1, 2]),),
    )
    result = run_policy(
        FullPolicy(vocab_size=12),
        case_,
        k=5,
        intersect_valid=True,
    )
    assert result.metrics["valid_recall"] == 1.0
    assert result.metrics["valid_precision"] == 1.0


def test_write_matrix(tmp_path) -> None:
    matrix = run_matrix([FullPolicy(vocab_size=12)], case(), k=5)
    path = tmp_path / "matrix.json"
    write_matrix(path, matrix, case_name="test", metadata={"model": "toy"})
    value = json.loads(path.read_text())
    assert value["schema_version"] == "baseline_matrix/1"
    assert value["case"] == "test"
    assert "full" in value["policies"]


def test_case_validation() -> None:
    with pytest.raises(BaselineRunnerError):
        HeadQualityCase(hidden=torch.zeros(2, 4), truth=((1,),))
    with pytest.raises(BaselineRunnerError):
        HeadQualityCase(
            hidden=torch.zeros(2, 4),
            truth=((1,), (2,)),
            valid_sets=(TokenSet([1]),),
        )
