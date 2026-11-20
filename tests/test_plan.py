from __future__ import annotations

import draftslice
from draftslice.plan import HeadPlan, Slice


def test_version() -> None:
    assert draftslice.__version__ == "0.0.1"


def test_slice_size() -> None:
    s = Slice(state_id=7, token_ids=(1, 2, 3))
    assert s.size == 3


def test_head_plan_defaults() -> None:
    plan = HeadPlan(slice=Slice(state_id=0, token_ids=(42,)))
    assert plan.kernel == "auto"
    assert plan.packed is False
    assert plan.estimated_bytes is None
