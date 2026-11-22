from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from draftslice.plan import (
    DraftContract,
    Execution,
    Guarantee,
    ServingPoint,
    TokenSet,
)
from draftslice.plan.compiler import PlanCompiler
from draftslice.plan.serialize import (
    CONTRACT_SCHEMA,
    compute_contract_id,
    contract_from_dict,
    contract_to_dict,
    plan_from_dict,
    plan_to_dict,
)
from draftslice.plan.types import GuaranteeKind


def _strict() -> DraftContract:
    valid = TokenSet([1, 3, 5, 7])
    return DraftContract(
        grammar_state_id="req-a:12",
        valid_set=valid,
        candidate_set=TokenSet([1, 5]),
        mode="strict",
        execution=Execution(kind="slice_gemv", rows=TokenSet([1, 5])),
        guarantee=Guarantee.exact_valid_set(),
        serving=ServingPoint(k=4, source="draftslice_tp2", head_share=0.25),
    )


def test_contract_is_frozen_and_validates_subset() -> None:
    contract = _strict()
    assert contract.is_sliced
    assert contract.guarantee.scope == "mu_proj"
    with pytest.raises(FrozenInstanceError):
        contract.mode = "full"  # type: ignore[misc]
    with pytest.raises(ValueError):
        DraftContract(
            grammar_state_id=None,
            valid_set=TokenSet([1]),
            candidate_set=TokenSet([2]),
            mode="strict",
            execution=Execution(kind="slice_gemv", rows=TokenSet([2])),
            guarantee=Guarantee.exact_valid_set(),
            serving=ServingPoint(),
        )


def test_contract_rejects_mu_star_scope() -> None:
    with pytest.raises(ValueError, match="mu_proj"):
        DraftContract(
            grammar_state_id=None,
            valid_set=None,
            candidate_set=None,
            mode="full",
            execution=Execution(kind="full"),
            guarantee=Guarantee(kind=GuaranteeKind.FULL, universe="vocab", scope="mu_star"),
            serving=ServingPoint(),
        )


def test_full_contract_carries_fallback_reason() -> None:
    contract = DraftContract(
        grammar_state_id="req-b:0",
        valid_set=TokenSet([2]),
        candidate_set=None,
        mode="full",
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        serving=ServingPoint(source="builtin"),
        fallback_reason="full_cuda_graph",
    )
    assert not contract.is_sliced
    assert contract.to_dict()["fallback_reason"] == "full_cuda_graph"


def test_contract_serialization_round_trip() -> None:
    contract = _strict()
    payload = contract_to_dict(contract)
    assert payload["schema_version"] == CONTRACT_SCHEMA
    assert payload["contract_id"] == compute_contract_id(contract)
    # Stable JSON and deterministic id under key order.
    assert json.loads(json.dumps(payload)) == payload
    restored = contract_from_dict(payload)
    assert restored == contract
    assert restored.serving.k == 4
    assert restored.guarantee.kind is GuaranteeKind.EXACT_VALID_SET


def test_calibrated_contract_round_trip() -> None:
    contract = DraftContract(
        grammar_state_id=None,
        valid_set=TokenSet([0, 2]),
        candidate_set=TokenSet([2]),
        mode="mask",
        execution=Execution(kind="slice_gemm", rows=TokenSet([2])),
        guarantee=Guarantee.calibrated(
            0.05, universe="valid", calibration_id="cal-1"
        ),
        serving=ServingPoint(k=1, source="draftslice+static", head_share=0.5),
    )
    assert contract_from_dict(contract_to_dict(contract)) == contract


def test_guarantee_serialization_keeps_scope_and_fallback_rule() -> None:
    plan = PlanCompiler(vocab_size=8, hidden_size=4).compile_full()
    payload = plan_to_dict(plan)
    assert payload["guarantee"]["scope"] == "mu_proj"
    assert payload["guarantee"]["fallback_rule"] == "full_head"
    assert plan_from_dict(payload).guarantee == plan.guarantee
