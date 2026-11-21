"""JSON serialization for plans.

The format is intentionally boring: a JSON object with explicit schema versions.
Small plans embed their token ids inline; a future revision can move large id
arrays to a sidecar without changing the logical format.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import (
    Execution,
    Guarantee,
    GuaranteeKind,
    HeadPlan,
    NodePlan,
    PlanCost,
    StepPlan,
    TokenSet,
)

PLAN_SCHEMA = "plan/1"
STEP_PLAN_SCHEMA = "step_plan/1"


class PlanFormatError(ValueError):
    """Raised when a serialized plan is malformed or unsupported."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _token_set_to_dict(tokens: TokenSet) -> dict[str, Any]:
    return {"encoding": "u32", "count": len(tokens), "ids": list(tokens.ids)}


def _token_set_from_dict(value: Any) -> TokenSet:
    if not isinstance(value, dict):
        raise PlanFormatError("token set must be an object")
    encoding = value.get("encoding", "u32")
    if encoding != "u32":
        raise PlanFormatError(f"unsupported token encoding: {encoding}")
    ids = value.get("ids")
    if not isinstance(ids, list):
        raise PlanFormatError("token set must contain an 'ids' list")
    try:
        tokens = TokenSet(ids)
    except (TypeError, ValueError) as exc:
        raise PlanFormatError(f"invalid token ids: {exc}") from exc
    count = value.get("count")
    if count is not None and count != len(tokens):
        raise PlanFormatError("token set count does not match ids")
    return tokens


def _guarantee_to_dict(guarantee: Guarantee) -> dict[str, Any]:
    return {
        "kind": guarantee.kind.value,
        "universe": guarantee.universe,
        "delta": guarantee.delta,
        "calibration_id": guarantee.calibration_id,
    }


def _guarantee_from_dict(value: Any) -> Guarantee:
    if not isinstance(value, dict):
        raise PlanFormatError("guarantee must be an object")
    try:
        kind = GuaranteeKind(value["kind"])
    except (KeyError, ValueError) as exc:
        raise PlanFormatError(f"invalid guarantee kind: {value.get('kind')!r}") from exc
    universe = value.get("universe")
    if universe not in ("vocab", "valid"):
        raise PlanFormatError(f"invalid guarantee universe: {universe!r}")
    try:
        return Guarantee(
            kind=kind,
            universe=universe,
            delta=value.get("delta"),
            calibration_id=value.get("calibration_id"),
        )
    except ValueError as exc:
        raise PlanFormatError(str(exc)) from exc


def _execution_to_dict(execution: Execution) -> dict[str, Any]:
    return {
        "kind": execution.kind,
        "rows": None if execution.rows is None else _token_set_to_dict(execution.rows),
        "layout": execution.layout,
    }


def _execution_from_dict(value: Any) -> Execution:
    if not isinstance(value, dict):
        raise PlanFormatError("execution must be an object")
    kind = value.get("kind")
    if kind not in ("full", "slice_gemv", "slice_gemm"):
        raise PlanFormatError(f"invalid execution kind: {kind!r}")
    rows_value = value.get("rows")
    rows = None if rows_value is None else _token_set_from_dict(rows_value)
    layout = value.get("layout", "native")
    if layout not in ("native", "leaf_contiguous", "packed"):
        raise PlanFormatError(f"invalid layout: {layout!r}")
    try:
        return Execution(kind=kind, rows=rows, layout=layout)
    except ValueError as exc:
        raise PlanFormatError(str(exc)) from exc


def _cost_to_dict(cost: PlanCost) -> dict[str, Any]:
    return {
        "static_bytes": cost.static_bytes,
        "dynamic_bytes": cost.dynamic_bytes,
        "macs": cost.macs,
        "alu_ops": cost.alu_ops,
        "launch_count": cost.launch_count,
        "predicted_us": dict(cost.predicted_us),
    }


def _cost_from_dict(value: Any) -> PlanCost:
    if value is None:
        return PlanCost.zero()
    if not isinstance(value, dict):
        raise PlanFormatError("cost must be an object")
    predicted = value.get("predicted_us", {})
    if not isinstance(predicted, dict):
        raise PlanFormatError("cost.predicted_us must be an object")
    try:
        return PlanCost(
            static_bytes=int(value.get("static_bytes", 0)),
            dynamic_bytes=int(value.get("dynamic_bytes", 0)),
            macs=int(value.get("macs", 0)),
            alu_ops=int(value.get("alu_ops", 0)),
            launch_count=int(value.get("launch_count", 0)),
            predicted_us={str(key): float(item) for key, item in predicted.items()},
        )
    except (TypeError, ValueError) as exc:
        raise PlanFormatError(f"invalid cost: {exc}") from exc


def plan_payload(plan: HeadPlan, *, include_id: bool = False) -> dict[str, Any]:
    """Return the canonical JSON payload for a head plan."""

    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "state_id": plan.state_id,
        "candidates": _token_set_to_dict(plan.candidates),
        "execution": _execution_to_dict(plan.execution),
        "guarantee": _guarantee_to_dict(plan.guarantee),
        "cost": _cost_to_dict(plan.cost),
        "source": plan.source,
    }
    if include_id:
        payload["plan_id"] = plan.plan_id or compute_plan_id(plan)
    return payload


def compute_plan_id(plan: HeadPlan) -> str:
    digest = hashlib.sha256(_canonical_json(plan_payload(plan)).encode()).hexdigest()
    return f"sha256:{digest}"


def plan_to_dict(plan: HeadPlan) -> dict[str, Any]:
    return plan_payload(plan, include_id=True)


def plan_from_dict(value: Any) -> HeadPlan:
    if not isinstance(value, dict):
        raise PlanFormatError("plan must be an object")
    version = value.get("schema_version")
    if version != PLAN_SCHEMA:
        raise PlanFormatError(f"unsupported plan schema version: {version!r}")
    state_id = value.get("state_id")
    if state_id is not None and (
        isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0
    ):
        raise PlanFormatError("state_id must be a non-negative integer or null")
    try:
        plan = HeadPlan(
            candidates=_token_set_from_dict(value.get("candidates")),
            execution=_execution_from_dict(value.get("execution")),
            guarantee=_guarantee_from_dict(value.get("guarantee")),
            state_id=state_id,
            cost=_cost_from_dict(value.get("cost")),
            source=str(value.get("source", "manual")),
            plan_id=str(value.get("plan_id", "")),
        )
    except ValueError as exc:
        raise PlanFormatError(str(exc)) from exc
    return plan


def dumps(plan: HeadPlan, *, indent: int | None = 2) -> str:
    return json.dumps(plan_to_dict(plan), indent=indent, sort_keys=True)


def loads(text: str) -> HeadPlan:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanFormatError(f"invalid JSON: {exc}") from exc
    return plan_from_dict(value)


def write_plan(path: str | Path, plan: HeadPlan) -> None:
    Path(path).write_text(dumps(plan) + "\n", encoding="utf-8")


def read_plan(path: str | Path) -> HeadPlan:
    return loads(Path(path).read_text(encoding="utf-8"))


def verify_plan_id(plan: HeadPlan) -> bool:
    """Return whether ``plan.plan_id`` matches the plan content."""

    if not plan.plan_id:
        return False
    return plan.plan_id == compute_plan_id(plan)


def step_plan_payload(step: StepPlan, *, include_id: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": STEP_PLAN_SCHEMA,
        "nodes": [
            {"node_id": node.node_id, "plan": plan_payload(node.plan, include_id=True)}
            for node in step.nodes
        ],
        "union_rows": _token_set_to_dict(step.union_rows),
        "kernel": step.kernel,
        "cost": _cost_to_dict(step.cost),
    }
    if include_id:
        payload["plan_id"] = step.plan_id or compute_step_plan_id(step)
    return payload


def compute_step_plan_id(step: StepPlan) -> str:
    digest = hashlib.sha256(_canonical_json(step_plan_payload(step)).encode()).hexdigest()
    return f"sha256:{digest}"


def step_plan_to_dict(step: StepPlan) -> dict[str, Any]:
    return step_plan_payload(step, include_id=True)


def step_plan_from_dict(value: Any) -> StepPlan:
    if not isinstance(value, dict):
        raise PlanFormatError("step plan must be an object")
    if value.get("schema_version") != STEP_PLAN_SCHEMA:
        raise PlanFormatError(
            f"unsupported step plan schema version: {value.get('schema_version')!r}"
        )
    nodes_value = value.get("nodes")
    if not isinstance(nodes_value, list):
        raise PlanFormatError("step plan must contain a 'nodes' list")
    nodes: list[NodePlan] = []
    for item in nodes_value:
        if not isinstance(item, dict) or not isinstance(item.get("node_id"), int):
            raise PlanFormatError("invalid node plan")
        nodes.append(NodePlan(node_id=item["node_id"], plan=plan_from_dict(item.get("plan"))))
    kernel = value.get("kernel", "auto")
    if kernel not in ("auto", "full", "slice_gemv", "slice_gemm"):
        raise PlanFormatError(f"invalid step kernel: {kernel!r}")
    try:
        return StepPlan(
            nodes=tuple(nodes),
            union_rows=_token_set_from_dict(value.get("union_rows")),
            kernel=kernel,
            cost=_cost_from_dict(value.get("cost")),
            plan_id=str(value.get("plan_id", "")),
        )
    except ValueError as exc:
        raise PlanFormatError(str(exc)) from exc


__all__ = [
    "PLAN_SCHEMA",
    "STEP_PLAN_SCHEMA",
    "PlanFormatError",
    "compute_plan_id",
    "compute_step_plan_id",
    "dumps",
    "loads",
    "plan_from_dict",
    "plan_payload",
    "plan_to_dict",
    "read_plan",
    "step_plan_from_dict",
    "step_plan_payload",
    "step_plan_to_dict",
    "verify_plan_id",
    "write_plan",
]
