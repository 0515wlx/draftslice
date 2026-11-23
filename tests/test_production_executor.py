from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.head_shard import HeadShard  # noqa: E402
from draftslice.integrations.quant_dequant import (  # noqa: E402
    QuantizedHead,
    dequantize_rows,
)
from draftslice.plan import (  # noqa: E402
    Execution,
    Guarantee,
    HeadPlan,
    NodePlan,
    StepPlan,
    TokenSet,
)
from draftslice.runtime import ProductionExecutor, ProductionExecutorError  # noqa: E402


def slice_plan(rows, candidates=None) -> HeadPlan:
    rows = TokenSet(rows)
    return HeadPlan(
        candidates=TokenSet(candidates) if candidates is not None else rows,
        execution=Execution(kind="slice_gemv", rows=rows),
        guarantee=Guarantee.exact_valid_set(),
        state_id=0,
    )


def full_plan(candidates=()) -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet(candidates),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
    )


def step(*nodes: tuple[int, HeadPlan]) -> StepPlan:
    slice_rows: list[int] = []
    for _, plan in nodes:
        if plan.execution.rows is not None:
            slice_rows.extend(plan.execution.rows.ids)
    return StepPlan(
        nodes=tuple(NodePlan(node_id=index, plan=plan) for index, plan in nodes),
        union_rows=TokenSet(slice_rows),
        kernel="full" if all(plan.execution.kind == "full" for _, plan in nodes) else "slice_gemm",
    )


def test_mixed_full_and_slice_matches_direct_path() -> None:
    torch.manual_seed(0)
    weight = torch.randn(32, 8)
    hidden = torch.randn(2, 8)
    executor = ProductionExecutor(weight, full_head=lambda h: h @ weight.T)
    plan = step(
        (0, slice_plan([1, 3, 5])),
        (1, full_plan([2, 4])),
    )
    result = executor.execute_step(hidden, plan)
    assert result.execution == "mixed"
    torch.testing.assert_close(result.nodes[0].logits, hidden[0:1] @ weight[[1, 3, 5]].T)
    torch.testing.assert_close(result.nodes[1].logits, hidden[1:2] @ weight[[2, 4]].T)


def test_mixed_full_first_then_slice_uses_the_right_rows() -> None:
    torch.manual_seed(5)
    weight = torch.randn(32, 8)
    hidden = torch.randn(2, 8)
    executor = ProductionExecutor(weight, full_head=lambda h: h @ weight.T)
    plan = step(
        (0, full_plan([2, 4])),
        (1, slice_plan([1, 3, 5])),
    )
    result = executor.execute_step(hidden, plan)
    torch.testing.assert_close(result.nodes[0].logits, hidden[0:1] @ weight[[2, 4]].T)
    torch.testing.assert_close(result.nodes[1].logits, hidden[1:2] @ weight[[1, 3, 5]].T)


def test_tree_union_scatter_parity() -> None:
    torch.manual_seed(1)
    weight = torch.randn(32, 8)
    hidden = torch.randn(3, 8)
    executor = ProductionExecutor(weight)
    plan = step(
        (0, slice_plan([1, 2, 9])),
        (1, slice_plan([2, 9, 17])),
        (2, slice_plan([17, 25])),
    )
    result = executor.execute_step(hidden, plan)
    assert result.execution == "slice"
    for row, node in enumerate(result.nodes):
        expected = hidden[row : row + 1] @ weight[list(node.token_ids)].T
        torch.testing.assert_close(node.logits, expected)


def test_full_node_requires_callback() -> None:
    weight = torch.randn(16, 8)
    executor = ProductionExecutor(weight)
    plan = step((0, full_plan([1])))
    with pytest.raises(ProductionExecutorError):
        executor.execute_step(torch.randn(1, 8), plan)


def test_tp_shard_maps_global_ids_to_local_rows() -> None:
    torch.manual_seed(2)
    weight = torch.randn(8, 4)
    hidden = torch.randn(1, 4)
    shard = HeadShard(
        tp_size=2,
        tp_rank=1,
        vocab_start=8,
        vocab_end=16,
        shard_rows=8,
        org_vocab_size=16,
        padded_vocab_size=16,
    )
    executor = ProductionExecutor(weight, shard=shard)
    result = executor.execute(hidden, slice_plan([9, 11]))
    assert result.token_ids == (9, 11)
    torch.testing.assert_close(result.logits, hidden @ weight[[1, 3]].T)


def test_tp_shard_rejects_unowned_rows() -> None:
    weight = torch.randn(8, 4)
    shard = HeadShard(
        tp_size=2,
        tp_rank=1,
        vocab_start=8,
        vocab_end=16,
        shard_rows=8,
        org_vocab_size=16,
        padded_vocab_size=16,
    )
    executor = ProductionExecutor(weight, shard=shard)
    with pytest.raises(ProductionExecutorError):
        executor.execute(torch.randn(1, 4), slice_plan([1, 2]))


def test_quantized_rows_are_dequantized_for_the_slice() -> None:
    torch.manual_seed(3)
    weight = torch.randint(-8, 8, (16, 4), dtype=torch.int8)
    scale = torch.rand(16) + 0.5
    head = QuantizedHead(
        weight=weight,
        out_features=16,
        in_features=4,
        transposed=False,
        scale=scale,
        scale_inv=None,
        out_dtype=torch.bfloat16,
    )
    hidden = torch.randn(1, 4, dtype=torch.bfloat16)
    executor = ProductionExecutor(None, quant_head=head)
    result = executor.execute(hidden, slice_plan([2, 5, 7]))
    expected = dequantize_rows(head, [2, 5, 7])
    torch.testing.assert_close(result.logits, hidden @ expected.T)


def test_strict_coverage_is_checked_before_execution() -> None:
    weight = torch.randn(32, 8)
    plan = slice_plan([1, 2])
    step_plan = step((0, plan))
    assert step_plan.nodes[0].plan.execution.covers(step_plan.nodes[0].plan.candidates)
    object.__setattr__(plan, "candidates", TokenSet([1, 2, 3]))
    executor = ProductionExecutor(weight)
    with pytest.raises(ProductionExecutorError):
        executor.execute_step(torch.randn(1, 8), step_plan)


def test_execute_single_plan() -> None:
    torch.manual_seed(4)
    weight = torch.randn(16, 4)
    hidden = torch.randn(1, 4)
    executor = ProductionExecutor(weight)
    result = executor.execute(hidden, slice_plan([1, 3]))
    torch.testing.assert_close(result.logits, hidden @ weight[[1, 3]].T)
