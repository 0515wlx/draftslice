"""Production plan executor.

The reference executor is intentionally tiny: one plan, one reference kernel.
This executor is the production-shaped counterpart used by the plan session:

* mixed full/slice nodes in one batch are supported;
* the kernel is selected from a :class:`KernelRegistry` (torch or Triton);
* TP shards map global token ids to this rank's weight rows;
* quantized heads dequantize only the rows a step reads;
* an optional CUDA-graph cache replays the slice kernel;
* strict coverage is checked before and after execution.

It never imports an engine. A full node needs a ``full_head`` callback because a
sliced or quantized executor cannot materialize a full vocabulary by itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..integrations.head_shard import HeadShard
from ..integrations.quant_dequant import QuantizedHead, dequantize_rows
from ..kernels.base import KernelError
from ..kernels.graph import CudaGraphCache
from ..kernels.registry import KernelRegistry
from ..kernels.torch import TorchFullKernel, TorchGatherKernel
from ..plan.types import HeadPlan, StepPlan, TokenSet


class ProductionExecutorError(KernelError):
    """The production executor cannot execute a plan safely."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Kernel output plus the execution metadata needed by callers."""

    plan_id: str
    kernel: str
    rows: TokenSet
    logits: Any

    def candidate_logits(self, plan: HeadPlan) -> Any:
        index = {token_id: position for position, token_id in enumerate(self.rows.ids)}
        missing = [token_id for token_id in plan.candidates.ids if token_id not in index]
        if missing:
            raise KernelError(f"execution rows miss candidates: {missing[:8]}")
        positions = [index[token_id] for token_id in plan.candidates.ids]
        if isinstance(self.logits, Sequence):
            return tuple(
                tuple(batch_row[position] for position in positions)
                for batch_row in self.logits
            )
        return self.logits[:, positions]


def _as_batch(hidden: Sequence[float] | Sequence[Sequence[float]]):
    if not hidden:
        raise KernelError("hidden must not be empty")
    first = hidden[0]
    if isinstance(first, (int, float)):
        return [hidden]  # type: ignore[list-item]
    return hidden


class ReferenceExecutor:
    """Execute a plan with a reference kernel.

    The executor is deliberately strict: a plan that claims exact valid-set
    coverage is rejected before the kernel runs if its execution rows do not
    cover the candidate set.
    """

    def __init__(
        self,
        weight: Sequence[Sequence[float]],
        *,
        registry: KernelRegistry | None = None,
    ) -> None:
        if not weight:
            raise ValueError("weight must not be empty")
        self.weight = weight
        self.registry = registry or KernelRegistry()

    def execute(
        self,
        hidden: Sequence[float] | Sequence[Sequence[float]],
        plan: HeadPlan,
    ) -> ExecutionResult:
        if plan.guarantee.kind.value == "exact_valid_set" and not plan.execution.covers(
            plan.candidates
        ):
            raise KernelError("strict plan does not cover its candidates")

        batch = _as_batch(hidden)
        kernel = self.registry.select(plan, len(batch))
        result = kernel.run(batch, plan, self.weight)

        if plan.guarantee.kind.value == "exact_valid_set" and not result.rows.covers(
            plan.candidates
        ):
            raise KernelError("kernel result does not cover the strict candidates")

        return ExecutionResult(
            plan_id=plan.plan_id,
            kernel=kernel.name,
            rows=result.rows,
            logits=result.logits,
        )


@dataclass(frozen=True, slots=True)
class ProductionNodeResult:
    """One draft-tree node's logits.

    ``token_ids`` are global token ids and columns of ``logits`` match them.
    ``rows`` are the kernel-space rows (local shard rows or dequantized-row
    positions) that were read.
    """

    node_id: int
    token_ids: tuple[int, ...]
    logits: Any
    rows: TokenSet
    kind: str

    @property
    def full(self) -> bool:
        return self.kind == "full"


@dataclass(frozen=True, slots=True)
class ProductionResult:
    """Result of one mixed full/slice step."""

    step_plan: StepPlan
    nodes: tuple[ProductionNodeResult, ...]
    kernel: str
    full_nodes: int
    slice_nodes: int

    @property
    def node_logits(self) -> tuple[Any, ...]:
        return tuple(node.logits for node in self.nodes)

    @property
    def execution(self) -> str:
        if self.slice_nodes == 0:
            return "full"
        if self.full_nodes == 0:
            return "slice"
        return "mixed"


FullHead = Callable[[Any], Any]


def default_registry(*, use_triton: bool = True) -> KernelRegistry:
    """The torch/Triton registry used in production tests and hosts."""

    kernels: list[Any] = [TorchFullKernel(), TorchGatherKernel()]
    if use_triton:
        try:  # pragma: no cover - optional dependency
            import torch

            from ..kernels.triton import TritonSliceKernel

            if torch.cuda.is_available():
                kernels.insert(1, TritonSliceKernel())
        except Exception:  # pragma: no cover - triton absent/broken
            pass
    return KernelRegistry(kernels)


class ProductionExecutor:
    """Execute full, strict-slice, and mixed step plans."""

    def __init__(
        self,
        weight: Any | None,
        *,
        full_head: FullHead | None = None,
        registry: KernelRegistry | None = None,
        shard: HeadShard | None = None,
        quant_head: QuantizedHead | None = None,
        graph_cache: CudaGraphCache | None = None,
        use_triton: bool = True,
    ) -> None:
        if weight is None and quant_head is None:
            raise ProductionExecutorError("a dense weight or a quantized head is required")
        self.weight = weight
        self.full_head = full_head
        self.registry = registry or default_registry(use_triton=use_triton)
        self.shard = shard
        self.quant_head = quant_head
        self.graph_cache = graph_cache

    # -- public API -------------------------------------------------------
    def execute(
        self,
        hidden: Any,
        plan: HeadPlan,
        *,
        full_head: FullHead | None = None,
    ) -> ProductionNodeResult:
        from ..plan.types import NodePlan

        step = StepPlan(
            nodes=(NodePlan(node_id=0, plan=plan),),
            union_rows=TokenSet() if plan.execution.kind == "full" else (
                plan.execution.rows or TokenSet()
            ),
            kernel="full" if plan.execution.kind == "full" else "slice_gemv",
        )
        result = self.execute_step(hidden, step, full_head=full_head)
        return result.nodes[0]

    def execute_step(
        self,
        hidden: Any,
        step: StepPlan,
        *,
        full_head: FullHead | None = None,
    ) -> ProductionResult:
        if not step.nodes:
            raise ProductionExecutorError("step plan has no nodes")
        batch = _batch_size(hidden)
        if batch != len(step.nodes):
            raise ProductionExecutorError(
                f"hidden has {batch} rows but the step has {len(step.nodes)} nodes"
            )
        self._validate_strict(step)

        full_nodes = [
            node for node in step.nodes if node.plan.execution.kind == "full"
        ]
        slice_nodes = [
            node for node in step.nodes if node.plan.execution.kind != "full"
        ]
        results: dict[int, ProductionNodeResult] = {}

        if full_nodes:
            callback = full_head or self.full_head
            if callback is None:
                raise ProductionExecutorError(
                    "a full node requires a full_head callback"
                )
            indices = [node.node_id for node in full_nodes]
            full_logits = callback(_select_rows(hidden, indices))
            vocab = int(getattr(full_logits, "shape", (0, 0))[-1])
            if vocab <= 0:
                raise ProductionExecutorError("full head returned no vocabulary rows")
            for offset, node in enumerate(full_nodes):
                ids = node.plan.candidates.ids
                bad = [token for token in ids if token >= vocab]
                if bad:
                    raise ProductionExecutorError(
                        f"candidate token {bad[0]} is outside the full head"
                    )
                results[node.node_id] = ProductionNodeResult(
                    node_id=node.node_id,
                    token_ids=ids,
                    logits=_columns(full_logits, offset, ids),
                    rows=TokenSet(range(vocab)),
                    kind="full",
                )

        kernel_name = ""
        if slice_nodes:
            kernel_name, slice_results = self._execute_slice_nodes(hidden, slice_nodes)
            results.update(slice_results)

        ordered = tuple(results[node.node_id] for node in step.nodes)
        return ProductionResult(
            step_plan=step,
            nodes=ordered,
            kernel=kernel_name or "torch_full",
            full_nodes=len(full_nodes),
            slice_nodes=len(slice_nodes),
        )

    # -- slice execution --------------------------------------------------
    def _execute_slice_nodes(
        self, hidden: Any, nodes: Sequence[Any]
    ) -> tuple[str, dict[int, ProductionNodeResult]]:
        global_union = TokenSet(
            token_id
            for node in nodes
            for token_id in (node.plan.execution.rows or node.plan.candidates).ids
        )
        if global_union.is_empty:
            return "", {
                node.node_id: ProductionNodeResult(
                    node_id=node.node_id,
                    token_ids=node.plan.candidates.ids,
                    logits=_empty_like(hidden, len(node.plan.candidates)),
                    rows=TokenSet(),
                    kind="slice",
                )
                for node in nodes
            }

        physical, mapping = self._row_mapping(global_union)
        weight = self.weight
        if self.quant_head is not None:
            rows = list(physical.ids)
            weight = dequantize_rows(self.quant_head, rows)
            if weight is None:
                raise ProductionExecutorError(
                    "cannot dequantize the rows a slice step reads"
                )
            kernel_rows = {
                global_id: position for position, global_id in enumerate(rows)
            }
        else:
            kernel_rows = mapping

        union_rows = TokenSet(kernel_rows.values())
        plans: list[HeadPlan] = []
        node_ids: list[int] = []
        for node in nodes:
            plan = node.plan
            try:
                candidates = TokenSet(kernel_rows[token] for token in plan.candidates.ids)
                rows = TokenSet(
                    kernel_rows[token] for token in (plan.execution.rows or plan.candidates).ids
                )
            except KeyError as exc:
                raise ProductionExecutorError(
                    f"node {node.node_id} row is not owned by this rank/shard"
                ) from exc
            plans.append(
                HeadPlan(
                    candidates=candidates,
                    execution=_execution(plan, rows),
                    guarantee=plan.guarantee,
                    state_id=plan.state_id,
                )
            )
            node_ids.append(node.node_id)

        if union_rows.is_empty:
            return "", {}
        batch = len(nodes)
        slice_indices = [node.node_id for node in nodes]
        hidden_rows = _select_rows(hidden, slice_indices)
        kernel = self._select_kernel(plans[0], batch, hidden_rows)
        row_tensor = _rows_tensor(union_rows, hidden_rows)
        if self.graph_cache is not None:
            logits = self.graph_cache.run(
                (
                    "production",
                    getattr(kernel, "name", "slice"),
                    _weight_id(weight),
                    batch,
                    _hidden_width(hidden),
                    str(getattr(hidden, "dtype", "")),
                    len(union_rows),
                ),
                lambda static_hidden, static_rows: kernel.run_tensor(
                    static_hidden, weight, static_rows
                ),
                hidden_rows,
                row_tensor,
            )
        else:
            logits = kernel.run_tensor(hidden_rows, weight, union_rows)

        out: dict[int, ProductionNodeResult] = {}
        for offset, (node, local_plan) in enumerate(zip(nodes, plans, strict=True)):
            if not union_rows.covers(local_plan.candidates):
                raise ProductionExecutorError(
                    f"kernel rows miss node {node.node_id}'s candidates"
                )
            index = {token: position for position, token in enumerate(union_rows.ids)}
            positions = [index[token] for token in local_plan.candidates.ids]
            selected = (
                logits[offset : offset + 1, positions]
                if hasattr(logits, "shape")
                else tuple(
                    tuple(logits[offset][position] for position in positions),
                )
            )
            out[node.node_id] = ProductionNodeResult(
                node_id=node.node_id,
                token_ids=node.plan.candidates.ids,
                logits=selected,
                rows=union_rows,
                kind="slice",
            )
        return getattr(kernel, "name", "slice"), out

    def _select_kernel(self, plan: HeadPlan, batch: int, hidden: Any) -> Any:
        kernels = self.registry.kernels
        if getattr(getattr(hidden, "device", None), "type", "") != "cuda":
            kernels = tuple(
                kernel
                for kernel in kernels
                if "triton" not in getattr(kernel, "name", "").lower()
            )
        if not kernels:
            raise ProductionExecutorError("no CPU-compatible slice kernel")
        for kernel in kernels:
            if kernel.supports(plan, batch):
                return kernel
        raise ProductionExecutorError(
            f"no kernel supports plan {plan.plan_id or '<anonymous>'}"
        )

    def _row_mapping(self, global_union: TokenSet) -> tuple[TokenSet, dict[int, int]]:
        """Map global ids to physical weight rows (or local shard rows)."""

        if self.shard is None:
            return global_union, {token: token for token in global_union.ids}
        physical: list[int] = []
        mapping: dict[int, int] = {}
        for token in global_union.ids:
            row = self.shard.local_row(token)
            if row is None or row >= self.shard.shard_rows:
                raise ProductionExecutorError(
                    f"token {token} is not owned by tp_rank {self.shard.tp_rank}"
                )
            physical.append(row)
            mapping[token] = row
        return TokenSet(physical), mapping

    # -- validation -------------------------------------------------------
    @staticmethod
    def _validate_strict(step: StepPlan) -> None:
        if step.kernel == "full":
            return
        for node in step.nodes:
            plan = node.plan
            if plan.guarantee.kind.value != "exact_valid_set":
                continue
            if not plan.execution.covers(plan.candidates):
                raise ProductionExecutorError(
                    f"node {node.node_id} execution rows miss its strict candidates"
                )
            if not plan.candidates.issubset(step.union_rows):
                raise ProductionExecutorError(
                    f"node {node.node_id} is not covered by the step union"
                )


def _execution(plan: HeadPlan, rows: TokenSet) -> Any:
    from ..plan.types import Execution

    return Execution(
        kind="slice_gemv" if plan.execution.kind == "full" else plan.execution.kind,
        rows=rows,
        layout=plan.execution.layout,
    )


def _batch_size(hidden: Any) -> int:
    shape = getattr(hidden, "shape", None)
    if shape is None or len(shape) != 2:
        raise ProductionExecutorError("hidden must be a 2-D tensor")
    return int(shape[0])


def _hidden_width(hidden: Any) -> int:
    return int(getattr(hidden, "shape", (0, 0))[1])


def _weight_id(weight: Any) -> int:
    pointer = getattr(weight, "data_ptr", None)
    if callable(pointer):
        try:
            return int(pointer())
        except Exception:  # pragma: no cover - defensive
            return 0
    return id(weight)


def _select_rows(hidden: Any, indices: Sequence[int]) -> Any:
    if len(indices) == _batch_size(hidden) and list(indices) == list(range(len(indices))):
        return hidden
    index = _index_tensor(indices, hidden)
    return hidden.index_select(0, index)


def _columns(logits: Any, row: int, ids: Sequence[int]) -> Any:
    if hasattr(logits, "shape"):
        index = _index_tensor(ids, logits)
        return logits[row : row + 1, index]
    return tuple(logits[row][token] for token in ids)


def _rows_tensor(rows: TokenSet, hidden: Any) -> Any:
    return _index_tensor(rows.ids, hidden)


def _index_tensor(values: Sequence[int], like: Any) -> Any:
    try:
        import torch

        return torch.tensor(list(values), dtype=torch.long, device=like.device)
    except ImportError as exc:  # pragma: no cover - torch optional
        raise ProductionExecutorError("torch is required for tensor plans") from exc


def _empty_like(hidden: Any, width: int) -> Any:
    if hasattr(hidden, "new_empty"):
        return hidden.new_empty((1, width))
    return tuple()


__all__ = [
    "ExecutionResult",
    "FullHead",
    "ProductionExecutor",
    "ProductionExecutorError",
    "ProductionNodeResult",
    "ProductionResult",
    "ReferenceExecutor",
    "default_registry",
]
