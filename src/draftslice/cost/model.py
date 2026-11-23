"""Roofline cost model for slice, screen, and full-head execution.

The model deliberately exposes its assumptions:

- static bytes are L2-resident when they fit, otherwise HBM;
- a slice reads dynamic rows with the gather bandwidth unless the caller marks
  the execution as fused with a contiguous layout;
- batch reuse is expressed by ``unique_rows``, not by multiplying traffic;
- time is ``launch + max(bytes time, ALU time, MMA time)``.

The scheduler uses predicted time; the benchmark suite reports prediction error.
"""

from __future__ import annotations

from ..plan.types import HeadPlan, LayoutKind, PlanCost, StepPlan
from .profiles import HardwareProfile, get_profile


class CostModelError(ValueError):
    """Raised when a plan cannot be costed."""


class CostModel:
    """A calibrated roofline model for one model shape."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        dtype_bytes: int = 2,
        profiles: dict[str, HardwareProfile] | None = None,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype_bytes = dtype_bytes
        self._profiles = dict(profiles) if profiles is not None else None

    @property
    def calibrated(self) -> bool:
        """Whether predictions may gate a production decision.

        The roofline constants are uncalibrated by default; only a
        :class:`~draftslice.cost.calibration.CalibratedCostModel` that passed its
        held-out MAPE gate returns ``True``.
        """

        return False

    def _profile(self, name_or_profile: str | HardwareProfile) -> HardwareProfile:
        if isinstance(name_or_profile, HardwareProfile):
            return name_or_profile
        if self._profiles is not None:
            try:
                return self._profiles[name_or_profile]
            except KeyError as exc:
                known = ", ".join(sorted(self._profiles))
                raise KeyError(
                    f"unknown hardware profile {name_or_profile!r}; known: {known}"
                ) from exc
        return get_profile(name_or_profile)

    def _rows(self, plan: HeadPlan) -> int:
        if plan.execution.kind == "full":
            return self.vocab_size
        if plan.execution.rows is None:
            raise CostModelError("slice plan has no execution rows")
        return len(plan.execution.rows)

    def _predict_rows(
        self,
        rows: int,
        *,
        kind: str,
        layout: LayoutKind,
        static_bytes: int,
        alu_ops: int,
        launch_count: int,
        batch: int,
        unique_rows: int | None,
        profile: str | HardwareProfile,
        fused: bool,
    ) -> PlanCost:
        if batch <= 0:
            raise ValueError("batch must be positive")
        hw = self._profile(profile)
        if unique_rows is None:
            unique_rows = min(rows * batch, self.vocab_size)
        if unique_rows < 0:
            raise ValueError("unique_rows must be non-negative")
        if unique_rows > self.vocab_size:
            raise CostModelError(f"unique_rows={unique_rows} exceeds vocab_size={self.vocab_size}")

        dynamic_bytes = unique_rows * self.hidden_size * self.dtype_bytes
        macs = unique_rows * self.hidden_size
        static_bw = hw.l2_bandwidth if static_bytes <= hw.l2_bytes else hw.hbm_bandwidth
        if kind == "full":
            dynamic_bw = hw.hbm_bandwidth
        elif fused and layout in ("leaf_contiguous", "packed"):
            dynamic_bw = hw.hbm_bandwidth
        else:
            dynamic_bw = hw.gather_bandwidth

        t_bytes = static_bytes / static_bw + dynamic_bytes / dynamic_bw
        t_alu = alu_ops / hw.alu_throughput
        t_mma = macs / hw.mma_throughput
        launches = max(1, launch_count)
        predicted_s = launches * hw.launch_overhead_s + max(t_bytes, t_alu, t_mma)

        return PlanCost(
            static_bytes=static_bytes,
            dynamic_bytes=dynamic_bytes,
            macs=macs,
            alu_ops=alu_ops,
            launch_count=launches,
            predicted_us={hw.name: predicted_s * 1e6},
        )

    def predict(
        self,
        plan: HeadPlan,
        *,
        batch: int = 1,
        unique_rows: int | None = None,
        profile: str | HardwareProfile = "a100",
        fused: bool = False,
    ) -> PlanCost:
        """Predict the cost of executing ``plan``.

        ``unique_rows`` is the number of distinct rows the kernel reads after
        batch/tree deduplication. The default is the conservative no-reuse
        estimate ``min(rows * batch, vocab_size)``. A caller measuring a shared
        union should pass the union size explicitly.
        """

        return self._predict_rows(
            self._rows(plan),
            kind=plan.execution.kind,
            layout=plan.execution.layout,
            static_bytes=plan.cost.static_bytes,
            alu_ops=plan.cost.alu_ops,
            launch_count=plan.cost.launch_count,
            batch=batch,
            unique_rows=unique_rows,
            profile=profile,
            fused=fused,
        )

    def predict_step(
        self,
        step: StepPlan,
        *,
        batch: int = 1,
        unique_rows: int | None = None,
        profile: str | HardwareProfile = "a100",
        fused: bool = False,
    ) -> PlanCost:
        """Predict the cost of one tree/step plan."""

        if step.kernel == "full":
            rows = self.vocab_size
            kind = "full"
            layout: LayoutKind = "native"
        else:
            rows = len(step.union_rows)
            kind = "slice_gemm" if step.kernel in ("slice_gemm", "auto") else "slice_gemv"
            layouts = {node.plan.execution.layout for node in step.nodes}
            layout = layouts.pop() if len(layouts) == 1 else "native"
        return self._predict_rows(
            rows,
            kind=kind,
            layout=layout,
            static_bytes=step.cost.static_bytes,
            alu_ops=step.cost.alu_ops,
            launch_count=step.cost.launch_count,
            batch=batch,
            unique_rows=unique_rows,
            profile=profile,
            fused=fused,
        )

    def predict_batches(
        self,
        plan: HeadPlan,
        batches: tuple[int, ...] = (1, 8, 32, 64),
        *,
        profile: str | HardwareProfile = "a100",
        fused: bool = False,
    ) -> dict[int, PlanCost]:
        return {
            batch: self.predict(plan, batch=batch, profile=profile, fused=fused)
            for batch in batches
        }


__all__ = ["CostModel", "CostModelError"]
