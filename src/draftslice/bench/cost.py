"""B2 cost-model calibration runner.

Measures the torch full/gather kernels on A100, compares them with the roofline
prediction, fits per-kind correction factors, and reports held-out MAPE. The
result is a JSON artifact; a gate failure leaves ``calibrated=false`` so the
production scheduler keeps defaulting to the full head.

Run:

```bash
python -m draftslice.bench.cost --out runs/cost_model_a100_20261122.json
```
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..cost.calibration import CostCalibration, CostSample
from ..cost.model import CostModel
from ..plan.types import Execution, Guarantee, HeadPlan, PlanCost, TokenSet
from .gpu import query_gpu


@dataclass(frozen=True, slots=True)
class CostBenchConfig:
    vocab_size: int = 151_936
    hidden_size: int = 2_560
    batches: tuple[int, ...] = (1, 8)
    train_rows: tuple[int, ...] = (64, 256, 1_024, 4_096)
    holdout_rows: tuple[int, ...] = (128, 2_048, 8_192)
    dtype: str = "bf16"
    warmup: int = 10
    iterations: int = 30
    profile: str = "a100"
    gate: float = 0.25
    device: str = "cuda:0"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        if not self.batches or any(batch <= 0 for batch in self.batches):
            raise ValueError("batches must be positive")
        if any(rows <= 0 or rows > self.vocab_size for rows in self.train_rows):
            raise ValueError("train_rows must be within the vocabulary")
        if any(rows <= 0 or rows > self.vocab_size for rows in self.holdout_rows):
            raise ValueError("holdout_rows must be within the vocabulary")
        if not (0.0 < self.gate <= 1.0):
            raise ValueError("gate must be in (0, 1]")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - engine optional
        raise RuntimeError("torch is required for B2 calibration") from exc
    return torch


def _dtype(name: str) -> Any:
    torch = _torch()
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _measure(fn: Any, *, warmup: int, iterations: int) -> float:
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def _full_plan(vocab_size: int, hidden_size: int) -> HeadPlan:
    return HeadPlan(
        candidates=TokenSet(range(vocab_size)),
        execution=Execution(kind="full"),
        guarantee=Guarantee.full(),
        cost=PlanCost.dense(vocab_size, hidden_size),
    )


def _slice_plan(rows: int, hidden_size: int) -> HeadPlan:
    tokens = TokenSet(range(rows))
    return HeadPlan(
        candidates=tokens,
        execution=Execution(kind="slice_gemv", rows=tokens),
        guarantee=Guarantee.exact_valid_set(),
        cost=PlanCost.dense(rows, hidden_size),
    )


def run(config: CostBenchConfig) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B2 calibration")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dtype = _dtype(config.dtype)
    weight = torch.randn(
        (config.vocab_size, config.hidden_size), device=device, dtype=dtype
    )
    from ..kernels.torch import TorchFullKernel, TorchGatherKernel

    full_kernel = TorchFullKernel()
    slice_kernel = TorchGatherKernel()
    model = CostModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        dtype_bytes=2 if config.dtype != "fp32" else 4,
    )
    full_plan = _full_plan(config.vocab_size, config.hidden_size)

    records: list[dict[str, Any]] = []
    samples: list[CostSample] = []

    full_hidden = torch.randn((1, config.hidden_size), device=device, dtype=dtype)
    full_measured = _measure(
        lambda: full_kernel.run_tensor(full_hidden, weight),
        warmup=config.warmup,
        iterations=config.iterations,
    )
    full_predicted = model.predict(full_plan, batch=1, profile=config.profile).predicted_us[
        config.profile
    ]
    samples.append(
        CostSample(
            kind="full",
            batch=1,
            rows=config.vocab_size,
            predicted_us=full_predicted,
            measured_us=full_measured,
            profile=config.profile,
        )
    )
    records.append(
        {
            "kind": "full",
            "batch": 1,
            "rows": config.vocab_size,
            "predicted_us": full_predicted,
            "measured_us": full_measured,
            "source": "torch_full",
        }
    )

    for rows in (*config.train_rows, *config.holdout_rows):
        plan = _slice_plan(rows, config.hidden_size)
        for batch in config.batches:
            hidden = torch.randn((batch, config.hidden_size), device=device, dtype=dtype)
            measured = _measure(
                lambda h=hidden, p=plan: slice_kernel.run_tensor(
                    h, weight, p.execution.rows
                ),
                warmup=config.warmup,
                iterations=config.iterations,
            )
            # The gather kernel reads each selected weight row once and reuses
            # it for the whole batch; do not count the rows once per batch row.
            predicted = model.predict(
                plan, batch=batch, unique_rows=rows, profile=config.profile
            ).predicted_us[config.profile]
            kind = "slice_gemv" if batch == 1 else "slice_gemm"
            samples.append(
                CostSample(
                    kind=kind,
                    batch=batch,
                    rows=rows,
                    predicted_us=predicted,
                    measured_us=measured,
                    profile=config.profile,
                )
            )
            records.append(
                {
                    "kind": kind,
                    "batch": batch,
                    "rows": rows,
                    "predicted_us": predicted,
                    "measured_us": measured,
                    "source": "torch_gather",
                }
            )

    # Split exactly by the declared train/holdout shapes so the reported gate is
    # on unseen shapes, not a random subset of the same rows.
    train = [
        sample
        for sample in samples
        if sample.kind == "full" or sample.rows in config.train_rows
    ]
    holdout = [
        sample
        for sample in samples
        if sample.kind != "full" and sample.rows in config.holdout_rows
    ]
    calibration = CostCalibration.fit(
        train,
        profile=config.profile,
        gate=config.gate,
        holdout_fraction=0.0,
        engine="torch",
        notes="B2 A100 full/gather; holdout rows are separate shapes",
    )
    if holdout:
        from ..cost.calibration import _mape

        holdout_mape = _mape(holdout, calibration.factors)
        calibration = CostCalibration(
            profile=calibration.profile,
            factors=calibration.factors,
            mape=calibration.mape,
            holdout_mape=holdout_mape,
            calibrated=holdout_mape <= calibration.gate,
            gate=calibration.gate,
            samples=calibration.samples,
            holdout_samples=len(holdout),
            created=calibration.created,
            engine=calibration.engine,
            notes=calibration.notes,
        )

    try:
        gpu = asdict(query_gpu(0))
    except Exception:  # pragma: no cover - no nvidia-smi
        gpu = None
    return {
        "schema": "cost-bench/1",
        "config": {
            **asdict(config),
            "batches": list(config.batches),
            "train_rows": list(config.train_rows),
            "holdout_rows": list(config.holdout_rows),
        },
        "gpu": gpu,
        "created_unix": time.time(),
        "calibration": calibration.to_dict(),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the B2 cost calibration.")
    parser.add_argument("--out", type=Path, default=Path("runs/cost_model_a100.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--gate", type=float, default=0.25)
    args = parser.parse_args(argv)
    config = CostBenchConfig(
        device=args.device, iterations=args.iterations, gate=args.gate
    )
    result = run(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calibration = result["calibration"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "calibrated": calibration["calibrated"],
                "mape": round(calibration["mape"], 4),
                "holdout_mape": round(calibration["holdout_mape"], 4),
                "factors": calibration["factors"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["CostBenchConfig", "main", "run"]
