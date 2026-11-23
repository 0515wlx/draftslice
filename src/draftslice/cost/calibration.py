"""Cost-model calibration (B2) and the calibrated-model gate.

The roofline model in :mod:`draftslice.cost.model` is a starting point, not a
measured claim. This module fits one multiplicative correction per execution
kind from B2 measurements, evaluates MAPE on held-out shapes, and marks the
result unusable above the gate (25% by default). An uncalibrated model must
never choose a production slice; :class:`~draftslice.runtime.scheduler.StepScheduler`
defaults to ``full`` in that case.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..plan.types import PlanCost
from .model import CostModel

SCHEMA = "cost-calibration/1"
KINDS = ("full", "slice_gemv", "slice_gemm")


class CalibrationError(ValueError):
    """A calibration artifact is malformed or cannot be fitted."""


@dataclass(frozen=True, slots=True)
class CostSample:
    """One measured plan/shape pair."""

    kind: str
    batch: int
    rows: int
    predicted_us: float
    measured_us: float
    profile: str = "a100"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise CalibrationError(f"invalid kind: {self.kind!r}")
        if self.batch <= 0 or self.rows <= 0:
            raise CalibrationError("batch and rows must be positive")
        if self.predicted_us <= 0 or self.measured_us <= 0:
            raise CalibrationError("times must be positive")

    @property
    def ratio(self) -> float:
        return self.measured_us / self.predicted_us

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "batch": self.batch,
            "rows": self.rows,
            "predicted_us": self.predicted_us,
            "measured_us": self.measured_us,
            "profile": self.profile,
        }


@dataclass(frozen=True, slots=True)
class CostCalibration:
    """Per-kind correction factors plus their held-out error."""

    profile: str
    factors: dict[str, float]
    mape: float
    holdout_mape: float
    calibrated: bool
    gate: float = 0.25
    samples: int = 0
    holdout_samples: int = 0
    created: str = ""
    engine: str = "torch"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.profile:
            raise CalibrationError("profile must not be empty")
        if not (0.0 < self.gate <= 1.0):
            raise CalibrationError("gate must be in (0, 1]")
        if self.mape < 0 or self.holdout_mape < 0:
            raise CalibrationError("MAPE must be non-negative")
        for kind, factor in self.factors.items():
            if kind not in (*KINDS, "default"):
                raise CalibrationError(f"invalid factor kind: {kind!r}")
            if factor <= 0:
                raise CalibrationError("factors must be positive")
        if self.calibrated and self.holdout_mape > self.gate:
            raise CalibrationError(
                "calibrated=true but holdout MAPE exceeds the gate"
            )

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        samples: list[CostSample] | tuple[CostSample, ...],
        *,
        profile: str = "a100",
        gate: float = 0.25,
        holdout_fraction: float = 0.3,
        engine: str = "torch",
        notes: str = "",
    ) -> CostCalibration:
        if not (0.0 <= holdout_fraction < 1.0):
            raise CalibrationError("holdout_fraction must be in [0, 1)")
        if not samples:
            raise CalibrationError("no calibration samples")
        train, holdout = _split(samples, holdout_fraction)
        factors = _factors(train)
        train_mape = _mape(train, factors)
        holdout_mape = _mape(holdout or train, factors) if holdout else train_mape
        calibrated = holdout_mape <= gate and set(factors) >= {"full", "slice_gemv"}
        return cls(
            profile=profile,
            factors=factors,
            mape=train_mape,
            holdout_mape=holdout_mape,
            calibrated=calibrated,
            gate=gate,
            samples=len(train),
            holdout_samples=len(holdout),
            created=_now(),
            engine=engine,
            notes=notes,
        )

    # -- application ------------------------------------------------------
    def factor(self, kind: str) -> float:
        return float(self.factors.get(kind, self.factors.get("default", 1.0)))

    def apply(self, kind: str, predicted_us: float) -> float:
        return float(predicted_us) * self.factor(kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "profile": self.profile,
            "factors": dict(sorted(self.factors.items())),
            "mape": self.mape,
            "holdout_mape": self.holdout_mape,
            "calibrated": self.calibrated,
            "gate": self.gate,
            "samples": self.samples,
            "holdout_samples": self.holdout_samples,
            "created": self.created,
            "engine": self.engine,
            "notes": self.notes,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True, ensure_ascii=True
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self.to_dict()
        document["hash"] = self.digest()
        target.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return document["hash"]

    @classmethod
    def load(cls, path: str | Path) -> CostCalibration:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CalibrationError(f"cannot read calibration {source}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"invalid calibration JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise CalibrationError("unsupported calibration document")
        if data.get("schema") == "cost-bench/1":
            # ``draftslice bench cost`` writes a report that embeds the
            # calibration document. Accept it directly so the documented
            # DRAFTSLICE_COST_CALIBRATION artifact works without a manual
            # unwrap step.
            nested = data.get("calibration")
            if not isinstance(nested, dict):
                raise CalibrationError("cost-bench report has no calibration")
            data = dict(nested)
        if data.get("schema") != SCHEMA:
            raise CalibrationError("unsupported calibration document")
        stored = data.pop("hash", None)
        calibration = cls(
            profile=str(data["profile"]),
            factors={str(k): float(v) for k, v in data["factors"].items()},
            mape=float(data["mape"]),
            holdout_mape=float(data["holdout_mape"]),
            calibrated=bool(data["calibrated"]),
            gate=float(data.get("gate", 0.25)),
            samples=int(data.get("samples", 0)),
            holdout_samples=int(data.get("holdout_samples", 0)),
            created=str(data.get("created", "")),
            engine=str(data.get("engine", "torch")),
            notes=str(data.get("notes", "")),
        )
        if stored is not None and stored != calibration.digest():
            raise CalibrationError("calibration hash mismatch")
        return calibration


class CalibratedCostModel(CostModel):
    """A :class:`CostModel` with per-kind calibration factors applied."""

    def __init__(self, *args: Any, calibration: CostCalibration, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calibration = calibration

    @property
    def calibrated(self) -> bool:
        return self.calibration.calibrated

    def predict(self, plan: Any, **kwargs: Any) -> PlanCost:
        cost = super().predict(plan, **kwargs)
        return self._apply(
            cost,
            "full" if plan.execution.kind == "full" else (
                "slice_gemm" if plan.execution.kind == "slice_gemm" else "slice_gemv"
            ),
        )

    def predict_step(self, step: Any, **kwargs: Any) -> PlanCost:
        cost = super().predict_step(step, **kwargs)
        if step.kernel == "full":
            kind = "full"
        else:
            kind = "slice_gemm" if step.kernel in ("slice_gemm", "auto") else "slice_gemv"
        return self._apply(cost, kind)

    def _apply(self, cost: PlanCost, kind: str) -> PlanCost:
        return replace(
            cost,
            predicted_us={
                name: value * self.calibration.factor(kind)
                for name, value in cost.predicted_us.items()
            },
        )


def _split(
    samples: list[CostSample] | tuple[CostSample, ...], holdout_fraction: float
) -> tuple[list[CostSample], list[CostSample]]:
    if holdout_fraction <= 0:
        return list(samples), []
    train: list[CostSample] = []
    holdout: list[CostSample] = []
    for sample in samples:
        key = f"{sample.kind}|{sample.batch}|{sample.rows}|{sample.profile}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") / 2**32
        (holdout if bucket < holdout_fraction else train).append(sample)
    if not train:
        train, holdout = list(samples), []
    return train, holdout


def _factors(samples: list[CostSample]) -> dict[str, float]:
    ratios: dict[str, list[float]] = {}
    for sample in samples:
        ratios.setdefault(sample.kind, []).append(sample.ratio)
    factors = {kind: statistics.median(values) for kind, values in ratios.items()}
    all_ratios = [sample.ratio for sample in samples]
    factors["default"] = statistics.median(all_ratios)
    return factors


def _mape(samples: list[CostSample], factors: dict[str, float]) -> float:
    errors = []
    for sample in samples:
        predicted = sample.predicted_us * factors.get(
            sample.kind, factors.get("default", 1.0)
        )
        errors.append(abs(predicted - sample.measured_us) / sample.measured_us)
    return statistics.fmean(errors) if errors else 0.0


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "CalibratedCostModel",
    "CalibrationError",
    "CostCalibration",
    "CostSample",
    "KINDS",
    "SCHEMA",
]
