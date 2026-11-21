"""GPU clock control and environment capture for performance runs.

Performance numbers are only meaningful when the SM clock is fixed. This module
queries the device, locks the SM clock with ``nvidia-smi -lgc``, verifies the
clock after locking, and always unlocks on context exit.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any


class GpuUnavailable(RuntimeError):
    """Raised when nvidia-smi or the requested GPU is unavailable."""


class ClockLockError(RuntimeError):
    """Raised when the requested clock could not be locked and verified."""


_FIELDS = (
    "index",
    "name",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.mem",
    "clocks.max.mem",
    "driver_version",
    "temperature.gpu",
    "power.draw",
    "pstate",
)


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    index: int
    name: str
    sm_clock_mhz: int
    max_sm_clock_mhz: int
    mem_clock_mhz: int | None
    max_mem_clock_mhz: int | None
    driver_version: str
    temperature_c: float | None
    power_w: float | None
    pstate: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "sm_clock_mhz": self.sm_clock_mhz,
            "max_sm_clock_mhz": self.max_sm_clock_mhz,
            "mem_clock_mhz": self.mem_clock_mhz,
            "max_mem_clock_mhz": self.max_mem_clock_mhz,
            "driver_version": self.driver_version,
            "temperature_c": self.temperature_c,
            "power_w": self.power_w,
            "pstate": self.pstate,
        }


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise GpuUnavailable("nvidia-smi is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise GpuUnavailable(f"command timed out: {' '.join(command)}") from exc


def _optional_int(value: str) -> int | None:
    text = value.strip()
    if not text or text.upper() in {"N/A", "[N/A]", "UNKNOWN"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _optional_float(value: str) -> float | None:
    text = value.strip()
    if not text or text.upper() in {"N/A", "[N/A]", "UNKNOWN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def query_gpu(index: int = 0) -> GpuSnapshot:
    """Return a snapshot of one GPU via nvidia-smi."""

    if index < 0:
        raise ValueError("GPU index must be non-negative")
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]
    result = _run(command)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed"
        raise GpuUnavailable(message)
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(_FIELDS):
        raise GpuUnavailable(f"unexpected nvidia-smi output: {line!r}")

    sm = _optional_int(parts[2])
    max_sm = _optional_int(parts[3])
    if sm is None or max_sm is None:
        raise GpuUnavailable(f"nvidia-smi did not report SM clocks: {line!r}")
    return GpuSnapshot(
        index=int(parts[0]),
        name=parts[1],
        sm_clock_mhz=sm,
        max_sm_clock_mhz=max_sm,
        mem_clock_mhz=_optional_int(parts[4]),
        max_mem_clock_mhz=_optional_int(parts[5]),
        driver_version=parts[6],
        temperature_c=_optional_float(parts[7]),
        power_w=_optional_float(parts[8]),
        pstate=parts[9] or None,
    )


class GpuClockController:
    """Lock one GPU to a fixed SM clock and verify it."""

    def __init__(
        self,
        index: int = 0,
        sm_clock_mhz: int | None = None,
        *,
        lock_memory: bool = False,
    ) -> None:
        self.index = index
        self.requested_sm_clock_mhz = sm_clock_mhz
        self.lock_memory = lock_memory
        self.target_sm_clock_mhz: int | None = None
        self._snapshot: GpuSnapshot | None = None
        self._locked = False
        self._warnings: list[str] = []

    @property
    def snapshot(self) -> GpuSnapshot:
        if self._snapshot is None:
            raise RuntimeError("clock controller has not been locked")
        return self._snapshot

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def _wait_for_clock(self, target: int, *, timeout_s: float = 3.0) -> GpuSnapshot:
        """Poll until nvidia-smi reports the locked clock.

        ``-lgc`` can take a moment to take effect on an idle GPU; an immediate
        query often still reports the previous low clock.
        """

        deadline = time.monotonic() + timeout_s
        after = query_gpu(self.index)
        while after.sm_clock_mhz != target and time.monotonic() < deadline:
            time.sleep(0.1)
            after = query_gpu(self.index)
        if after.sm_clock_mhz != target:
            raise ClockLockError(
                f"SM clock is {after.sm_clock_mhz} MHz after locking, expected {target}"
            )
        return after

    def lock(self) -> GpuSnapshot:
        before = query_gpu(self.index)
        target = self.requested_sm_clock_mhz or before.max_sm_clock_mhz
        if target <= 0 or target > before.max_sm_clock_mhz:
            raise ClockLockError(
                f"requested SM clock {target} is outside 1..{before.max_sm_clock_mhz}"
            )

        result = _run(["nvidia-smi", "-i", str(self.index), "-lgc", str(target)])
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "lock failed"
            raise ClockLockError(f"nvidia-smi -lgc failed: {message}")

        try:
            after = self._wait_for_clock(target)
        except Exception:
            _run(["nvidia-smi", "-i", str(self.index), "-rgc"])
            raise

        self.target_sm_clock_mhz = target
        self._snapshot = after
        self._locked = True

        if self.lock_memory and after.max_mem_clock_mhz is not None:
            memory = _run(
                ["nvidia-smi", "-i", str(self.index), "-lmc", str(after.max_mem_clock_mhz)]
            )
            text = (memory.stdout + memory.stderr).lower()
            if memory.returncode != 0 or "not supported" in text:
                self._warnings.append(
                    "memory clock lock is not supported on this GPU; SM clock is fixed"
                )
        return after

    def unlock(self) -> None:
        if not self._locked:
            return
        try:
            _run(["nvidia-smi", "-i", str(self.index), "-rgc"])
            if self.lock_memory:
                _run(["nvidia-smi", "-i", str(self.index), "-rmc"])
        finally:
            self._locked = False

    def __enter__(self) -> GpuClockController:
        self.lock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unlock()


__all__ = [
    "ClockLockError",
    "GpuClockController",
    "GpuSnapshot",
    "GpuUnavailable",
    "query_gpu",
]
