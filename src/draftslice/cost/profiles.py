"""Hardware profiles for the first cost model.

The constants are datasheet values plus the fit points from the local research
workspace. They are starting points, not benchmark claims: every profile must be
recalibrated on the target machine before its predictions are published.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """A small roofline profile for one accelerator."""

    name: str
    hbm_bandwidth: float
    l2_bandwidth: float
    l2_bytes: int
    alu_throughput: float
    mma_throughput: float
    launch_overhead_s: float
    gather_bandwidth: float
    dtype_bytes: int = 2
    source: str = "datasheet + local fit"

    def __post_init__(self) -> None:
        for name in (
            "hbm_bandwidth",
            "l2_bandwidth",
            "alu_throughput",
            "mma_throughput",
            "launch_overhead_s",
            "gather_bandwidth",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.l2_bytes <= 0:
            raise ValueError("l2_bytes must be positive")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")


A100 = HardwareProfile(
    name="a100",
    hbm_bandwidth=2.00e12,
    l2_bandwidth=6.0e12,
    l2_bytes=40_000_000,
    alu_throughput=1.0e12,
    mma_throughput=156e12,
    launch_overhead_s=16e-6,
    gather_bandwidth=0.53e12,
)

H100 = HardwareProfile(
    name="h100",
    hbm_bandwidth=3.35e12,
    l2_bandwidth=12.0e12,
    l2_bytes=50_000_000,
    alu_throughput=2.0e12,
    mma_throughput=494e12,
    launch_overhead_s=12e-6,
    gather_bandwidth=0.90e12,
)

PROFILES: dict[str, HardwareProfile] = {profile.name: profile for profile in (A100, H100)}


def get_profile(name: str) -> HardwareProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown hardware profile {name!r}; known: {known}") from exc


__all__ = ["A100", "H100", "PROFILES", "HardwareProfile", "get_profile"]
