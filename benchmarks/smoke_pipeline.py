"""Engine-free smoke pipeline for CI and nightly runs.

Builds a tiny screener artifact, evaluates it on a synthetic case, writes a
factorial result, then renders and verifies a release report. It exercises the
same code paths as a real release without a model or dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - CI installs the bench extra
    print("torch is not installed; skipping the screener smoke pipeline")
    raise SystemExit(0) from None

from draftslice.bench.factorial import FactorialCase, run_factorial, write_factorial
from draftslice.bench.release import build_release, verify_manifest
from draftslice.screen import StaticPolicy, build_screener_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/smoke-pipeline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    screener = StaticPolicy(rows=(1, 3, 5, 7))
    bundle = build_screener_bundle(
        screener,
        vocab_size=16,
        hidden_size=8,
        calibration={"delta": 0.05, "calibration_id": "smoke"},
    )
    bundle.write(out / "artifact")

    case = FactorialCase(
        name="synthetic-tool-call",
        hidden=torch.randn(4, 8),
        truth=((1, 3), (1, 5), (3, 7), (1, 7)),
    )
    factorial = run_factorial([screener], [case], k=2)
    factorial_path = out / "factorial.json"
    write_factorial(factorial_path, factorial)

    head = {
        "schema_version": "bench/1",
        "name": "synthetic-head",
        "metrics": {"top1": 0.5, "set_recall@5": 0.75},
        "warnings": ["synthetic smoke data"],
    }
    head_path = out / "head.json"
    head_path.write_text(json.dumps(head, indent=2) + "\n", encoding="utf-8")

    manifest, _ = build_release(
        "smoke-release",
        [head_path, factorial_path],
        out=out / "report.md",
        metadata={"synthetic": True},
        root=out,
    )
    problems = verify_manifest(manifest)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"wrote {out / 'report.md'}")
    print(f"wrote {out / 'report.manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
