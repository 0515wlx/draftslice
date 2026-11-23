"""B4 end-to-end runner for SGLang EAGLE/MTP/DFlash.

The runner loads one SGLang engine per arm, runs the same structured prompts,
and records throughput, validity, per-request latency, greedy token equality,
and the DraftSlice contract/reason counts written by the plugin JSONL sink.

It is engine-optional: importing this module does not import SGLang.

```bash
python -m draftslice.bench.e2e_sglang \
  --model /mnt/data0601/model/Qwen3.5-4B \
  --arms builtin_mtp,draftslice_mtp \
  --batches 1,4 --reps 2 --out runs/p4_sglang_mtp.json
```
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .e2e import DEFAULT_CASES, PromptCase, _validate
from .gpu import query_gpu

SCHEMA = "e2e-sglang/1"


class SglangBenchError(RuntimeError):
    """The SGLang B4 run cannot execute."""


@dataclass(frozen=True, slots=True)
class SglangArm:
    name: str
    speculative_algorithm: str = "MTP"
    draft_model: str | None = None
    draftslice: bool = False
    method: str = "draftslice_eagle3"

    def __post_init__(self) -> None:
        if not self.name:
            raise SglangBenchError("arm name must not be empty")


@dataclass(frozen=True, slots=True)
class SglangBenchConfig:
    model: str
    arms: tuple[SglangArm, ...]
    batches: tuple[int, ...] = (1, 4)
    repetitions: int = 2
    cases: tuple[str, ...] = ("tool_call", "json_array")
    spec_steps: int = 3
    spec_tokens: int = 4
    max_new_tokens: int | None = None
    tp_size: int = 1
    mem_fraction_static: float = 0.55
    max_running_requests: int = 0
    temperature: float = 0.0
    seed: int = 0
    device: str = "cuda:0"
    disable_cuda_graph: bool = False
    warmup: bool = True
    trace_dir: str = "runs/p4_sglang_traces"

    def __post_init__(self) -> None:
        if not self.arms:
            raise SglangBenchError("at least one arm is required")
        if not self.batches or any(batch <= 0 for batch in self.batches):
            raise SglangBenchError("batches must be positive")
        if self.repetitions <= 0:
            raise SglangBenchError("repetitions must be positive")

    def selected_cases(self) -> tuple[PromptCase, ...]:
        if not self.cases:
            return DEFAULT_CASES
        wanted = set(self.cases)
        return tuple(case for case in DEFAULT_CASES if case.name in wanted)


def _engine(config: SglangBenchConfig, arm: SglangArm) -> Any:
    try:
        from sglang.srt.entrypoints.engine import Engine
    except ImportError as exc:  # pragma: no cover - engine optional
        raise SglangBenchError("SGLang is required for the SGLang B4 run") from exc
    kwargs: dict[str, Any] = {
        "model_path": config.model,
        "tp_size": config.tp_size,
        "mem_fraction_static": config.mem_fraction_static,
        "dtype": "bfloat16",
        "random_seed": config.seed,
        "log_level": "error",
    }
    if config.device:
        kwargs["device"] = config.device
    if config.disable_cuda_graph:
        # The EAGLE head hook is bypassed by the captured graph path today.
        kwargs["disable_cuda_graph"] = True
    if config.max_running_requests > 0:
        kwargs["max_running_requests"] = config.max_running_requests
    if arm.speculative_algorithm:
        kwargs["speculative_algorithm"] = arm.speculative_algorithm
        kwargs["speculative_num_steps"] = config.spec_steps
        kwargs["speculative_eagle_topk"] = 1
        kwargs["speculative_num_draft_tokens"] = config.spec_tokens
    if arm.draft_model:
        kwargs["speculative_draft_model_path"] = arm.draft_model
    return Engine(**kwargs)


def _sampling(case: PromptCase, config: SglangBenchConfig, max_tokens: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "temperature": config.temperature,
        "top_p": 1.0,
    }
    if case.schema is not None:
        params["json_schema"] = json.dumps(case.schema)
    elif case.regex is not None:
        params["regex"] = case.regex
    return params


def _arm_env(arm: SglangArm, trace_path: Path) -> dict[str, str]:
    if not arm.draftslice:
        return {
            "DRAFTSLICE_SGLANG": "0",
            "DRAFTSLICE_TRACE_JSONL": "",
        }
    return {
        "DRAFTSLICE_SGLANG": "1",
        "DRAFTSLICE_MODEL": os.path.basename(os.path.normpath(arm.draft_model or "")),
        "DRAFTSLICE_METHOD": arm.method,
        "DRAFTSLICE_WORKLOAD": "tool_call",
        "DRAFTSLICE_QUANT": "auto",
        "DRAFTSLICE_GRAPH": "eager",
        "DRAFTSLICE_TRACE_SAMPLE": "1.0",
        "DRAFTSLICE_TRACE_JSONL": str(trace_path),
    }


def _apply_env(values: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in values.items():
        previous[key] = os.environ.get(key)
        if value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _trace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": 0, "reasons": {}, "actions": {}}
    reasons: dict[str, int] = {}
    actions: dict[str, int] = {}
    records = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records += 1
        contract = record.get("contract") or {}
        reason = contract.get("fallback_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        mode = contract.get("mode")
        if mode:
            actions[mode] = actions.get(mode, 0) + 1
    return {"records": records, "reasons": reasons, "actions": actions}


def run_arm(
    config: SglangBenchConfig,
    arm: SglangArm,
    *,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    trace_path = Path(config.trace_dir) / f"{arm.name}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()
    previous = _apply_env(_arm_env(arm, trace_path))
    engine = None
    records: list[dict[str, Any]] = []
    try:
        engine = _engine(config, arm)
        cases = config.selected_cases()
        if config.warmup:
            for case in cases:
                warm = min(8, case.max_tokens)
                engine.generate(
                    [case.prompt] * max(config.batches),
                    _sampling(case, config, warm),
                )
        for case in cases:
            for batch in config.batches:
                prompts = [case.prompt] * batch
                for repetition in range(config.repetitions):
                    max_tokens = (
                        case.max_tokens
                        if config.max_new_tokens is None
                        else config.max_new_tokens
                    )
                    start = time.perf_counter()
                    outputs = engine.generate(
                        prompts, _sampling(case, config, max_tokens)
                    )
                    wall_s = time.perf_counter() - start
                    rows = [row if isinstance(row, dict) else row.__dict__ for row in outputs]
                    texts = [str(row.get("text", "")) for row in rows]
                    token_rows = [
                        list(row.get("output_ids") or ()) for row in rows
                    ]
                    valid_flags: list[bool] = []
                    errors: list[str] = []
                    for text in texts:
                        valid, reason = _validate(text, case)
                        valid_flags.append(valid)
                        if reason and not valid:
                            errors.append(reason)
                    output_tokens = sum(len(tokens) for tokens in token_rows)
                    latencies = [
                        1000.0 * float((row.get("meta_info") or {}).get("e2e_latency", 0.0))
                        for row in rows
                    ]
                    record = {
                        "arm": arm.name,
                        "case": case.name,
                        "domain": case.domain,
                        "batch": batch,
                        "repetition": repetition,
                        "wall_s": wall_s,
                        "output_tokens": output_tokens,
                        "tokens_per_s": 0.0 if wall_s <= 0 else output_tokens / wall_s,
                        "validity": (
                            sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
                        ),
                        "validity_errors": errors[:4],
                        "latency_ms": latencies,
                        "ttft_ms": None,
                        "itl_ms": None,
                        "token_ids": token_rows,
                        "texts": texts,
                    }
                    records.append(record)
                    if on_progress:
                        on_progress(
                            f"{arm.name} {case.name} B={batch} rep={repetition} "
                            f"{record['tokens_per_s']:.1f} tok/s "
                            f"valid={record['validity']:.2f}"
                        )
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass
        _restore_env(previous)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    summary = {
        "records": len(records),
        "tokens_per_s": _mean(r["tokens_per_s"] for r in records),
        "validity": _mean(r["validity"] for r in records),
        "mean_latency_ms": _mean(
            _mean(r["latency_ms"]) for r in records if r["latency_ms"]
        ),
        "trace": _trace_summary(trace_path),
        "errors": [],
    }
    return {
        "schema": SCHEMA,
        "arm": asdict(arm),
        "model": config.model,
        "config": {
            **asdict(config),
            "arms": [asdict(a) for a in config.arms],
            "batches": list(config.batches),
            "cases": list(config.cases),
        },
        "gpu": _gpu(),
        "summary": summary,
        "records": records,
    }


def run(config: SglangBenchConfig, *, on_progress: Any | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for arm in config.arms:
        results[arm.name] = run_arm(config, arm, on_progress=on_progress)
    comparison = _compare(results)
    return {
        "schema": SCHEMA,
        "model": config.model,
        "arms": results,
        "comparison": comparison,
        "gpu": _gpu(),
    }


def _compare(results: dict[str, Any]) -> dict[str, Any]:
    base = results.get("builtin_mtp") or next(iter(results.values()), None)
    out: dict[str, Any] = {}
    if base is None:
        return out
    base_records = {
        (r["case"], r["batch"], r["repetition"]): r for r in base["records"]
    }
    for name, result in results.items():
        ratios = []
        equal = []
        for record in result["records"]:
            key = (record["case"], record["batch"], record["repetition"])
            other = base_records.get(key)
            if other is None or other["tokens_per_s"] <= 0:
                continue
            ratios.append(record["tokens_per_s"] / other["tokens_per_s"])
            equal.append(record["token_ids"] == other["token_ids"])
        out[name] = {
            "throughput_ratio": _mean(ratios) if ratios else None,
            "greedy_equal": all(equal) if equal else None,
            "validity": result["summary"]["validity"],
            "trace": result["summary"]["trace"],
        }
    return out


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return 0.0 if not items else sum(items) / len(items)


def _gpu() -> dict[str, Any] | None:
    try:
        return asdict(query_gpu(0))
    except Exception:  # pragma: no cover - no nvidia-smi
        return None


def _arm(value: str) -> SglangArm:
    if value == "builtin_mtp":
        return SglangArm("builtin_mtp", "FROZEN_KV_MTP")
    if value == "draftslice_mtp":
        return SglangArm(
            "draftslice_mtp",
            "FROZEN_KV_MTP",
            draftslice=True,
            method="draftslice_mtp",
        )
    if value == "builtin_dflash":
        return SglangArm("builtin_dflash", "DFLASH")
    if value == "draftslice_dflash":
        return SglangArm(
            "draftslice_dflash",
            "DFLASH",
            draftslice=True,
            method="draftslice_dflash",
        )
    if value == "builtin_eagle":
        return SglangArm("builtin_eagle", "EAGLE3")
    if value == "draftslice_eagle":
        return SglangArm(
            "draftslice_eagle", "EAGLE3", draftslice=True, method="draftslice_eagle3"
        )
    # Qwen3.5 built-in MTP heads run through SGLang's EAGLE worker with the
    # target as the draft model; FROZEN_KV_MTP is broken in SGLang 0.5.23.
    if value == "builtin_eagle_mtp":
        return SglangArm("builtin_eagle_mtp", "EAGLE")
    if value == "draftslice_eagle_mtp":
        return SglangArm(
            "draftslice_eagle_mtp",
            "EAGLE",
            draftslice=True,
            method="draftslice_mtp",
        )
    raise SglangBenchError(f"unknown arm: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SGLang B4 comparison.")
    parser.add_argument("--model", default="/mnt/data0601/model/Qwen3.5-4B")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument(
        "--arms",
        default="builtin_mtp,draftslice_mtp",
        help=(
            "comma-separated: builtin_mtp,draftslice_mtp,builtin_dflash,"
            "builtin_eagle_mtp,draftslice_eagle_mtp,..."
        ),
    )
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--cases", default="tool_call,json_array")
    parser.add_argument("--spec-steps", type=int, default=3)
    parser.add_argument("--spec-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.55)
    parser.add_argument("--max-running-requests", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="eager engine; required for the EAGLE head hook in this SGLang build",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/p4_sglang.json"))
    args = parser.parse_args(argv)

    arms = []
    for token in args.arms.split(","):
        arm = _arm(token.strip())
        if args.draft_model and (
            arm.name.endswith("dflash") or "eagle_mtp" in arm.name
        ):
            arm = SglangArm(
                arm.name,
                arm.speculative_algorithm,
                draft_model=args.draft_model,
                draftslice=arm.draftslice,
                method=arm.method,
            )
        arms.append(arm)
    config = SglangBenchConfig(
        model=args.model,
        arms=tuple(arms),
        batches=tuple(int(value) for value in args.batches.split(",")),
        repetitions=args.reps,
        cases=tuple(c.strip() for c in args.cases.split(",") if c.strip()),
        spec_steps=args.spec_steps,
        spec_tokens=args.spec_tokens,
        max_new_tokens=args.max_new_tokens,
        tp_size=args.tp,
        mem_fraction_static=args.mem_fraction,
        max_running_requests=args.max_running_requests,
        device=args.device,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    result = run(config, on_progress=print)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out": str(args.out), "comparison": result["comparison"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["SglangArm", "SglangBenchConfig", "main", "run", "run_arm"]
