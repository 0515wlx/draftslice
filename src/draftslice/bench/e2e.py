"""B4 end-to-end speculative-decoding runner.

The runner is deliberately engine-optional: importing this module does not import
vLLM. ``run_method`` imports vLLM lazily, loads one model configuration, runs the
same prompts through every requested workload, and reports validity, acceptance
length, throughput, and wall latency. A matrix run repeats the same procedure for
each speculative method and aggregates the results.

B4 is the only tier that imports an inference engine. Keep B0-B3 engine-free.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .gpu import query_gpu

E2E_SCHEMA = "e2e/1"


class E2EError(RuntimeError):
    """Raised when an end-to-end run cannot be executed."""


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One structured or free-form workload prompt."""

    name: str
    prompt: str
    domain: str = "default"
    schema: dict[str, Any] | None = None
    regex: str | None = None
    max_tokens: int = 64

    def __post_init__(self) -> None:
        if not self.name or not self.prompt:
            raise ValueError("name and prompt must not be empty")
        if self.schema is not None and self.regex is not None:
            raise ValueError("a case uses either a schema or a regex, not both")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


DEFAULT_CASES: tuple[PromptCase, ...] = (
    PromptCase(
        name="json_person",
        domain="json_schema",
        prompt=(
            "Return one JSON object for a person named Alice, age 30, city Paris. Output only JSON."
        ),
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
            "additionalProperties": False,
        },
        max_tokens=48,
    ),
    PromptCase(
        name="tool_call",
        domain="tool_call",
        prompt=(
            "Call the weather tool for Paris in Celsius. Output one JSON object "
            "with keys tool, arguments.city, and arguments.unit."
        ),
        schema={
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["weather"]},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city", "unit"],
                    "additionalProperties": False,
                },
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        },
        max_tokens=64,
    ),
    PromptCase(
        name="json_array",
        domain="json_schema",
        prompt=(
            "Return a JSON array of exactly three objects. Each object has an "
            "integer id and a string label. Output only JSON."
        ),
        schema={
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string"},
                },
                "required": ["id", "label"],
                "additionalProperties": False,
            },
        },
        max_tokens=96,
    ),
    PromptCase(
        name="regex_date",
        domain="regex",
        prompt="Output today's date as YYYY-MM-DD. Output only the date.",
        regex=r"\d{4}-\d{2}-\d{2}",
        max_tokens=16,
    ),
    PromptCase(
        name="math",
        domain="math",
        prompt="What is 17 * 23 + 19? Answer with the number only.",
        max_tokens=32,
    ),
    PromptCase(
        name="chat",
        domain="chat",
        prompt="In one short sentence, explain why speculative decoding speeds up LLM inference.",
        max_tokens=48,
    ),
)

DEFAULT_METHODS = ("mtp", "draftslice_mtp")


@dataclass(frozen=True, slots=True)
class E2EConfig:
    model: str
    draft_model: str | None = None
    methods: tuple[str, ...] = DEFAULT_METHODS
    batch_sizes: tuple[int, ...] = (1, 8)
    max_tokens: int | None = None
    spec_tokens: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.55
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    warmup: int = 1
    repetitions: int = 1
    case_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not self.methods:
            raise ValueError("methods must not be empty")
        if not self.batch_sizes or any(batch <= 0 for batch in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")
        if self.spec_tokens < 0:
            raise ValueError("spec_tokens must be non-negative")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.case_names is not None:
            known = {case.name for case in DEFAULT_CASES}
            missing = set(self.case_names) - known
            if missing:
                raise ValueError(f"unknown cases: {', '.join(sorted(missing))}")

    def cases(self) -> tuple[PromptCase, ...]:
        if self.case_names is None:
            return DEFAULT_CASES
        return tuple(case for case in DEFAULT_CASES if case.name in self.case_names)


def _metric_value(sample: Any) -> float | None:
    value = getattr(sample, "value", None)
    if isinstance(value, (int, float)):
        return float(value)
    values = getattr(sample, "values", None)
    if isinstance(values, (list, tuple)) and values:
        return float(sum(values))
    total = getattr(sample, "sum", None)
    if isinstance(total, (int, float)):
        return float(total)
    return None


def metric_map(samples: Any) -> dict[str, float]:
    """Flatten a prometheus metric snapshot into ``name -> value``.

    vLLM returns a list of samples with a name, labels, and either a scalar
    value or a vector. Multiple label sets for one name are summed, which is the
    right aggregation for counters in a single-engine run.
    """

    result: dict[str, float] = {}
    for sample in samples or ():
        name = str(getattr(sample, "name", ""))
        value = _metric_value(sample)
        if name and value is not None:
            result[name] = result.get(name, 0.0) + value
    return result


def _metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def _sum_metric(delta: dict[str, float], prefix: str) -> tuple[float, float]:
    total = 0.0
    count = 0.0
    for name, value in delta.items():
        if name.startswith(prefix) and name.endswith("_sum"):
            total += value
        elif name.startswith(prefix) and name.endswith("_count"):
            count += value
    return total, count


def _validate(text: str, case: PromptCase) -> tuple[bool, str | None]:
    if case.schema is None and case.regex is None:
        return True, None
    stripped = text.strip()
    if case.regex is not None:
        return bool(re.fullmatch(case.regex, stripped)), None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return False, f"json: {exc}"
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - optional dependency
        return True, "jsonschema is not installed"
    try:
        jsonschema.validate(value, case.schema)
    except Exception as exc:
        return False, f"schema: {exc}"
    return True, None


def _sampling_params(
    case: PromptCase,
    config: E2EConfig,
    *,
    max_tokens: int,
) -> Any:
    from vllm.sampling_params import SamplingParams, StructuredOutputsParams

    structured = None
    if case.schema is not None:
        structured = StructuredOutputsParams(json=case.schema)
    elif case.regex is not None:
        structured = StructuredOutputsParams(regex=case.regex)
    return SamplingParams(
        n=1,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        max_tokens=max_tokens,
        structured_outputs=structured,
        ignore_eos=False,
    )


def _build_llm(config: E2EConfig, method: str) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - engine optional
        raise E2EError("vLLM is required for B4") from exc

    kwargs: dict[str, Any] = {
        "model": config.model,
        "dtype": "bfloat16",
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": config.tensor_parallel_size,
        "enforce_eager": config.enforce_eager,
        "seed": config.seed,
        "disable_log_stats": False,
    }
    if method not in ("full", "none"):
        kwargs["spec_method"] = method
        kwargs["spec_tokens"] = config.spec_tokens
        if config.draft_model is not None:
            kwargs["spec_model"] = config.draft_model
    return LLM(**kwargs)


def _case_max_tokens(case: PromptCase, config: E2EConfig) -> int:
    return case.max_tokens if config.max_tokens is None else config.max_tokens


def run_method(
    config: E2EConfig,
    method: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one engine configuration across the configured cases and batches."""

    if method not in config.methods:
        raise E2EError(f"method {method!r} is not in config.methods")

    cases = config.cases()
    if not cases:
        raise E2EError("no cases selected")

    if on_progress:
        on_progress(f"loading {config.model} method={method}")
    llm = _build_llm(config, method)
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        # Warm up kernels, CUDA graphs, and the prefix cache for every measured
        # (case, batch) outside the timed region. Warming only ``cases[0]`` left
        # the first measured repetition of every other configuration cold
        # (reported ~10x slower in the B4 sweep), which contaminates medians
        # when the repetition count is small.
        if config.warmup:
            for warm_case in cases:
                warm_tokens = min(8, _case_max_tokens(warm_case, config))
                for warm_batch in config.batch_sizes:
                    warm_prompt = [warm_case.prompt] * warm_batch
                    # Warm the structured-output path too, not just the model
                    # forward: the grammar compile, shadow registration, and
                    # slice-kernel shapes only run for structured requests.
                    # Plain ``SamplingParams`` left every measured rep to pay
                    # the first grammar state and Triton shape compile.
                    llm.generate(
                        warm_prompt,
                        _sampling_params(warm_case, config, max_tokens=warm_tokens),
                        use_tqdm=False,
                    )

        for case in cases:
            for batch in config.batch_sizes:
                prompts = [case.prompt] * batch
                for repetition in range(config.repetitions):
                    before = metric_map(llm.get_metrics())
                    start = time.perf_counter()
                    outputs = llm.generate(
                        prompts,
                        _sampling_params(case, config, max_tokens=_case_max_tokens(case, config)),
                        use_tqdm=False,
                    )
                    wall_s = time.perf_counter() - start
                    after = metric_map(llm.get_metrics())
                    delta = {
                        name: after.get(name, 0.0) - before.get(name, 0.0)
                        for name in set(before) | set(after)
                    }

                    texts = [output.outputs[0].text for output in outputs]
                    token_rows = [list(output.outputs[0].token_ids) for output in outputs]
                    valid_flags: list[bool] = []
                    validity_errors: list[str] = []
                    for text in texts:
                        valid, reason = _validate(text, case)
                        valid_flags.append(valid)
                        if reason is not None and not valid:
                            validity_errors.append(reason)

                    output_tokens = sum(len(row) for row in token_rows)
                    drafts = _metric_delta(before, after, "vllm:spec_decode_num_drafts")
                    draft_tokens = _metric_delta(before, after, "vllm:spec_decode_num_draft_tokens")
                    accepted = _metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens")
                    ttft_sum, ttft_count = _sum_metric(delta, "vllm:time_to_first_token_seconds")
                    itl_sum, itl_count = _sum_metric(delta, "vllm:inter_token_latency_seconds")
                    record = {
                        "case": case.name,
                        "domain": case.domain,
                        "batch": batch,
                        "repetition": repetition,
                        "wall_s": wall_s,
                        "output_tokens": output_tokens,
                        "tokens_per_s": 0.0 if wall_s <= 0 else output_tokens / wall_s,
                        "validity": sum(valid_flags) / len(valid_flags),
                        "validity_errors": validity_errors[:4],
                        "spec_drafts": drafts,
                        "spec_draft_tokens": draft_tokens,
                        "spec_accepted_tokens": accepted,
                        "tau": 0.0 if drafts <= 0 else accepted / drafts,
                        "acceptance_rate": 0.0 if draft_tokens <= 0 else accepted / draft_tokens,
                        "ttft_ms": None if ttft_count <= 0 else 1000.0 * ttft_sum / ttft_count,
                        "itl_ms": None if itl_count <= 0 else 1000.0 * itl_sum / itl_count,
                        "texts": texts,
                        "token_ids": token_rows,
                    }
                    records.append(record)
                    if on_progress:
                        on_progress(
                            f"{method} {case.name} B={batch} rep={repetition} "
                            f"{record['tokens_per_s']:.1f} tok/s "
                            f"tau={record['tau']:.2f} valid={record['validity']:.2f}"
                        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            del llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    structured = [
        record
        for record in records
        if record["domain"]
        in {
            "json_schema",
            "tool_call",
            "regex",
        }
    ]
    summary = {
        "records": len(records),
        "cases": len(cases),
        "validity": _mean(record["validity"] for record in records),
        "structured_validity": _mean(record["validity"] for record in structured)
        if structured
        else 1.0,
        "tokens_per_s": _mean(record["tokens_per_s"] for record in records),
        "tau": _mean(record["tau"] for record in records if record["spec_drafts"] > 0),
        "acceptance_rate": _mean(
            record["acceptance_rate"] for record in records if record["spec_draft_tokens"] > 0
        ),
        "wall_s": sum(record["wall_s"] for record in records),
        "errors": errors,
    }
    return {
        "schema_version": E2E_SCHEMA,
        "method": method,
        "model": config.model,
        "draft_model": config.draft_model,
        "config": {
            **asdict(config),
            "methods": list(config.methods),
            "case_names": None if config.case_names is None else list(config.case_names),
        },
        "gpu": _gpu_dict(),
        "summary": summary,
        "records": records,
    }


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return 0.0 if not items else sum(items) / len(items)


def _physical_gpu_index() -> int:
    """Return the physical GPU the engine actually uses.

    vLLM and PyTorch honour ``CUDA_VISIBLE_DEVICES``; a B4 run pinned with
    ``CUDA_VISIBLE_DEVICES=4`` must not report GPU 0's clocks and temperature.
    """

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return 0
    first = raw.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        # UUID / MIG identifiers cannot be passed to ``nvidia-smi -i`` here.
        return 0


def _gpu_dict() -> dict[str, Any] | None:
    try:
        snapshot = query_gpu(_physical_gpu_index()).to_dict()
    except Exception:  # pragma: no cover - no nvidia-smi
        return None
    snapshot["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return snapshot


def run_matrix(
    config: E2EConfig,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run every method and attach a paired comparison."""

    results: dict[str, Any] = {}
    for method in config.methods:
        results[method] = run_method(config, method, on_progress=on_progress)
    comparison = compare_methods(results)
    return {
        "schema_version": E2E_SCHEMA,
        "model": config.model,
        "config": {
            **asdict(config),
            "methods": list(config.methods),
            "case_names": None if config.case_names is None else list(config.case_names),
        },
        "gpu": _gpu_dict(),
        "methods": results,
        "comparison": comparison,
    }


def _key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (record["case"], int(record["batch"]), int(record["repetition"]))


def compare_methods(results: dict[str, Any]) -> dict[str, Any]:
    """Paired comparison of the first method (baseline) against the others."""

    if not results:
        return {}
    names = list(results)
    baseline_name = names[0]
    baseline = {_key(record): record for record in results[baseline_name]["records"]}
    comparison: dict[str, Any] = {"baseline": baseline_name, "methods": {}}
    for name in names[1:]:
        rows = {}
        output_equal = 0
        checked = 0
        for record in results[name]["records"]:
            base = baseline.get(_key(record))
            if base is None:
                continue
            tokens_equal = bool(base["token_ids"] == record["token_ids"])
            output_equal += int(tokens_equal)
            checked += 1
            key = f"{record['case']}|B={record['batch']}"
            rows[key] = {
                "tokens_per_s": record["tokens_per_s"],
                "baseline_tokens_per_s": base["tokens_per_s"],
                "speedup": (
                    0.0
                    if base["tokens_per_s"] <= 0
                    else record["tokens_per_s"] / base["tokens_per_s"]
                ),
                "validity": record["validity"],
                "baseline_validity": base["validity"],
                "tau": record["tau"],
                "baseline_tau": base["tau"],
                "greedy_output_equal": tokens_equal,
            }
        comparison["methods"][name] = {
            "rows": rows,
            "greedy_output_equal_rate": 0.0 if checked == 0 else output_equal / checked,
            "mean_speedup": _mean(row["speedup"] for row in rows.values()),
            "validity": results[name]["summary"]["validity"],
            "baseline_validity": results[baseline_name]["summary"]["validity"],
        }
    return comparison


def write_e2e_result(path: str | Path, result: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_CASES",
    "DEFAULT_METHODS",
    "E2E_SCHEMA",
    "E2EConfig",
    "E2EError",
    "PromptCase",
    "compare_methods",
    "metric_map",
    "run_matrix",
    "run_method",
    "write_e2e_result",
]
