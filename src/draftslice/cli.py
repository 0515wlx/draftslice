"""Command-line entry point for DraftSlice (pre-alpha)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .bench.constraint import (
    ConstraintBenchError,
    ConstraintRecord,
    load_constraint_records,
    run_constraint_replay,
    write_constraint_result,
)
from .bench.e2e import E2EConfig, E2EError, run_matrix, run_method, write_e2e_result
from .bench.factorial import (
    BaselineRunnerError,
    load_cases,
    run_factorial,
    write_factorial,
)
from .bench.gpu import ClockLockError, GpuClockController, GpuUnavailable
from .bench.harness import HarnessError, evaluate, load_records, write_result
from .bench.kernel import (
    KERNEL_NAMES,
    KernelBenchConfig,
    KernelBenchError,
    run_kernel_bench,
    write_kernel_result,
)
from .bench.release import (
    ReleaseError,
    ReleaseManifest,
    build_release,
    compare_manifests,
    verify_manifest,
)
from .constraints.fixture import ConstraintFixtureError, fixture_from_dict, load_constraint_fixture
from .plan.compiler import Budget, PlanBudgetError, PlanCompiler, PlanPolicy
from .plan.serialize import write_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="draftslice",
        description="Exact, constraint-aware draft-head slicing for speculative decoding.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    compile_cmd = sub.add_parser(
        "compile",
        help="Compile a constraint state into a head plan.",
    )
    source = compile_cmd.add_mutually_exclusive_group()
    source.add_argument(
        "--schema",
        help=(
            "Path to a constraint fixture. M1 accepts {'valid_ids': [...]} or "
            "{'states': [...]}; JSON Schema/CFG adapters land later."
        ),
    )
    source.add_argument(
        "--valid-ids",
        help="Comma-separated token ids for the reference backend.",
    )
    compile_cmd.add_argument("--out", required=True, help="Output plan path.")
    compile_cmd.add_argument("--state-id", type=int, default=None, help="State to compile.")
    compile_cmd.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size.")
    compile_cmd.add_argument("--hidden-size", type=int, default=None, help="Hidden width.")
    compile_cmd.add_argument(
        "--mode",
        choices=("strict", "full"),
        default="strict",
        help="Plan mode (default: strict).",
    )
    compile_cmd.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Fall back to the full head above this many candidates.",
    )
    compile_cmd.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Fall back to the full head above this many dynamic bytes.",
    )
    compile_cmd.add_argument(
        "--no-full-fallback",
        action="store_true",
        help="Fail instead of falling back to the full head.",
    )

    bench_cmd = sub.add_parser(
        "bench",
        help="Benchmark draft-head slicing on a target and drafter.",
    )
    bench_cmd.add_argument("--config", help="Path to a TOML/YAML config.")
    bench_sub = bench_cmd.add_subparsers(dest="bench_command")
    accuracy = bench_sub.add_parser(
        "accuracy",
        help="Evaluate candidate sets against cached oracle top-k records.",
    )
    accuracy.add_argument("--input", required=True, help="JSONL candidate records.")
    accuracy.add_argument("--out", required=True, help="Output result JSON.")
    accuracy.add_argument("--k", type=int, default=5, help="Top-k for set metrics.")
    accuracy.add_argument("--name", default="accuracy", help="Result name.")
    accuracy.add_argument(
        "--weights",
        help="Comma-separated OA prefix weights (default: uniform).",
    )

    kernel = bench_sub.add_parser(
        "kernel",
        help="Measure full/gather/fused kernels with the SM clock fixed.",
    )
    kernel.add_argument("--out", required=True, help="Output result JSON.")
    kernel.add_argument("--vocab-size", type=int, default=248_320)
    kernel.add_argument("--hidden-size", type=int, default=2_560)
    kernel.add_argument("--batch", default="1,8", help="Comma-separated batch sizes.")
    kernel.add_argument(
        "--rows",
        default="128,1024,4096,8192,16384",
        help="Comma-separated row counts.",
    )
    kernel.add_argument(
        "--kernels",
        default=",".join(KERNEL_NAMES),
        help="Comma-separated kernel names (full, torch_gather, triton_slice).",
    )
    kernel.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Skip the CUDA-graph replay measurements.",
    )
    kernel.add_argument(
        "--graph-rows-limit",
        type=int,
        default=65_536,
        help="Largest row count to capture in a CUDA graph.",
    )
    kernel.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    kernel.add_argument("--warmup", type=int, default=10)
    kernel.add_argument("--iterations", type=int, default=30)
    kernel.add_argument("--gpu", type=int, default=0, help="GPU index to lock and measure.")
    kernel.add_argument("--device", default=None, help="Torch device (default: cuda:<gpu>).")
    kernel.add_argument(
        "--sm-clock",
        type=int,
        default=None,
        help="SM clock in MHz (default: the GPU's maximum SM clock).",
    )
    kernel.add_argument(
        "--allow-unlocked",
        action="store_true",
        help="Run without fixing the SM clock (results are marked unlocked).",
    )

    constraint = bench_sub.add_parser(
        "constraint",
        help="Replay recorded grammar states through the strict compiler (B3).",
    )
    constraint.add_argument("--input", default=None, help="JSON/JSONL state replay.")
    constraint.add_argument("--out", required=True, help="Output result JSON.")
    constraint.add_argument(
        "--schema",
        default=None,
        help="JSON schema file used to generate a real xgrammar trajectory.",
    )
    constraint.add_argument("--tokenizer", default=None, help="Tokenizer path for --schema.")
    constraint.add_argument("--regex", default=None, help="Regex used with --tokenizer.")
    constraint.add_argument(
        "--max-steps",
        type=int,
        default=32,
        help="Maximum trajectory steps when generating from a grammar.",
    )
    constraint.add_argument("--vocab-size", type=int, default=248_320)
    constraint.add_argument("--hidden-size", type=int, default=2_560)
    constraint.add_argument("--max-candidates", type=int, default=None)
    constraint.add_argument("--max-bytes", type=int, default=None)
    constraint.add_argument("--no-cache", action="store_true")
    constraint.add_argument("--cache-bytes", type=int, default=64 * 1024 * 1024)
    constraint.add_argument(
        "--prefer-kernel",
        choices=("auto", "slice_gemv", "slice_gemm"),
        default="auto",
    )

    e2e = bench_sub.add_parser(
        "e2e",
        help="Run B4 end-to-end speculative decoding through vLLM.",
    )
    e2e.add_argument("--out", required=True, help="Output result JSON.")
    e2e.add_argument("--model", required=True, help="Target model path or HF id.")
    e2e.add_argument("--draft-model", default=None, help="Optional draft model path.")
    e2e.add_argument(
        "--method",
        default=None,
        help="Run one speculative method (for example mtp or draftslice_mtp).",
    )
    e2e.add_argument(
        "--methods",
        default="mtp,draftslice_mtp",
        help="Comma-separated methods for a matrix run.",
    )
    e2e.add_argument("--batch", default="1,8", help="Comma-separated batch sizes.")
    e2e.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case names (default: all B4 cases).",
    )
    e2e.add_argument("--max-tokens", type=int, default=None)
    e2e.add_argument("--spec-tokens", type=int, default=1)
    e2e.add_argument("--temperature", type=float, default=0.0)
    e2e.add_argument("--top-p", type=float, default=1.0)
    e2e.add_argument("--max-model-len", type=int, default=2048)
    e2e.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    e2e.add_argument("--tensor-parallel-size", type=int, default=1)
    e2e.add_argument("--warmup", type=int, default=1)
    e2e.add_argument("--repetitions", type=int, default=1)
    e2e.add_argument("--enforce-eager", action="store_true")

    factorial = bench_sub.add_parser(
        "factorial",
        help="Run a screener x workload matrix over cached hidden states.",
    )
    factorial.add_argument("--input", required=True, help="Torch case file.")
    factorial.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Screener artifact bundle directory (repeatable).",
    )
    factorial.add_argument(
        "--include-full",
        action="store_true",
        help="Add the full-head reference screener.",
    )
    factorial.add_argument("--vocab-size", type=int, default=None)
    factorial.add_argument("--out", required=True, help="Output result JSON.")
    factorial.add_argument("--k", type=int, default=5)
    factorial.add_argument("--dtype-bytes", type=int, default=2)
    factorial.add_argument("--intersect-valid", action="store_true")

    screen_cmd = sub.add_parser(
        "screen",
        help="Train screeners and build screener artifacts.",
    )
    screen_sub = screen_cmd.add_subparsers(dest="screen_command")

    train_router = screen_sub.add_parser(
        "train-router",
        help="Train an MLP router from group marginals.",
    )
    train_router.add_argument("--input", required=True, help="Torch training file.")
    train_router.add_argument("--out", required=True, help="Artifact bundle directory.")
    train_router.add_argument("--top-c", type=int, required=True)
    train_router.add_argument("--hidden-dim", type=int, default=None)
    train_router.add_argument("--epochs", type=int, default=200)
    train_router.add_argument("--learning-rate", type=float, default=1e-2)
    train_router.add_argument("--batch-size", type=int, default=256)
    train_router.add_argument("--validation-fraction", type=float, default=0.2)
    train_router.add_argument("--weight-decay", type=float, default=0.0)
    train_router.add_argument("--seed", type=int, default=0)
    train_router.add_argument("--name", default="mlp_router")

    report = bench_sub.add_parser(
        "report",
        help="Validate result files and render a release report.",
    )
    report.add_argument(
        "--input", action="append", required=True, help="Result JSON (repeatable)."
    )
    report.add_argument("--out", required=True, help="Output Markdown path.")
    report.add_argument("--name", required=True, help="Release name.")
    report.add_argument(
        "--metadata", default=None, help="Optional JSON object of release metadata."
    )

    validate = bench_sub.add_parser(
        "validate",
        help="Verify a release manifest and its result hashes.",
    )
    validate.add_argument("--manifest", required=True, help="release .manifest.json")
    validate.add_argument("--no-hash", action="store_true", help="Skip file hashes.")

    compare = bench_sub.add_parser(
        "compare",
        help="Render a B5 comparison between two release manifests.",
    )
    compare.add_argument("--base", required=True, help="Baseline manifest.")
    compare.add_argument("--candidate", required=True, help="Candidate manifest.")
    compare.add_argument("--out", required=True, help="Output Markdown path.")

    serve_cmd = sub.add_parser(
        "serve",
        help="Run the DraftSlice runtime.",
    )
    serve_cmd.add_argument("--config", required=True, help="Path to a TOML/YAML config.")

    return parser


def _compile(args: argparse.Namespace) -> int:
    if args.schema is None and args.valid_ids is None:
        print("draftslice compile: provide --schema or --valid-ids.")
        return 2

    try:
        if args.valid_ids is not None:
            ids = [int(item.strip()) for item in args.valid_ids.split(",") if item.strip()]
            fixture = fixture_from_dict(
                {
                    "state_id": 0 if args.state_id is None else args.state_id,
                    "valid_ids": ids,
                    **({"vocab_size": args.vocab_size} if args.vocab_size else {}),
                    **({"hidden_size": args.hidden_size} if args.hidden_size else {}),
                }
            )
        else:
            fixture = load_constraint_fixture(Path(args.schema))
    except (ConstraintFixtureError, ValueError) as exc:
        print(f"draftslice compile: {exc}")
        return 2

    state = fixture.primary
    if args.state_id is not None:
        matches = [item for item in fixture.states if item.state_id == args.state_id]
        if not matches:
            print(f"draftslice compile: state {args.state_id} not found in fixture")
            return 2
        state = matches[0]

    vocab_size = args.vocab_size or fixture.vocab_size
    if vocab_size is None:
        vocab_size = max(state.valid_ids.ids, default=-1) + 1
    hidden_size = args.hidden_size or fixture.hidden_size or 0

    compiler = PlanCompiler(vocab_size=vocab_size, hidden_size=hidden_size)
    budget = Budget(
        max_candidates=args.max_candidates,
        max_bytes=args.max_bytes,
        allow_full_fallback=not args.no_full_fallback,
    )
    policy = PlanPolicy(mode=args.mode)
    try:
        plan = compiler.compile_strict(state, budget=budget, policy=policy)
    except (PlanBudgetError, ValueError) as exc:
        print(f"draftslice compile: {exc}")
        return 2

    write_plan(args.out, plan)
    print(
        f"wrote {args.out}: {plan.plan_id} state={plan.state_id} "
        f"candidates={plan.size} execution={plan.execution.kind} "
        f"bytes={plan.cost.total_bytes}"
    )
    return 0


def _bench_accuracy(args: argparse.Namespace) -> int:
    weights = None
    if args.weights:
        try:
            weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
        except ValueError:
            print("draftslice bench accuracy: weights must be comma-separated numbers")
            return 2
    try:
        records = load_records(args.input)
        result = evaluate(records, name=args.name, k=args.k, weights=weights)
    except (HarnessError, ValueError) as exc:
        print(f"draftslice bench accuracy: {exc}")
        return 2
    write_result(args.out, result)
    metrics = result.metrics
    print(
        f"wrote {args.out}: n={result.n} top1={metrics['top1']:.4f} "
        f"recall@{args.k}={metrics[f'set_recall@{args.k}']:.4f} "
        f"OA={metrics['OA']:.4f}"
    )
    return 0


def _parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _bench_kernel(args: argparse.Namespace) -> int:
    device = args.device or f"cuda:{args.gpu}"
    if device == "cuda":
        device = f"cuda:{args.gpu}"
    if device.startswith("cuda:"):
        try:
            device_index = int(device.rsplit(":", 1)[1])
        except ValueError:
            print(f"draftslice bench kernel: invalid device {device!r}")
            return 2
        if device_index != args.gpu:
            print("draftslice bench kernel: --device index must match --gpu")
            return 2

    try:
        batch_sizes = _parse_int_list(args.batch, "batch")
        row_counts = _parse_int_list(args.rows, "rows")
        kernels = tuple(item.strip() for item in args.kernels.split(",") if item.strip())
        config = KernelBenchConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            batch_sizes=batch_sizes,
            row_counts=row_counts,
            dtype=args.dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
            kernels=kernels,
            cuda_graph=not args.no_cuda_graph,
            graph_rows_limit=args.graph_rows_limit,
        )
    except ValueError as exc:
        print(f"draftslice bench kernel: {exc}")
        return 2

    try:
        if args.allow_unlocked:
            result = run_kernel_bench(config)
        else:
            with GpuClockController(args.gpu, args.sm_clock) as clock:
                result = run_kernel_bench(config, clock=clock.snapshot)
                for warning in clock.warnings:
                    print(f"draftslice bench kernel: warning: {warning}")
    except (ClockLockError, GpuUnavailable, KernelBenchError, ImportError) as exc:
        print(f"draftslice bench kernel: {exc}")
        return 2

    write_kernel_result(args.out, result)
    print(f"wrote {args.out}: clock_fixed={result['clock_fixed']}")
    if result["graph"]["available"]:
        print(
            "  cuda graphs: "
            f"captures={result['graph']['captures']} "
            f"replays={result['graph']['replays']} "
            f"fallbacks={result['graph']['fallbacks']}"
        )
    for item in result["results"]:
        print(
            f"  {item['kernel']:20s} {item['mode']:5s} B={item['batch']:<3d} "
            f"rows={item['rows']:<7d} median={item['median_us']:8.2f} us "
            f"p90={item['p90_us']:8.2f} us cv={item['cv']:.3f} "
            f"bw={item['weight_gbps']:7.1f} GB/s"
        )
    return 0


def _bench_constraint(args: argparse.Namespace) -> int:
    try:
        if args.input is not None:
            records = load_constraint_records(args.input)
        elif args.schema is not None or args.regex is not None:
            if args.tokenizer is None:
                print("draftslice bench constraint: --tokenizer is required with a grammar")
                return 2
            from .constraints.xgrammar import XGrammarSession, record_trajectory

            if args.schema is not None:
                schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
                session = XGrammarSession.from_json_schema(schema, tokenizer=args.tokenizer)
            else:
                session = XGrammarSession.from_regex(args.regex, tokenizer=args.tokenizer)
            raw = record_trajectory(session, max_tokens=args.max_steps)
            records = [
                ConstraintRecord(
                    state_id=int(item["state_id"]),
                    valid_ids=tuple(item["valid_ids"]),
                )
                for item in raw
            ]
        else:
            print("draftslice bench constraint: provide --input or --schema/--regex")
            return 2
        result = run_constraint_replay(
            records,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_candidates=args.max_candidates,
            max_bytes=args.max_bytes,
            use_cache=not args.no_cache,
            cache_bytes=args.cache_bytes,
            policy=PlanPolicy(mode="strict", prefer_kernel=args.prefer_kernel),
        )
    except (ConstraintBenchError, ValueError, OSError, ImportError) as exc:
        print(f"draftslice bench constraint: {exc}")
        return 2
    write_constraint_result(args.out, result)
    summary = result["summary"]
    print(
        f"wrote {args.out}: states={summary['states']} "
        f"violations={summary['violations']} fallback={summary['fallback_rate']:.3f} "
        f"p50={summary['compile_p50_us']:.1f}us p90={summary['compile_p90_us']:.1f}us"
    )
    if summary["cache"] is not None:
        print(
            f"  cache: hits={summary['cache']['hits']} "
            f"misses={summary['cache']['misses']} "
            f"hit_rate={summary['cache']['hit_rate']:.3f}"
        )
    return 0


def _bench_e2e(args: argparse.Namespace) -> int:
    try:
        batch_sizes = _parse_int_list(args.batch, "batch")
        methods = (
            (args.method,)
            if args.method
            else tuple(item.strip() for item in args.methods.split(",") if item.strip())
        )
        cases = (
            None
            if args.cases is None
            else tuple(item.strip() for item in args.cases.split(",") if item.strip())
        )
        config = E2EConfig(
            model=args.model,
            draft_model=args.draft_model,
            methods=methods,
            batch_sizes=batch_sizes,
            max_tokens=args.max_tokens,
            spec_tokens=args.spec_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            warmup=args.warmup,
            repetitions=args.repetitions,
            case_names=cases,
        )
    except ValueError as exc:
        print(f"draftslice bench e2e: {exc}")
        return 2

    try:
        if args.method:
            result = run_method(config, args.method)
        else:
            result = run_matrix(config)
    except (E2EError, ImportError, RuntimeError) as exc:
        print(f"draftslice bench e2e: {exc}")
        return 2

    write_e2e_result(args.out, result)
    print(f"wrote {args.out}")
    if args.method:
        summary = result["summary"]
        print(
            f"  {args.method}: records={summary['records']} "
            f"validity={summary['validity']:.4f} "
            f"tok/s={summary['tokens_per_s']:.2f} tau={summary['tau']:.3f}"
        )
    else:
        for method, value in result["methods"].items():
            summary = value["summary"]
            print(
                f"  {method}: records={summary['records']} "
                f"validity={summary['validity']:.4f} "
                f"tok/s={summary['tokens_per_s']:.2f} tau={summary['tau']:.3f}"
            )
        for method, value in result["comparison"].get("methods", {}).items():
            print(
                f"  vs {method}: speedup={value['mean_speedup']:.3f} "
                f"greedy_equal={value['greedy_output_equal_rate']:.3f}"
            )
    return 0


def _bench_factorial(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.input)
        from .screen.artifacts import load_screener_bundle
        from .screen.policies import FullPolicy

        screeners = [load_screener_bundle(path)[0] for path in args.artifact]
        if args.include_full:
            vocab_size = args.vocab_size
            if vocab_size is None:
                max_token = max(
                    [token for case in cases for row in case.truth for token in row]
                    + [
                        token
                        for case in cases
                        for row in (case.valid_sets or ())
                        if row is not None
                        for token in row
                    ]
                )
                vocab_size = max_token + 1
            screeners.append(FullPolicy(vocab_size=vocab_size))
        if not screeners:
            print("draftslice bench factorial: provide --artifact or --include-full.")
            return 2
        result = run_factorial(
            screeners,
            cases,
            k=args.k,
            intersect_valid=args.intersect_valid,
            dtype_bytes=args.dtype_bytes,
        )
    except (BaselineRunnerError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"draftslice bench factorial: {exc}")
        return 2
    write_factorial(args.out, result)
    print(f"wrote {args.out}")
    for cell in result["cells"]:
        print(
            f"  {cell['screener']} / {cell['case']}: "
            f"coverage={cell['coverage']:.4f} "
            f"mean_candidates={cell['mean_candidates']:.1f}"
        )
    return 0


def _screen_train_router(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("draftslice screen train-router: torch is required.")
        return 2
    try:
        payload = torch.load(Path(args.input), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("training file must contain an object")
        hidden = payload["hidden"]
        groups = tuple(tuple(int(token) for token in group) for group in payload["groups"])
        if "group_mass" in payload:
            group_mass = payload["group_mass"]
        elif "tokens" in payload:
            vocab_size = payload.get("vocab_size")
            if vocab_size is None:
                vocab_size = 1 + max(
                    (int(token) for row in payload["tokens"] for token in row),
                    default=0,
                )
            from .screen.router import group_mass_from_tokens

            group_mass = group_mass_from_tokens(
                payload["tokens"], groups, vocab_size=int(vocab_size)
            )
        else:
            raise ValueError("training file needs 'group_mass' or 'tokens'")
        from .screen.artifacts import build_screener_bundle
        from .screen.router import train_router

        policy, report = train_router(
            hidden,
            group_mass,
            groups,
            top_c=args.top_c,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            weight_decay=args.weight_decay,
            seed=args.seed,
            name=args.name,
        )
        vocab_size = payload.get("vocab_size")
        if vocab_size is None:
            vocab_size = 1 + max(
                (token for group in groups for token in group), default=0
            )
        calibration = dict(payload.get("calibration") or {})
        calibration.setdefault("calibration_id", f"router-{args.seed}")
        calibration["training"] = report.to_dict()
        bundle = build_screener_bundle(
            policy,
            vocab_size=int(vocab_size),
            hidden_size=int(hidden.shape[1]),
            calibration=calibration,
        )
        bundle.write(args.out)
    except KeyError as exc:
        print(f"draftslice screen train-router: missing key {exc}")
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"draftslice screen train-router: {exc}")
        return 2

    report_path = Path(args.out) / "training_report.json"
    report_path.write_text(
        json.dumps(
            {"screener": policy.name, "vocab_size": int(vocab_size), **report.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.out} (loss {report.initial_loss:.4f} -> "
        f"{report.final_loss:.4f}, val={report.validation_loss})"
    )
    return 0


def _bench_report(args: argparse.Namespace) -> int:
    try:
        metadata = {} if args.metadata is None else json.loads(args.metadata)
        if not isinstance(metadata, dict):
            raise ValueError("--metadata must be a JSON object")
        manifest, report = build_release(
            args.name,
            args.input,
            out=args.out,
            metadata=metadata,
        )
    except (ReleaseError, json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"draftslice bench report: {exc}")
        return 2
    print(
        f"wrote {args.out} and {Path(args.out).with_suffix('.manifest.json')} "
        f"({len(manifest.files)} files, commit={manifest.git_commit or 'unknown'})"
    )
    if report.count("\n## "):
        print(f"  sections: {report.count(chr(10) + '## ')}")
    return 0


def _bench_validate(args: argparse.Namespace) -> int:
    try:
        manifest = ReleaseManifest.load(args.manifest)
        problems = verify_manifest(manifest, check_hashes=not args.no_hash)
    except (ReleaseError, json.JSONDecodeError, OSError) as exc:
        print(f"draftslice bench validate: {exc}")
        return 2
    if problems:
        for problem in problems:
            print(f"  {problem}")
        return 2
    print(f"ok: {len(manifest.files)} files verified for {manifest.name}")
    return 0


def _bench_compare(args: argparse.Namespace) -> int:
    try:
        base = ReleaseManifest.load(args.base)
        candidate = ReleaseManifest.load(args.candidate)
        report = compare_manifests(base, candidate)
    except (ReleaseError, json.JSONDecodeError, OSError) as exc:
        print(f"draftslice bench compare: {exc}")
        return 2
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "compile":
        return _compile(args)
    if args.command == "bench":
        if args.bench_command == "accuracy":
            return _bench_accuracy(args)
        if args.bench_command == "kernel":
            return _bench_kernel(args)
        if args.bench_command == "constraint":
            return _bench_constraint(args)
        if args.bench_command == "e2e":
            return _bench_e2e(args)
        if args.bench_command == "factorial":
            return _bench_factorial(args)
        if args.bench_command == "report":
            return _bench_report(args)
        if args.bench_command == "validate":
            return _bench_validate(args)
        if args.bench_command == "compare":
            return _bench_compare(args)
        if args.config:
            print("draftslice bench: config-driven runners land in M2; no run started.")
            return 2
        print("draftslice bench: provide a subcommand (for example 'accuracy').")
        return 2
    if args.command == "screen":
        if args.screen_command == "train-router":
            return _screen_train_router(args)
        print("draftslice screen: provide a subcommand (for example 'train-router').")
        return 2
    if args.command == "serve":
        print("draftslice serve: the engine adapters land in M4; no server started.")
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


__all__ = ["build_parser", "main"]
