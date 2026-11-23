# DraftSlice

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/0515wlx/draftslice/actions/workflows/ci.yml/badge.svg)](https://github.com/0515wlx/draftslice/actions/workflows/ci.yml)

**Exact, constraint-aware draft-head slicing for speculative decoding.**

DraftSlice targets the LM head of EAGLE / DFlash / MTP draft models. When the
output is constrained by a grammar or JSON schema, most vocabulary rows can
never be proposed. DraftSlice compiles the grammar state into a **head plan**
(a slice of the draft head), executes only the valid rows, caches the slice
across requests and steps, and exposes its cost to the draft scheduler.

> Don't score what you can't propose.

The system is evaluated at the **server** level (throughput, TTFT/ITL, validity,
and cost), not by acceptance length alone. `tau` is a tunable acceptance metric:
when reducing draft or verification work lowers server latency, DraftSlice may
deliberately trade some `tau` for a better operating point. The target verifier
remains exact.

## Status

**Pre-alpha.** The public package contains the plan core (strict and calibrated
compilation, serialization, compiler-level plan cache, reference executor), the
runtime pieces (cost model, vocabulary layout, candidate-set and packed-row
caches, union/tree executor, acceptance-aware scheduler), a unified screener
registry and screened plan compilation (strict, constrained-calibrated, and
free-form calibrated), fixed-clock Torch/Triton kernels with shape dispatch,
CUDA-graph replay and opt-in autotuning, an xgrammar constraint adapter with B3
replay, a vLLM plugin with a strict constrained draft path, a per-position
grammar shadow, and an opt-in constrained/free-form screener hook, a B4
end-to-end runner, and a fair offline baseline suite
including CFHead/FlashHead/VocabTrim-style reference policies. M5 tooling now
includes group-marginal router training, verified screener artifacts, a
factorial evaluator, and screened-plan cache keys; M6
includes hashed release manifests, Markdown reports, B5 comparison, and an
SGLang general-plugin hook (draft block position 0, validated end to end with
Qwen3.5 built-in MTP). The vLLM DFlash
adapter slices block position 0 exactly and has a real Qwen3.5-4B run. The
public DFlash draft
`z-lab/Qwen3.5-4B-DFlash` is verified with vLLM's built-in `dflash` path;
`draftslice_dflash` slices block position 0 with the exact worker shadow, while
later positions stay on the full head. Per-position slicing after block position
0 is future work. The remaining release work is a multi-seed result run on the
prescribed datasets and real SGLang EAGLE validation.

The current engine path exercises the strict grammar slice directly (worker
shadow + candidate rows + fused kernel). The offline plan/compiler, cost model,
scheduler, union executor, and candidate/row caches are implemented and tested,
but they are **not yet the production engine path**; wiring them into the engine
is tracked in [docs/architecture.md](docs/architecture.md). The production
control plane has started: `runtime/reasons.py` (stable fallback codes),
`runtime/trace.py` (`trace/1` step records), `runtime/profile.py` (hashed
`profiles/1` profit table) and `runtime/config.py` (`config/1` TOML + env
overlay). The policy that consumes them is the next milestone. Research code,
experimental data, and benchmark results are intentionally not part of this
repository.

> **Production default (2026-11-22):** built-in speculative decoding; DraftSlice
> is a measured per-cell profile, not a blanket structured path. On Qwen3.5-4B at
> B=8 the strict slice is a regression (0.767× on a mixed workload), so the
> default profile table enables no B=8 cells. See
> [docs/production-architecture.md](docs/production-architecture.md).

> **Lean exact-candidate mode (2026-11-23):** `DRAFTSLICE_VLLM_LEAN=1` is an
> opt-in throughput mode for eager, TP=1, bf16 MTP with structured traffic. On
> the production chat/tool shape (Qwen3.5-4B, B=1, k=8) it measures **+6.1%**
> over built-in MTP (127.03 → 134.76 tok/s, single server) with identical greedy
> outputs. It caches device payloads, reuses the output buffer, and drops
> per-step contracts/policy/plan from the hot path; an uncalibrated cost
> artifact disables slicing at startup. See
> [docs/lean-slice.md](docs/lean-slice.md).

## Scope

- Draft-side LM head only; the target verifier stays exact.
- Exact grammar/schema-constrained row selection (no dropped valid tokens).
- Slice/plan cache across requests and decode steps.
- Draft-aware scheduling: draft depth, tree width, verify width.
- Composable with low-rank, subset, or sparse draft heads for free-form states.

## Non-goals

- Approximating the target LM head.
- Replacing general draft-head compression (e.g. SlimSpec or SpecVocab).
- Lossy speculative verification.

## Why the draft head

For a speculative decoding step,

```text
T_draft = T_backbone + T_head
S       = (tau + 1) / (T_draft / T_target + 1)
```

For single-layer EAGLE/MTP drafters the LM head can dominate `T_draft`; for
DFlash it is smaller but still material. A full-vocabulary projection followed
by a grammar mask pays for many rows that can never be proposed. DraftSlice
turns the grammar state into an executable slice of the head before projection.

## Architecture

```text
grammar / schema
      |
      v
 FSM state ------> HeadPlan (valid ids, row layout, kernel)
      |                    |
      |                    v
      |              SliceCache (state -> plan, hot rows)
      |                    |
      v                    v
 draft backbone -----> slice_gemv / slice_gemm / full fallback
      |
      v
 draft tree ------> target verifier (exact)
```

See [docs/design.md](docs/design.md) for the public design notes. The target
component boundaries, plan formats, and milestone definitions are in
[docs/architecture.md](docs/architecture.md). The benchmark tiers, metrics, and
measurement protocols are in [docs/benchmark.md](docs/benchmark.md). The gap
analysis and competitive plan are in
[docs/competitive-plan.md](docs/competitive-plan.md).

## Install

```bash
python -m pip install -e ".[dev]"

# GPU kernels and performance tests
python -m pip install -e ".[dev,kernels]"

# vLLM plugin
python -m pip install -e ".[dev,kernels,vllm]"
```

## CLI

```bash
draftslice --help
draftslice compile --valid-ids 1,3,5 --vocab-size 8 --out plan.json
draftslice compile --schema tests/fixtures/token_set.json --out plan.json
draftslice bench accuracy --input records.jsonl --out result.json
draftslice bench kernel --gpu 0 --batch 1,8,32,64 \
  --rows 128,1024,4096,8192,16384 --out kernel.json
draftslice bench constraint --schema schema.json --tokenizer /path/to/model \
  --max-steps 64 --out b3.json
draftslice bench e2e --model /path/to/model --method draftslice_mtp \
  --batch 1,8 --out b4.json
draftslice bench --config configs/example.toml
draftslice serve --config configs/example.toml
```

The CLI is pre-alpha. `compile` writes a real plan from the reference constraint
fixture backend, `bench accuracy` evaluates cached candidate records, and
`bench kernel` measures full/gather/fused kernels with the SM clock locked. The
config-driven benchmark runner and `serve` are still placeholders.

## vLLM plugin

Installing the package registers a `vllm.general_plugins` entry point. The plugin
adds `draftslice_mtp`, `draftslice_eagle3`, and `draftslice_dflash` speculative
methods:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"draftslice_mtp","num_speculative_tokens":1}'
```

The worker-side xgrammar shadow advances with the tokens the target actually
sampled, so draft step 0 gets the exact next-state valid set rather than the
one-step-ahead scheduler bitmask. DFlash slices block position 0 the same way;
later block positions stay on the full head. **Slicing is profile-gated**
(`DRAFTSLICE_PROFILE_MODE=observe|enforce`): the default is built-in, because
free-form and some structured workloads regress when slicing is enabled
globally. Covered engine paths: TP>1 via per-rank vocabulary shards with a
cross-rank max-reduce of the partial logits and an agreed full-head row set (the
2026-11-22 MoE TP=2 hang was rank-local logits; it is fixed and has a two-rank
regression test), quantized heads via candidate-row dequantization, reasoning
parsers via a boundary-gated shadow, local-argmax reduction via a sliced
`get_draft_top_tokens` override, and FULL decode graphs via a pre-replay static
packed bitmask. Every decision emits a `DraftContract` with the fail-closed
reason; inspect `draftslice_stats()` for `fallback_reasons`, `last_contract`,
`tp_combined_calls`, and TP shard fields. The target verifier remains exact. See
[docs/architecture.md §16](docs/architecture.md#16-vllm-integration) for the hook
points and limitations.

The production policy is table-driven:

```bash
vllm serve ... --speculative-config '{"method":"draftslice_mtp","num_speculative_tokens":8}' \
  DRAFTSLICE_PROFILE_TABLE=profiles/4b.json \
  DRAFTSLICE_PROFILE_MODE=enforce          # off | observe | enforce
```

`off`/`observe` execute the stock path with no shadow or valid-set work;
`enforce` slices only rows whose exact `(model, engine, method, workload, batch,
k, tp, quant, graph, valid)` cell is enabled. A missing, disabled, or
engine-mismatched cell is built-in; a missing table file or hash mismatch is a
hard startup error. vLLM only exposes the grammar format, so a deployment maps
formats onto workload classes explicitly, e.g.
`DRAFTSLICE_WORKLOAD_MAP='{"json":"tool_call","json_object":"tool_call"}'`.

DFlash support: `draftslice_dflash` slices block position 0 with the exact
worker shadow. Later block positions stay on the full head because their grammar
state depends on draft tokens sampled inside the same parallel block. See
[docs/architecture.md §18.5](docs/architecture.md#185-dflash).

## Roadmap

- **M0** project skeleton, interfaces, CI (this repository)
- **M1** plan core: strict compiler, plan format, reference executor, and
  conformance suite
- **M2** runtime, slice cache, cost model, and scheduler
- **M3** optimized slice kernels: shape dispatch, row-block GEMV, tensor-core
  GEMM, baseline kernel parity, CUDA-graph replay, union/tree execution, and
  runtime autotuning (done); a tree-specific fused union kernel and engine-owned
  graph buckets remain
- **M4** vLLM integration: plugin, strict draft-step-0 path, worker shadow, and
  structured end-to-end runs (done), per-rank TP>1 shard slicing with replicated
  logits, a concrete `DraftContract`, quantized heads, reasoning parsers,
  local-argmax and FULL-graph contracts (done); the production policy/profile
  table wiring and server-level tau/latency tuning are next
- **M5** fair baseline suite (implemented, including CFHead/FlashHead reference
  policies), artifact/partition/conformal core, calibrated plan compilation,
  the acceptance model, router training, screener artifacts, and the factorial
  matrix (implemented); published multi-seed calibrated runs next
- **M6** SGLang general-plugin hook (draft block position 0, validated with
  Qwen3.5 built-in MTP), DFlash
  block-position-0 slicing (real 4B run), release manifests, reports, B5
  comparison, and CI gates (implemented); per-position DFlash and multi-seed
  release run next

Each milestone has a definition of done in
[docs/architecture.md](docs/architecture.md); the benchmark gates are in
[docs/benchmark.md](docs/benchmark.md).

## Related work

- SlimSpec: low-rank draft LM head — <https://arxiv.org/abs/2605.10453>
- SpecVocab: context-dependent draft vocabulary — <https://arxiv.org/html/2602.13836>
- vLLM context-aware sparse LM head — <https://github.com/vllm-project/vllm/pull/67897>
- Grammar-aware draft sampling in vLLM — <https://github.com/vllm-project/vllm/pull/47885>
- CASD: contract-aware speculative decoding — <https://www.computer.org/csdl/proceedings-article/sse/2026/119800a290/2joOwwfu2Ji>

## Writing

- [The 6% that survived: exact draft-head slicing in a real serving stack](docs/blog/2026-11-when-grammar-aware-drafting-pays.md)
  — why the 1.19× experiment was 1.00× in production, and which optimization
  remained.

## Development environments

vLLM and SGLang pin different `torch` patch releases, so they cannot share one
Python environment. The repository ships one main environment and two engine
siblings:

```bash
bash scripts/setup_envs.sh          # wlx_draftslice + _vllm + _sglang
# or install one at a time:
pip install -r requirements-base.txt    # wlx_draftslice (no engine)
pip install -r requirements.txt         # wlx_draftslice_vllm (vLLM 0.34.0)
pip install -r requirements-sglang.txt  # wlx_draftslice_sglang (SGLang 0.5.23)
```

See [docs/environment.md](docs/environment.md) for the pin matrix, exact lock
files, and the `pip check`/test gates. Run `bash scripts/verify_env.sh` to
validate all three environments in one command.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For security issues, see
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
