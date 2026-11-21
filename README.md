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
SGLang boundary (engine validation is blocked by a missing `sgl_kernel` in the
reference install). The public DFlash draft
`z-lab/Qwen3.5-4B-DFlash` is verified with vLLM's built-in `dflash` path;
DraftSlice's DFlash adapter is future work. The remaining release work is a
multi-seed result run on the prescribed datasets and validation of the SGLang
engine hook. Research code, experimental data, and benchmark results are
intentionally not part of this repository.

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
adds `draftslice_mtp`, `draftslice_eagle3`, and a safe pass-through
`draftslice_dflash` speculative method:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"draftslice_mtp","num_speculative_tokens":1}'
```

The first cut slices the draft LM head for draft step 0 using the captured
structured-output bitmask. **Slicing is opt-in** (`DRAFTSLICE_VLLM_SLICE=1`)
because the captured bitmask is one grammar state ahead of the draft token: on a
changing grammar it can drop valid draft tokens and lower acceptance. The safe
default is to fall back to the full head. The target verifier remains exact. See
[docs/architecture.md §16](docs/architecture.md#16-vllm-integration-first-cut)
for the hook points and limitations.

DFlash is supported through vLLM's built-in `dflash` path with the public draft
`z-lab/Qwen3.5-4B-DFlash`. DraftSlice registers `draftslice_dflash` as a safe
pass-through: parallel block drafting needs per-position grammar state, so
block-position slicing is not enabled. See
[docs/architecture.md §18.5](docs/architecture.md#185-dflash).

## Roadmap

- **M0** project skeleton, interfaces, CI (this repository)
- **M1** plan core: strict compiler, plan format, reference executor, and
  conformance suite
- **M2** runtime, slice cache, cost model, and scheduler
- **M3** optimized slice kernels: shape dispatch, row-block GEMV, tensor-core
  GEMM, baseline kernel parity, and CUDA-graph replay (done); union/tree
  execution and runtime autotuning next
- **M4** vLLM integration: plugin and strict draft-step-0 path (done); TP/quantized
  hardening and end-to-end structured workloads next
- **M5** fair baseline suite (implemented, including CFHead/FlashHead reference
  policies), artifact/partition/conformal core, calibrated plan compilation,
  the acceptance model, router training, screener artifacts, and the factorial
  matrix (implemented); published multi-seed calibrated runs next
- **M6** SGLang boundary (implemented; engine validation blocked), DFlash
  per-block adapter (blocked on per-position grammar state), release manifests,
  reports, B5 comparison, and CI gates (implemented); multi-seed release run
  next

Each milestone has a definition of done in
[docs/architecture.md](docs/architecture.md); the benchmark gates are in
[docs/benchmark.md](docs/benchmark.md).

## Related work

- SlimSpec: low-rank draft LM head — <https://arxiv.org/abs/2605.10453>
- SpecVocab: context-dependent draft vocabulary — <https://arxiv.org/html/2602.13836>
- vLLM context-aware sparse LM head — <https://github.com/vllm-project/vllm/pull/67897>
- Grammar-aware draft sampling in vLLM — <https://github.com/vllm-project/vllm/pull/47885>
- CASD: contract-aware speculative decoding — <https://www.computer.org/csdl/proceedings-article/sse/2026/119800a290/2joOwwfu2Ji>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For security issues, see
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
