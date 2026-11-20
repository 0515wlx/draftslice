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

**Pre-alpha.** This repository currently contains the project skeleton and
stable interfaces only. Research code, experimental data, benchmark results,
and work-in-progress designs are intentionally not part of this repository.

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

See [docs/design.md](docs/design.md) for the public design notes.

## Install

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
draftslice --help
draftslice compile --schema tool.json --out plan.json
draftslice bench --config configs/example.toml
draftslice serve --config configs/example.toml
```

The CLI is a pre-alpha skeleton; commands currently validate arguments and
fail with a clear "not implemented" message.

## Roadmap

- **M0** project skeleton, interfaces, CI (this repository)
- **M1** grammar-state plan compiler and plan format
- **M2** slice cache and draft-head kernels
- **M3** vLLM and SGLang integrations
- **M4** evaluation on EAGLE / DFlash / MTP with structured workloads

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
