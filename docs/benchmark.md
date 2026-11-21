# DraftSlice benchmark plan

Status: proposed, pre-alpha. This document defines what DraftSlice must measure,
how to measure it, and what counts as a valid result. It does not report
benchmark results.

Related documents:

- [design.md](design.md) — problem statement and scope.
- [architecture.md](architecture.md) — components, modes, and contracts.

## 1. What the benchmark must prove

The benchmark suite exists to answer six claims. Each claim has a dedicated
measurement tier.

| # | Claim | Primary evidence |
|---|---|---|
| C1 | Strict mode never drops a grammar-valid token. | B0/B3 coverage and plan checks; B4 output equality |
| C2 | DraftSlice is on the accuracy/resource Pareto frontier. | B1 head-quality curves at iso-bytes and iso-latency |
| C3 | Slicing and screening reduce measured head latency, not just MACs. | B2 kernel microbench with roofline and effective bandwidth |
| C4 | The gain survives end-to-end speculative decoding. | B4 acceptance length, throughput, and latency |
| C5 | The result is robust to distribution shift and configuration. | B1/B4 robustness matrix; calibration checks |
| C6 | The result is reproducible. | manifests, hashes, seeds, one-command reports |

A result that only reports top-1 accuracy, or only MACs, or only a single
hand-picked configuration, is not sufficient.

## 2. Benchmark principles

1. **Isolate the layer.** Head quality is measured without the target verifier.
   Kernel latency is measured without the draft backbone. End-to-end is measured
   only after both layers are understood.
2. **Compare at iso-resource.** Every method gets a knob sweep. Compare the best
   point at a fixed byte, MAC, or latency budget, not an arbitrary point.
3. **Exactness before speed.** A faster method that violates strict coverage or
   changes constrained output is a failure, not a trade-off.
4. **Report distributions.** Mean-only metrics hide tail behavior. Report
   median, p10/p90/p99, and confidence intervals.
5. **Use independent splits.** Calibration, validation, and test data must be
   separated by document, session, or prompt. No hidden leakage.
6. **Report negative results.** Failed configurations and fallback rates are part
   of the result.
7. **Pre-register the primary metric.** The headline table uses the metric chosen
   before the run; exploratory metrics are labeled as such.
8. **No cherry-picking.** A release report includes all pre-registered workloads,
   including those where DraftSlice loses.

## 3. Benchmark tiers

| Tier | Name | Scope | Hardware | Frequency | Runtime target |
|---|---|---|---|---|---|
| B0 | Conformance | Unit and property tests for plans, kernels, cache, scheduler | CPU (+ tiny GPU) | every PR | < 2 min |
| B1 | Head accuracy | Offline top-k / order accuracy and candidate cost | 1 GPU | nightly, release | 10–60 min |
| B2 | Kernel | Head-only latency, bandwidth, roofline, cost-model error | 1 GPU | nightly, release | 10–30 min |
| B3 | Constraint plan | Grammar/schema state coverage, compile/cache behavior | CPU or 1 GPU | nightly, release | 5–30 min |
| B4 | End-to-end | Speculative decoding with a real engine | 1–8 GPUs | release | hours |
| B5 | Regression | Compare a candidate run against a stored baseline | CI | nightly, release | minutes |

B0–B3 must run without an inference engine. B4 is the only tier that imports
vLLM or SGLang.

## 4. Workloads and datasets

Datasets are versioned and content-hashed. Data files stay outside the repository;
the repository stores adapters, manifests, and download instructions.

The primary measurement mix is **UltraChat-200K (`train_sft`) + ShareGPT**,
referred to by the logical name `eagle3-mix-v1`. Configs and result manifests
store only that name and a content hash. The local data root is supplied at
runtime (for example through an operator environment variable such as
`DRAFTSLICE_DATA_ROOT`) and must never be committed; public configs must not
contain absolute dataset paths.

### 4.1 Head-quality datasets (B1)

| Name | Content | Split rule | Intended use |
|---|---|---|---|
| `teacher-forced-mix-v1` | English web, code, math, multilingual QA, chat/instruct | document-level; train/calib/test disjoint | general free-form head quality |
| `on-policy-gen-v1` | hidden states from the drafter's own generated tokens | prompt/session-level | deployment-distribution shift |
| `eagle3-mix-v1` | UltraChat-200K `train_sft` + ShareGPT, MTP states | session-level | EAGLE-3 / MTP draft head |
| `structured-v1` | JSON Schema, tool calls, regex, SQL/CFG | schema-level; unseen schemas in test | constrained slicing |

`structured-v1` must bucket states by valid-set size:

- tiny: `|V| < 0.1%` of vocabulary;
- small: `0.1%–1%`;
- medium: `1%–10%`;
- large: `10%–50%`;
- huge: `> 50%` (slicing may not pay off; fallback expected).

The bucket distribution is reported with the results. A benchmark that only uses
tiny valid sets overstates the benefit.

### 4.2 Kernel datasets (B2)

No text is needed. The kernel runner synthesizes shapes from the matrix in §8 and
uses a fixed hidden-state sample for numerical checks. It must also include at
least one real candidate set from B1/B3 so that row locality is realistic.

### 4.3 End-to-end workloads (B4)

| Workload | Examples | Why it matters |
|---|---|---|
| Tool calling / JSON Schema | BFCL-style requests, function schemas, nested objects | large valid-set variation; strictness |
| Grammar-constrained | SQL, regex, custom CFG | many states, tiny valid sets |
| Code | HumanEval/MBPP-style prompts | long outputs, free-form states |
| Math | GSM8K-style prompts | free-form, high-entropy |
| Chat | open-ended instructions | free-form baseline |
| Mixed | schema turns followed by free-form turns | mode transitions |

Use openly licensed datasets where possible. If a dataset cannot be redistributed,
store only the hash, adapter name, and reproduction instructions.

### 4.4 Synthetic edge cases

- `|V| = 1` and `|V| = 2`.
- `|V|` just below and just above every fallback threshold.
- Alternating schema states across adjacent tokens.
- Unicode and tokenizer-boundary cases.
- Batch and tree shapes with zero or one candidate row.
- Cache-cold and cache-warm requests.
- Very long prompts and very long constrained outputs.

## 5. Baselines

Every B1/B4 table includes the following where applicable:

| Baseline | Purpose |
|---|---|
| `full` | exact oracle; latency and accuracy reference |
| `static-M` | frequency-truncated head at several `M` |
| `svd-r` | SVD low-rank head at several ranks |
| `trained-lr-r` | trained low-rank head at several ranks |
| `cfhead-like` | tile/index screen, reimplemented fairly |
| `flashhead-like` | cluster/centroid screen |
| `dynaspec-like` | MLP router + cluster shortlist |
| `draftslice-strict` | grammar exact slicing only |
| `draftslice-calibrated` | screen + exact rescore, conformal budget |
| `draftslice-hybrid` | constrained calibrated mode |
| `oracle-leaf-mass` | non-deployable upper bound on leaf screening |
| `dflash-block-8/16` | official `z-lab/Qwen3.5-4B-DFlash` end-to-end baseline |
| `no-spec` | end-to-end non-speculative baseline |

Implemented reference policies live in `draftslice.bench.baselines` and
`draftslice.bench.official_baselines` and share one `CandidatePolicy` interface:
`FullPolicy`, `StaticPolicy`, `LowRankPolicy` (SVD/trained), `GroupPolicy`
(centroid shortlist), `RouterPolicy` (MLP shortlist), `OracleGroupPolicy` (upper
bound), `CFHeadPolicy` (tile codebook + index map), `FlashHeadPolicy`
(balanced-cluster probes), and `make_vocabtrim_policy`. `draftslice.bench.baseline_runner`
runs them over the same hidden states and oracle top-k and writes a
`baseline_matrix/1` result. The CFHead/FlashHead/VocabTrim implementations are
reference reimplementations and must be labelled as such.

Fairness rules:

- Same model, tokenizer, dtype, hardware, and batch.
- Same kernel effort where possible; if a baseline uses a reference kernel and
  DraftSlice uses a fused kernel, label the comparison as "reference
  implementation" and report both.
- Baselines get their own knob sweep. Do not compare DraftSlice's best point to a
  baseline's default point.
- Reimplementations are clearly marked; published numbers from other papers are
  not mixed with local measurements.

## 6. Metrics

### 6.1 Notation

For a query with hidden state `h`:

- `V(h)`: grammar/schema-valid token set, or the full vocabulary.
- `U(h)`: the measurement universe: `V(h)` for constrained modes, and the full
  vocabulary for `freeform_calibrated` and `full`.
- `T_k(h)`: true top-k token ids under the full head, **restricted to `U(h)`**.
  For constrained modes this is the valid top-k, not the unconstrained top-k.
- `A(h)`: candidate set.
- `E(h)`: executed rows.
- `C_j(h) = 1[T_j(h) ⊆ A(h)]`, for `j = 1..k`.
- `R_k(h) = |A(h) ∩ T_k(h)| / k`.

Using the valid top-k matters: a strict plan that covers every valid token should
score 100% on constrained top-1 coverage even when the unconstrained top-1 is
grammar-invalid.

### 6.2 Accuracy and coverage

| Metric | Definition | Notes |
|---|---|---|
| `top1` | `P(T_1 ⊆ A)` | exact because scoring inside `A` is exact |
| `set_recall@k` | `E[R_k]` | expected fraction of true top-k in the candidate set |
| `prefix_coverage@j` | `P(T_j ⊆ A)` | stricter than recall |
| `allk` | `P(T_k ⊆ A)` | all top-k covered |
| `OA` | `Σ_j w_j · prefix_coverage@j` | order accuracy; default uniform and `1/log2(j+1)` weights |
| `valid_recall` | `|A ∩ V| / |V|` | must be 1.0 in strict mode |
| `valid_precision` | `|A ∩ V| / |A|` | must be 1.0 in constrained modes |
| `plan_equals_valid` | `A == V` | strict-mode boolean |
| `empirical_coverage` | `P(T_k ⊆ A)` for the claimed delta | calibrated mode |
| `coverage_gap` | `empirical_coverage - (1 - delta)` | report with CI |

`top1`, `top5`, and `OA` are not interchangeable. Report all three for a headline
table. `allk` is reported separately because it is much more expensive.

### 6.3 Resource metrics

| Metric | Definition |
|---|---|
| `m` | mean candidate-set size `|A|` |
| `executed_m` | mean executed rows `|E|` |
| `static_bytes` | bytes read every step (router, tables, codebooks) |
| `dynamic_bytes` | `2 · d · |E|` for bf16 |
| `macs` | `router_macs + |E| · d` |
| `alu_ops` | index/LUT/scan operations |
| `mac_ratio` | `macs / (V · d)` |
| `byte_ratio` | `(static_bytes + dynamic_bytes) / (2 · V · d)` |
| `candidate_ratio` | `|E| / V` |
| `plan_compile_us` | p50/p90 plan compilation time |
| `cache_hit_rate` | per cache, cold and warm |

MACs are hardware-independent and useful for cross-paper comparison. Latency is
the deployment metric. Report both.

### 6.4 Latency metrics

| Metric | Definition |
|---|---|
| `latency_median_us` | median over repetitions |
| `latency_p90_us`, `latency_p99_us` | tail latency |
| `effective_bandwidth_gbps` | bytes moved / latency |
| `roofline_efficiency` | measured time / model-predicted ideal |
| `launch_overhead_us` | extrapolated `t0` from a size sweep |
| `speedup_vs_full` | full latency / method latency |
| `speedup_vs_reference_kernel` | reference kernel latency / optimized latency |
| `cost_model_mape` | mean absolute percentage error of predicted vs measured |

### 6.5 Acceptance and end-to-end metrics

| Metric | Definition |
|---|---|
| `tau` | mean accepted draft tokens per verification step |
| `tau_p50` / `tau_p90` | acceptance-length distribution |
| `acceptance_rate_by_depth` | per draft-tree depth |
| `sumC6` | `Σ_{j=1..6} P(prefix_j ⊆ A)`; offline proxy |
| `delta_target_top1` | target top-1 agreement drop vs full head |
| `delta_tau_prefix` | 6-step prefix-acceptance drop; offline proxy |
| `throughput_tok_s` | generated tokens per second |
| `ttft_ms` | time to first token |
| `itl_ms` | inter-token latency, p50/p90/p99 |
| `output_validity` | fraction of outputs that satisfy the schema/grammar |
| `task_metric` | benchmark-specific quality (pass@1, exact match, tool-call success) |
| `verifier_overhead_us` | target verification time attributable to plans |
| `speedup_vs_full` | end-to-end baseline / method |
| `speedup_vs_no_spec` | non-speculative baseline / method |

For strict constrained decoding, `output_validity` must be 1.0 by construction. If
it is not, the run is invalid and the bug must be fixed before performance is
reported.

### 6.6 Quality-invariance checks

Speed is meaningless if output quality changes silently.

- **Greedy equality:** in strict mode, the full sequence must match full-head
  constrained decoding token for token.
- **Distribution test:** for a fixed prompt at nonzero temperature, draw `N`
  samples from both paths and compare next-token frequencies with a chi-square or
  total-variation test. Report the test and sample size.
- **Task metric:** report the task metric with a confidence interval. A speedup
  that costs more than the pre-registered quality budget is a regression.

## 7. Measurement protocols

### 7.1 B0 conformance

- Run on every PR and every Python version in CI.
- Property tests: coverage, idempotence, union/intersection, serialization
  round-trip, cache key collisions, permutation bijection.
- Kernel tests: reference vs optimized, all dtypes and shapes, deterministic
  tie-breaking.
- Scheduler tests: strict coverage under every fallback path; deadline handling.
- Failure injection: unknown state, OOM, unsupported kernel, hash mismatch.

### 7.2 B1 head accuracy

1. Load a fixed artifact bundle and a fixed hidden-state dataset.
2. Compute oracle top-k once in float32; cache by `(model, tokenizer, dataset,
   hidden_hash)`.
3. For each method and knob, compute candidate sets on the test split.
4. Compute metrics with a bootstrap over the split unit (document/session).
5. Verify that scoring inside the candidate set is exact by comparing to the
   oracle on the overlap.
6. Report per-domain and aggregate results.

Rules:

- `eval()` mode, no dropout, no gradient.
- Deployment dtype for latency; float32 for oracle logits.
- Calibration data is never part of the test split.
- Report `N`, split unit, and number of unique documents/sessions.
- Report all knob values, not only the best one.

### 7.3 B2 kernel microbenchmark

1. Lock the SM clock to a fixed frequency before any warmup:
   `nvidia-smi -i <gpu> -lgc <mhz>`. Verify the clock with a fresh query. If the
   lock cannot be acquired or verified, fail the performance run; do not report
   an unlocked number as a result.
2. Record the clock snapshot, GPU model, driver, temperature, power, PyTorch,
   CUDA, and Triton versions in the result.
3. Warm up for at least 20 iterations.
4. Time at least 100 iterations with CUDA events.
5. Report median and p90; discard the first warmup block.
6. Unlock the clock in a `finally` block (`nvidia-smi -rgc`).
7. Run one process per GPU; no concurrent jobs.
8. Measure components separately: router, top-k, gather, dot/GEMM, scatter.
9. Measure cold and warm cache; report the cache state.
10. Compare against `torch` reference, naive union, fused union, and the roofline
    ideal.
11. Report coefficient of variation; rerun if it is above the pre-registered
    threshold (for example 5%).

Implemented commands:

```bash
# Perf test: requires root/sudo to lock the SM clock; fails if the lock fails.
DRAFTSLICE_RUN_PERF=1 DRAFTSLICE_GPU=0 pytest tests/perf -q -s

# Kernel runner with the same fixed-clock policy.
draftslice bench kernel --gpu 0 --batch 1,8 --rows 1024,4096,8192 \
  --warmup 20 --iterations 100 --out runs/kernel.json

# B3 constraint replay from recorded states or a real xgrammar trajectory.
draftslice bench constraint --input states.jsonl --out runs/b3.json
draftslice bench constraint --schema schema.json \
  --tokenizer /path/to/model --max-steps 64 --out runs/b3_xgrammar.json

# B4 end-to-end; run one method per process for isolation.
draftslice bench e2e --model /path/to/model --method mtp \
  --batch 1,8 --out runs/b4_mtp.json
draftslice bench e2e --model /path/to/model --method draftslice_mtp \
  --batch 1,8 --out runs/b4_draftslice_mtp.json
```

The result records `clock_fixed` and the verified `clock.sm_clock_mhz`. A reader
must treat `clock_fixed=false` as an exploratory measurement, not a benchmark
result. `draftslice bench kernel --allow-unlocked` exists only for debugging.

The kernel result uses `schema_version = "kernel_bench/1"` and contains:

- `gpu` and `clock`: device name, driver, SM/memory clocks, temperature, power;
- `clock_fixed`: whether the SM clock was locked and verified;
- `torch_version`, `triton_version`, and `triton_available`;
- `graph`: availability plus capture, replay, fallback, and disabled-reason
  counters;
- one row per `(kernel, mode, batch, rows)` with `median_us`, `p90_us`, `cv`,
  `weight_gbps`, `speedup_vs_full`, `speedup_vs_torch_gather`,
  `speedup_vs_eager` for graph rows, and `numerical_max_abs_diff` where
  applicable.

Every implementation is measured in both eager and CUDA-graph mode when graphs
are available: `full`, `torch_gather`, `triton_slice`, and their `_graph`
variants. A comparison that enables graph replay for only one side is invalid.

`draftslice bench kernel` defaults to `B=1,8` and
`m=128,1024,4096,8192,16384`; use `--batch` and `--rows` for the full matrix,
`--kernels` to select implementations, and `--no-cuda-graph` to omit the replay
rows.

Cost-model validation:

- Fit the model on a calibration set of shapes.
- Report error on a disjoint holdout set.
- Do not fit and report on the same shapes.

### 7.4 B3 constraint plan quality

1. Generate real constrained trajectories with the grammar backend.
2. Record every grammar state, its valid set, and the transition that produced it.
3. Compile plans for the recorded states.
4. Measure compile time, plan size, cache hit rate, and fallback rate.
5. Verify `A == V` for strict plans and `A ⊆ V` for constrained calibrated plans.
6. Replay the trajectory and verify that the plan at each step covers the valid set.

Report the distribution of valid-set sizes and grammar-state reuse. A cache hit
rate measured on synthetic random states is not meaningful.

### 7.5 B4 end-to-end

Controls:

- Same engine commit, model weights, tokenizer, sampler, seed, max tokens, and
  stop conditions for baseline and method.
- Same batch/concurrency schedule.
- Warm up the engine; measure steady state.
- Separate prefill/TTFT from decode/ITL.
- Report request-level and token-level latency.
- Run at least three repetitions per configuration; report variance.

Procedure:

1. Run full-head baseline and DraftSlice on the same workload.
2. Verify output validity and, for strict mode, greedy equality.
3. Measure acceptance length and its distribution.
4. Measure throughput, TTFT, ITL, and total latency.
5. Compute speedup per workload and batch.
6. Report the head's share of decode time so the speedup is interpretable.

If the engine cannot run a configuration, report it as unsupported rather than
dropping it silently.

The first vLLM matrix compares `mtp` with `draftslice_mtp` on the same model,
sampler, seed, batch, and workload. Slicing is disabled by default because the
worker's captured grammar bitmask is one state ahead of the draft token; enabling
it with `DRAFTSLICE_VLLM_SLICE=1` is an experiment, not a strict claim. The
default `draftslice_mtp` path is a pass-through with explicit counters, so the
B4 gate first checks no regression, then evaluates the experimental slice path.

Report the speculator's `sliced_calls`, `fallback_calls`, `sliced_rows`,
`full_rows`, `slice_disabled_calls`, and `mask_stride`
(`DRAFTSLICE_VLLM_DEBUG=1`) alongside tau and throughput. A strict claim needs
greedy equality and 100% output validity; a lower tau with the same validity is a
regression, not a win. The current safe default is expected to match the
baseline; the experimental slice path must not be reported as a win until a
worker-side grammar shadow fixes the state timing.

The DFlash matrix uses the public `z-lab/Qwen3.5-4B-DFlash` draft, block sizes 8
and 16, and the same target/workload. The first DFlash adapter is not allowed to
slice block positions from a single captured bitmask; it must either receive a
per-position grammar state or fall back to the full head. Report block-position
sliced/fallback counts and acceptance per position.

## 8. Benchmark matrix

A full Cartesian product is too large. Use a staged design.

### Stage 1 — method screening (B1)

- Methods: all B1 baselines plus DraftSlice modes.
- Datasets: `teacher-forced-mix-v1`, `on-policy-gen-v1`, `eagle3-mix-v1`.
- Fixed anchors: 1%, 3%, 5%, 10%, 20% of full-head bytes and MACs.
- Output: accuracy vs bytes/MACs Pareto curves.

### Stage 2 — constraint behavior (B1 + B3)

- Methods: `draftslice-strict`, `draftslice-hybrid`, `full`.
- Datasets: `structured-v1` across all valid-set buckets.
- Knobs: schema family, valid-set size, grammar depth.
- Output: strict coverage, candidate ratio, compile/cache behavior, fallback rate.

### Stage 3 — kernel selection (B2)

- Shapes: `B ∈ {1, 8, 32, 64}`; `m ∈ {1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k,
  248k}`.
- Layouts: native, leaf-contiguous, packed.
- Implementations: reference, naive union, fused union, ideal.
- Output: latency, effective bandwidth, roofline efficiency, cost-model error.

### Stage 4 — end-to-end (B4)

- Configurations: the top 2–3 head designs from Stage 1/2, with the best kernel
  from Stage 3 per batch.
- Workloads: tool calling, grammar-constrained, code, math, chat, mixed.
- Batches/concurrency: 1, 8, 32.
- Output: acceptance length, speedup, validity, quality.

### Stage 5 — robustness and ablations

- On-policy vs teacher-forced.
- Calibration size and distribution shift.
- Cross-model (for example Qwen3.5-9B/27B, Llama-3.1-8B) where feasible.
- Cross-tokenizer.
- Ablations: exact rescoring on/off, soft vs hard router targets, `K`, `C`,
  router hidden size, layout, fused kernel, cache on/off, union vs per-row,
  static-prior union, grammar-only vs calibrated.
- Output: a table of what matters and what does not.

## 9. Statistics

- **Split unit:** document for pretraining text, session/prompt for on-policy
  data, schema for structured data.
- **Confidence intervals:** 95% bootstrap over the split unit. Report the number
  of units and the number of queries.
- **Paired comparisons:** paired bootstrap for method vs baseline on the same
  queries.
- **Multiple comparisons:** Holm–Bonferroni for the pre-registered headline
  claims.
- **Minimum sample sizes:** at least 5,000 teacher-forced states, 2,000 on-policy
  states, 1,000 structured prompts, and 5 latency repetitions per shape, unless a
  power analysis justifies a different number.
- **Latency:** report median and p90/p99; do not summarize with the mean alone.
- **Significance is not enough:** report effect size and the practical decision
  (for example bytes saved at the same accuracy).
- **No p-hacking:** write down the primary metric and workload set before the run.

## 10. Result schema

The B1 offline harness consumes one JSONL record per query:

```json
{"candidates": [1, 2, 3], "truth": [1, 2, 3, 4, 5], "valid": [1, 2, 3, 4, 5], "domain": "tool_call"}
```

`truth` is the full-head top-k restricted to the measurement universe; `valid`
is present for constrained records. The harness writes a versioned summary:

Every run writes a self-describing result. A result without a manifest is not
accepted.

```json
{
  "schema_version": "bench/1",
  "bench_id": "head-a100-2026-...-a1b2c3",
  "kind": "head_accuracy",
  "created_at": "2026-01-01T00:00:00Z",
  "git": {
    "commit": "faa5bd2",
    "dirty": false
  },
  "env": {
    "python": "3.11.9",
    "torch": "2.x",
    "cuda": "12.x",
    "driver": "5xx.xx",
    "gpu": "NVIDIA A100-SXM4-80GB",
    "triton": "3.x"
  },
  "model": {
    "name": "Qwen3.5-4B",
    "revision": "...",
    "vocab_size": 248320,
    "hidden_size": 2560,
    "dtype": "bf16",
    "tokenizer_hash": "sha256:..."
  },
  "dataset": {
    "name": "teacher-forced-mix-v1",
    "version": "1.0",
    "hash": "sha256:...",
    "split": "test",
    "n": 5000,
    "units": 312
  },
  "config": {
    "mode": "freeform_calibrated",
    "K": 1024,
    "C": 16,
    "router_hidden": 512,
    "delta": 0.05,
    "layout": "leaf_contiguous",
    "batch": 1
  },
  "method": {
    "name": "draftslice-calibrated",
    "artifact_bundle": "sha256:...",
    "kernel": "fused"
  },
  "metrics": {
    "top1": 0.0,
    "set_recall@5": 0.0,
    "prefix_coverage@5": 0.0,
    "OA": 0.0,
    "all5": 0.0,
    "m": 0.0,
    "executed_m": 0.0,
    "byte_ratio": 0.0,
    "mac_ratio": 0.0,
    "empirical_coverage": 0.0,
    "coverage_gap": 0.0
  },
  "uncertainty": {
    "method": "bootstrap",
    "unit": "document",
    "level": 0.95,
    "intervals": {}
  },
  "latency_us": {
    "median": 0.0,
    "p90": 0.0,
    "p99": 0.0,
    "cv": 0.0
  },
  "warnings": []
}
```

The numbers above are placeholders. Real results are written by the harness.

Result storage:

```text
runs/<bench_id>/
  manifest.json
  config.toml
  metrics.json
  per_query.jsonl
  latency.jsonl
  plans/              # optional sampled plans
  report.md           # generated
```

`per_query` records are optional for large runs but required for any published
claim so the analysis can be audited.

## 11. CLI and configuration

```bash
# Compile and inspect a plan (reference fixture backend)
draftslice compile --schema tests/fixtures/token_set.json --out plan.json

# Implemented: B1 over cached candidate records
draftslice bench accuracy --input records.jsonl --out result.json --k 5

# Planned: config-driven tiers
draftslice bench conformance --config benchmarks/configs/b0.toml
draftslice bench accuracy  --config benchmarks/configs/b1_a100.toml
draftslice bench kernel    --config benchmarks/configs/b2_a100.toml
draftslice bench constraint --config benchmarks/configs/b3_structured.toml
draftslice bench e2e       --config benchmarks/configs/b4_vllm.toml

# Planned: compare and validate
draftslice bench validate runs/<bench_id>/manifest.json
draftslice bench compare runs/base runs/candidate --out report.md
```

Example B1 configuration:

```toml
[bench]
kind = "head_accuracy"
seed = 0
out = "runs"

[dataset]
name = "teacher-forced-mix-v1"
split = "test"

[model]
name = "Qwen3.5-4B"
dtype = "bf16"

[method]
name = "draftslice-calibrated"
mode = "freeform_calibrated"
artifact_bundle = "artifacts/qwen35-4b/leaf1024-router512"
delta = 0.05

[budget]
# Sweep these; report the full curve, not only the best point.
candidates = [4, 8, 16, 32, 64]
target_oa = [0.90, 0.95]

[hardware]
profile = "a100"
batch = [1, 8, 32, 64]

[report]
pareto = true
per_domain = true
```

## 12. CI, nightly, and release gates

### Pull request

- B0 conformance, lint, type check.
- A tiny synthetic B1 case on CPU to catch end-to-end plumbing errors.
- Skip GPU tiers if no GPU is available; mark the check as skipped.

### Nightly

- B1 accuracy on the cached test splits.
- B2 kernel microbench on the primary GPU.
- B3 constraint plan replay.
- Compare against the stored baseline; fail on a regression beyond the
  pre-registered threshold.

### Release

- Full B1–B4, multi-seed, all workloads.
- Strict coverage must be 100%.
- Kernel numerical checks must pass.
- Task-quality regression must be within the pre-registered budget.
- Every table and figure must be reproducible from a manifest.
- Publish the report outside the repository or in a release artifact. The
  repository stores the harness and configs, not the data.

## 13. Reporting

A release report contains:

1. **Environment and manifest.** Hardware, versions, commits, dataset hashes.
2. **Head-quality Pareto.** Accuracy vs bytes and vs predicted/measured latency,
   with confidence bands.
3. **Kernel table.** Latency, bandwidth, roofline efficiency, and cost-model
   error for each shape and implementation.
4. **Constraint table.** Strict coverage, valid-set buckets, plan sizes, cache
   hit rates, fallback rates.
5. **End-to-end table.** Tau, throughput, TTFT/ITL, validity, task metric, and
   speedup per workload.
6. **Ablations.** What mattered and what did not.
7. **Negative results.** Configurations that lost, and why.
8. **Reproduction commands.** Exact commands for every table and figure.

Suggested plots:

- top-1 / top-5 / OA vs `byte_ratio` (one line per method).
- top-1 / top-5 / OA vs measured latency.
- effective bandwidth vs `m`, with the fitted `t0 + bytes/BW` curve.
- empirical coverage vs nominal coverage, with the diagonal.
- end-to-end speedup vs batch and workload.
- accepted length distribution for the top configurations.

## 14. Threats to validity

| Threat | Mitigation |
|---|---|
| Teacher-forced states differ from on-policy states | report both; calibrate on deployment distribution |
| Target head differs from drafter head | measure both; label the result |
| MACs hide gather/top-k overhead | measure latency and effective bandwidth |
| Kernel effort differs between methods | label reference vs optimized; report both |
| One model/tokenizer is not representative | add models in the robustness stage |
| Calibration leakage inflates coverage | separate calibration and test splits by unit |
| Offline accuracy does not predict online tau | report `sumC6`/`Δτ_prefix` and online tau |
| Small valid sets overstate the benefit | bucket by valid-set size; report fallback rate |
| CUDA graphs change the comparison | report graph capture rate and eager fallback |
| Measurement noise | warmup, repetitions, fixed clocks, CV threshold |
| Engine overhead dominates | report the head's share of decode time |
| Dataset contamination | content hashes and documented splits |
| Cherry-picked configuration | pre-register the metric and report the full sweep |

## 15. Initial targets and open questions

These are project targets, not measured results.

- Strict coverage: 100% on every test; zero valid-token drops.
- Kernel correctness: top-k agreement with the float32 reference on the test set;
  documented numerical tolerance.
- Cost model: median absolute percentage error ≤ 25% on held-out shapes.
- Head quality: a point at ≤ 5% of full-head bytes with no worse than a
  pre-registered OA/top-1 drop on structured workloads.
- End-to-end: speedup > 1.0 at batch ≥ 8 on structured workloads; no regression
  on free-form workloads; target > 1.2 in the best structured configuration.
- Calibrated coverage: within ±2 percentage points of nominal, with a confidence
  interval that contains the nominal value.
- Cache: report cold/warm hit rates; no claim without both.

Open questions:

1. What is the default mode: strict (safe) or calibrated (faster)?
2. Which acceptance model predicts online tau well enough for scheduling?
3. Where is the per-row/union crossover as a function of `B`, `m`, and GPU?
4. Does leaf-contiguous layout help on the target kernel, or is it neutral?
5. How much does calibration-set size and distribution shift move `C*`?
6. Can tree-level union execution reduce traffic enough to justify the planner
   complexity?
7. Which grammar backend gives stable state ids for `PlanCache` reuse?
8. What is the smallest public benchmark set that still supports the headline
   claims without overfitting?

## 16. Definition of a publishable result

A result is publishable only when all of the following hold:

- The harness version and git commit are recorded.
- The model, tokenizer, dataset, and artifact bundle hashes are recorded.
- Calibration and test splits are independent.
- Strict coverage is verified, or the calibrated guarantee is reported with its
  delta and empirical coverage.
- Accuracy, resource, and latency metrics are reported together.
- Confidence intervals are reported for accuracy and coverage.
- The kernel measurement includes warmup, repetitions, and the software/hardware
  environment.
- End-to-end results include validity and quality checks.
- The full knob sweep is available, not only the best point.
- Negative results and fallback rates are included.
- The commands reproduce every table and figure.
