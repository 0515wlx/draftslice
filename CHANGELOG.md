# Changelog

## Unreleased

- Add the experiment-converged production architecture. Routing experiments on
  Qwen3.5-4B/B8 show that a blanket structured DraftSlice profile is a
  regression (0.767× mixed workload, p99 +36%), while identical built-in
  pooling is neutral/slightly positive (1.022×). Production default is now
  built-in MTP; DraftSlice is enabled only by a measured, content-hashed
  per-cell profile table. See `docs/production-architecture.md`.
- Add routing experiment drivers and raw results under `runs/routing/`
  (gitignored) for the per-case matrix, the two-pool gateway, per-workload k
  routing, and the summary table.
- Make TP>1 a supported slicing path in the vLLM adapter: detect each rank's
  contiguous vocabulary shard from the LM-head metadata, slice only that rank's
  weight rows, and scatter logits back to global token-id columns. A rank whose
  valid set has no local token takes a full-head fallback masked to V, so no
  grammar-invalid token can be proposed. The full-head row set is rank-independent
  so the TP logits all-gather stays balanced. Unprovable shard layouts, quantized
  heads, FULL decode CUDA graphs, reasoning-parser shadows, and local-argmax
  reduction all fail closed with recorded reasons. Verify the custom
  `draftslice_mtp` config and end-to-end slicing under vLLM 0.34 TP=2
  (Qwen3.5-0.8B structured JSON, valid output, slicing active on both ranks).
- Cover the remaining production paths with concrete contracts instead of
  silent full-head fallbacks: reasoning-parser shadows now activate only after
  the end-of-reasoning boundary is observed; quant configs no longer reject bf16
  heads and fp8/int8 heads dequantize candidate rows; local-argmax reduction uses
  a sliced `get_draft_top_tokens` override with vLLM's `(value,id)` TP reduction;
  FULL decode CUDA graphs apply a per-replay static packed bitmask with the fused
  mask kernel (step≥1 stays full head because its state is GPU-resident); SGLang
  TP heads slice local rows and return local logits for the engine's own gather.
- Implement the production `DraftContract` data model (`grammar_state_id`, V, A,
  mode, execution, guarantee, serving point) with serialization and deterministic
  ids. The vLLM and SGLang adapters now emit a concrete contract per step and
  expose `fallback_reasons` / `last_contract` through their stats. The contract
  structurally enforces the μ^proj scope (no grammar-faithful claim).
- Add an opt-in fused packed-bitmask kernel for DFlash all-position masking
  (`DRAFTSLICE_VLLM_BLOCK_ALLPOS_FAST=1`) and an opt-in one-hot shortcut for
  single-token valid sets (`DRAFTSLICE_VLLM_DETERMINISTIC=1`). Measured: the
  fused mask removes the high-batch penalty of all-position masking (B=8:
  1.18× over stock DFlash and 1.04× over position-0 slicing); the `|V|=1`
  shortcut found zero qualifying rows on the reference tool schemas and is a
  negative result. Both are default-off.
- Add an opt-in vLLM workload router (`DRAFTSLICE_VLLM_ROUTER=1`) and a
  batch-gated DFlash all-position grammar mask
  (`DRAFTSLICE_VLLM_BLOCK_ALLPOS=1`, `..._MAX_BATCH`, `..._DEPTH`). The router
  sends all-free-form batches through the stock speculator without shadow or
  mapping bookkeeping; all-position masking advances one cloned matcher per
  request and masks every block position under greedy sampling. Both are
  default-off and leave current behavior unchanged. Add `row_mask`,
  `begin_block`, `block_mask`, `block_advance`, and pinned async bitmask uploads
  to the grammar shadow, plus unit tests.
- Refresh README/architecture/competitive-plan/benchmark/design docs to the
  current implementation: the worker shadow, DFlash block-position-0 slicing,
  SGLang MTP hook, screener/artifact/router tooling, union executor, and the
  actual engine fast path. Separate the offline plan/cost/scheduler/union layer
  from the production fast path, and make server-level throughput/TTFT/ITL the
  objective; `tau` is now an explicit tunable that may be traded for latency.
- Add the `DRAFTSLICE_VLLM_FULL_MASK=1` diagnostic control (full head + exact
  grammar mask). It is off by default and is used to separate acceptance gains
  from slicing gains in B4 experiments.
- Add `mask_logits_to_valid()` with tests; no default behavior change.
- Split development into three verified environments because vLLM 0.34 and
  SGLang 0.5.23 pin incompatible torch/xgrammar/flashinfer stacks:
  `wlx_draftslice` (engine-free), `wlx_draftslice_vllm`, and
  `wlx_draftslice_sglang`. Add `requirements-base.txt`, `requirements.txt`,
  `requirements-sglang.txt`, exact `*.lock.txt` files, `scripts/setup_envs.sh`,
  `scripts/verify_env.sh`, and `docs/environment.md`. All three environments
  pass `pip check` and the project test suite.
- Add explicit B4 calibrated modes: `draftslice bench e2e --mode
  constrained_calibrated|freeform_calibrated --screener ... --screener-args ...`
  configures the vLLM screener for that run, and `E2EConfig.screener_env()`
  exports/restores the overlay around engine construction. Strict mode rejects a
  screener so a benchmark label cannot silently mean approximate screening.
- Add M5 screener tooling: `screen.router` trains an MLP group router from
  teacher logits or observed tokens; `screen.artifacts` serializes
  full/static/group/router/low-rank screeners into verified `screener/1`
  bundles; `draftslice screen train-router` builds a router artifact from a
  torch training file; and `compile_screened` caches plans by the exact
  candidate tuple, delta, universe, calibration id, budget, and layout.
- Add the vLLM artifact/free-form screener path:
  `DRAFTSLICE_VLLM_SCREENER_ARTIFACT` loads a verified bundle and records its
  calibration id/delta, and `DRAFTSLICE_VLLM_SCREENER_UNIVERSE=vocab` screens
  free-form rows while still intersecting rows that have a grammar state.
- Add `bench.factorial` and `draftslice bench factorial` for a screener x
  workload matrix with exact top-k coverage, mean candidate count, and
  candidate bytes per cell.
- Add M6 release tooling: `bench.release` hashes result files, writes a
  self-describing manifest, renders Markdown for every implemented schema, and
  supports B5 release comparison; CLI `bench report`, `bench validate`, and
  `bench compare`. Add nightly/release workflows and an engine-free smoke
  pipeline.
- Add the SGLang general plugin: `draftslice.integrations.sglang` installs via
  `sglang.srt.plugins`, slices draft block position 0 from the per-request
  grammar mask through the shared row kernel, and keeps later positions full.
  Unsupported heads and missing/oversized grammars fall back to the original
  head; `DRAFTSLICE_SGLANG=1` enables the hook. Validated end to end with SGLang
  0.5.23 + Qwen3.5-0.8B built-in MTP: step-0 slicing fires, structured JSON is
  valid, and greedy output ids match the hook-off run.
- Enable DFlash block-position-0 slicing in the vLLM adapter. Later block
  positions stay on the full head because their grammar state depends on tokens
  sampled inside the same parallel block. Validated with Qwen3.5-4B +
  Qwen3.5-4B-DFlash: `block_step0_rows` increments, validity 1.0.
- Add an opt-in vLLM screener hook. `DRAFTSLICE_VLLM_SCREENER` names a
  `screen.registry` source and `DRAFTSLICE_VLLM_SCREENER_ARGS` supplies its JSON
  kwargs; the adapter intersects candidates with the exact worker-side grammar
  set (`A := A ∩ V`) before building sliced draft logits. Screener failures and
  empty intersections fall back to the full head, and
  `screener_calls/rows/dropped/empty_rows/errors` are exposed through
  `draftslice_stats()`. This is constrained-calibrated semantics: only candidate
  selection is approximate, and the target verifier remains exact.
- Add a unified candidate-source layer. `screen.registry` exposes full, static/
  VocabTrim, SVD/trained low-rank, group/cluster, MLP router, and oracle
  screeners (CFHead/FlashHead/VocabTrim are lazy reference sources), and
  `PlanCompiler.compile_screened` / `compile_step(screener=...)` compile them
  into the same `HeadPlan`/`StepPlan` used by the strict path. Constrained-
  calibrated plans enforce `A := A ∩ V`; free-form plans leave candidates
  unchanged; strict mode rejects a screener so valid tokens cannot be dropped
  silently. Empty or over-budget rows fall back to the full head. The offline
  baseline policies now re-export the same `Screener` implementations.
- Warm each B4 `(case, batch)` with the case's own structured sampling params,
  so grammar compilation, shadow registration, and the first slice-kernel shape
  are paid outside the measured region. The old plain-params warm-up made the
  first measured repetition look like a slicing regression (0.8B `tool_call`
  now measures 1.22×/1.13× at B=1/8 instead of 0.61–0.80×).
- Add a fused per-row draft-head slice kernel (`run_rows_into`) that gathers
  candidate weight rows, computes the dot product, applies logit post-processing,
  and scatters directly into the full-vocabulary output in one launch.
- Bucket candidate widths to powers of two so all grammar states in a batch share
  one padded `[rows, width]` payload and one kernel launch per bucket. Synthetic
  B=32 host time drops to ~0.19 ms/call (from ~0.5 ms).
- Upload the padded payload from a pinned, event-guarded ring with
  `non_blocking=True`; pageable `.to("cuda")` was synchronizing with the target
  forward already queued on the stream.
- Cache the default slice kernel process-wide and use the async uploader for
  fallback row/candidate indices; keep rows wider than
  `DRAFTSLICE_VLLM_FUSED_MAX_WIDTH` on the grouped gather+GEMM path.
- Detect a FULL decode CUDA graph and disable slicing at speculator
  construction, keeping the graph path full-head exact; expose
  `graph_replay_calls`.
- Add a worker-side xgrammar shadow for exact draft valid sets. The shadow is
  advanced by the tokens the target actually sampled (accepted drafts plus the
  bonus), so draft step 0 no longer lags the scheduler bitmask by one token.
  For `spec_tokens>1`, each draft position gets a forked matcher, replacing the
  fixed row-offset heuristic.
- Cache shadow valid sets by grammar definition plus accepted-token prefix.
  Identical prompts in a batch share one `fill_bitmask` call; a state-key hit is
  ~0.02 ms/row and removes the 232k-token unpack path entirely.
- Add shadow lifecycle hooks through `GPUModelRunner.add_requests` and
  `postprocess_sampled`, and expose shadow counters/timings in the debug stats.
- Validate on Qwen3.5-4B eager: tool_call k=4 tau improves 2.14 -> 3.80 at
  B=1/8 with identical greedy output and validity 1.0; steady-state throughput
  is ~1.0x at B=8 after kernel-shape warm-up.
- Warm every measured `(case, batch)` in the B4 runner instead of only
  `cases[0]`; the old warm-up left the first repetition of each other
  configuration cold and contaminated small-repetition medians.
- Speed up the vLLM grammar hot path: decode bitmask rows with `np.unpackbits`,
  skip unpacking with a popcount gate when the valid set is wider than the slice
  limit, cache decoded sets by packed-row bytes, and use a fast `TokenSet` path
  for already-sorted ids.
- Read `input_batch.idx_mapping_np` (the host mirror) so valid-set construction
  no longer copies a CUDA tensor during decode; the previous `.tolist()` raised
  during CUDA-graph capture and stopped slicing. Keep the host mapping across
  `propose` because `sample_draft` runs after it returns.
- Add the B4 MoE baseline matrix for Qwen3.5-35B-A3B (`spec_tokens` 1 and 4,
  `B=1/8/32/64`, TP=2) and the corresponding report.
- Add union/tree execution: one deduplicated union launch with per-node scatter,
  a `StepPlan` path, and numerical tests against per-node execution.
- Add compiler-level plan caching, a packed-row cache, an opt-in runtime kernel
  autotuner, and an acceptance model with `Scheduler.choose_best`.
- Add an xgrammar constraint adapter (JSON Schema/regex) with bitmask decoding,
  fork support, and B3 trajectory recording; add `draftslice bench constraint`.
- Add the artifact pipeline: deterministic vocabulary partition, split-conformal
  calibration, content-hashed artifact bundles, and `compile_calibrated`.
- Add the B4 vLLM end-to-end runner (`draftslice bench e2e`) with batch sweeps,
  structured/freeform workloads, validity checks, acceptance length, throughput,
  and paired greedy comparison; add `configs/b4_vllm.toml`.
- Make vLLM slicing opt-in (`DRAFTSLICE_VLLM_SLICE=1`) after B4 found that the
  captured bitmask is one grammar state ahead of the draft token. The safe
  default is a full-head pass-through with counters; B4 then matches the
  baseline on validity, greedy equality, and tau.
- Improve the vLLM logits scatter with a flat `index_put_` path and document the
  opt-in graph mode.
- Add tests for union execution, acceptance/scheduler v2, row cache, constraint
  replay, artifact bundles, conformal calibration, autotuning, and the B4 runner.
- Add tuned Triton slice kernels: a row-block GEMV for `B=1` and a
  shape-dispatched tensor-core GEMM for `B>=2`, covering `B=1..64` and
  `m=1..65536` with a shared-memory guard and `OutOfResources` fallback.
- Add an optional CUDA-graph replay cache for the torch baselines and Triton
  kernel, including row/hidden buffer refresh, LRU eviction, capture-failure
  fallback, and capture/replay/fallback counters.
- Make the B2 kernel runner measure eager and graph replay for full, gather, and
  fused paths, so kernel comparisons are fair; fix device selection for
  non-default GPUs and add tiny/medium/large default shapes.
- Add an optional `DRAFTSLICE_VLLM_CUDA_GRAPH=1` path to the vLLM logits helper
  and retry clock-lock verification on an idle GPU.
- Add kernel tuning and graph-cache tests, and extend the Triton shape coverage.
- Update the architecture plan with an implementation snapshot, the kernel
  dispatch/graph contracts, and the remaining kernel work.
- Add the architecture plan (`docs/architecture.md`): execution modes,
  component boundaries, plan/artifact formats, and milestone definitions.
- Add the benchmark plan (`docs/benchmark.md`): tiers, metrics, protocols, and
  release gates.
- Refresh the README roadmap to match the plan.
- Implement the M1 plan core: `TokenSet`, `HeadPlan`, `StepPlan`, guarantees,
  strict compilation with full fallback, plan serialization, a byte-budgeted
  plan cache, a dependency-free reference kernel/executor, and B1 metric helpers.
- Add the reference constraint fixture backend and make `draftslice compile`
  write a real plan.
- Add the first M2 building blocks: a roofline cost model with A100/H100
  profiles, native/grouped vocabulary layouts, a candidate-set cache, and a
  deadline-aware strict/full scheduler with a full-head fallback.
- Add the B1 offline harness, `draftslice bench accuracy`, and the versioned
  result schema.
- Add PyTorch gather/full baselines and Triton fused slice kernels: a scalar-dot
  path for B=1 and a tiled tensor-core GEMM path for B>1.
- Add numerical equivalence tests for the torch and Triton kernels, including
  empty sets, non-contiguous rows, non-power-of-two widths, bf16, and candidate
  projection.
- Add the B2 kernel runner, GPU clock controller, fixed-SM-clock performance test,
  and `draftslice bench kernel`.
- Add the vLLM plugin: `draftslice_mtp` / `draftslice_eagle3` custom speculators,
  the structured-output grammar bridge, draft-step-0 LM-head slicing, and
  full-head fallbacks for unsupported TP, quantization, dtype, and later steps.
- Add the fair baseline suite: shared `CandidatePolicy` API, full/static/SVD/
  trained-low-rank/cluster/router/oracle policies, and the `baseline_matrix/1`
  runner.
- Add reference implementations of the official CFHead, FlashHead, and VocabTrim
  baselines under the shared policy interface, with attribution and citation.
  The CFHead and FlashHead candidate sets were checked against the official
  implementation on a small weight/config and match exactly.
- Verify the public `z-lab/Qwen3.5-4B-DFlash` checkpoint with vLLM's built-in
  `dflash` path; document the future DFlash adapter and its per-block grammar
  state requirement.
- Document the remaining design (union/tree executor, scheduler v2, artifact
  builder, SGLang, release benchmark) in the architecture plan.
- Fix the primary measurement mix to UltraChat-200K (`train_sft`) + ShareGPT
  (`eagle3-mix-v1`) and document that local dataset roots are never committed.
- Add the competitive plan (`docs/competitive-plan.md`): gap analysis,
  advantage thesis, P0/P1/P2 workstreams, gates, and kill criteria.

## 0.0.1

- Initial public skeleton.
- Project scope, interfaces, CLI placeholder, and CI.
- No runtime implementation or benchmark results yet.
