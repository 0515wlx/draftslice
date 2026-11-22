# DraftSlice architecture plan

> Production defaults and the measured profile policy are in
> [`production-architecture.md`](production-architecture.md). This file is the
> component and milestone reference; §0 and §18 track implementation status.

Status: pre-alpha, core implemented. This document describes the target
architecture and the contracts between components. It does not report benchmark
results. Sections 13 and 18 track what is implemented and what remains.

Related documents:

- [design.md](design.md) — public problem statement and scope.
- [benchmark.md](benchmark.md) — benchmark tiers, metrics, and protocols.
- [competitive-plan.md](competitive-plan.md) — gap analysis and the path to a
  defensible advantage.

## 0. Implementation snapshot

Status date: 2026-11-22. There are two execution layers, and they must not be
confused:

- **Offline plan layer**: plan/compiler, cost model, scheduler, acceptance model,
  union executor, candidate/row caches, screeners, artifacts, and release
  tooling. These are implemented and unit-tested, but they are **not yet the
  production engine hot path**.
- **Engine fast path**: the vLLM and SGLang adapters currently consume an exact
  `TokenSet` from the worker grammar shadow and call `build_draft_logits` / the
  fused row kernel directly. This is the path with real-engine runs.

Completed:

- M0 skeleton.
- M1 plan core: types, strict compiler, serialization, plan cache (wired into the
  compiler), reference executor, reference constraint fixture, CLI compile.
- M2 runtime foundation: roofline cost model and A100/H100 profiles, native and
  grouped layouts, candidate-set cache, packed-row cache, deadline-aware
  strict/full scheduler, B1 harness, B2 kernel runner, B3 constraint replay.
- M3 eager kernels: row-block GEMV for `B=1`, tensor-core slice GEMM for
  `B>=2`, shape dispatch tables for `B=1..64` and `m=1..65536`, a CUDA-graph
  replay cache, an opt-in runtime autotuner, graph parity for the full-head and
  gather baselines, and union/tree execution with per-node scatter. The
  tree-specific fused union kernel and engine-owned graph buckets are still open.
- M4 integration: vLLM plugin with `draftslice_mtp`, `draftslice_eagle3`, and
  `draftslice_dflash`; grammar bridge; worker-side xgrammar shadow advanced by
  the verified tokens; per-position forks for `spec_tokens>1`; draft block
  position 0 slicing; full-head fallbacks. TP>1 slices each rank's contiguous
  vocabulary shard, scatters the result back to global token-id columns, then
  max-reduces the partial logits across ranks so every rank has the same
  distribution. The engine samples and advances the grammar shadow on every
  rank, so rank-local logits would let the replicas diverge and deadlock a later
  TP all-gather; `plan_agreement` also unions the full-head row set before the
  collective. Quantized fp8/int8 heads dequantize candidate rows; unsupported
  scale layouts fall back to a full head masked to V. FULL decode graphs
  refresh a static packed bitmask before replay and use `mask`/full for step≥1;
  local-argmax reduction uses a sliced `get_draft_top_tokens` override. Every
  decision emits a `DraftContract` with the fail-closed reason. These paths are code-complete and unit-tested;
  real-engine gates are tracked per profile cell. The 2026-11-22 35B-A3B TP=2
  hang was a rank-local sliced-logits divergence (different `full_rows` per rank
  in the engine logits all-gather); it is fixed and covered by a two-rank
  collective test. MoE TP>1 still needs a same-run positive gate before any cell
  is enabled.
- M5 baseline and calibration core: shared policy interface plus full/static/SVD/
  trained low-rank/group/router/oracle and CFHead/FlashHead/VocabTrim reference
  policies; vocabulary partition, split-conformal calibration, artifact bundle
  hashing, calibrated plan compilation, and an acceptance model with
  `Scheduler.choose_best`.
- Unified candidate-source layer: `screen.registry` maps names to screeners
  (full/static/low-rank/group/router/oracle, with lazy CFHead/FlashHead/VocabTrim),
  and `PlanCompiler.compile_screened` / `compile_step(screener=...)` enforce
  `A := A ∩ V` for constrained-calibrated, keep `A` for free-form, reject a
  screener in strict mode, and fall back to the full head on empty/over-budget
  rows. The vLLM adapter can opt into the same constrained intersection
  (`DRAFTSLICE_VLLM_SCREENER`) on top of its worker-side grammar shadow, with
  per-call candidate/drop counters and a full-head fallback on screener errors
  or empty intersections. Screener artifacts (`screener/1` bundles) and
  `DRAFTSLICE_VLLM_SCREENER_ARTIFACT` load the trained policy without changing
  the candidate algorithm; `DRAFTSLICE_VLLM_SCREENER_UNIVERSE=vocab` enables the
  free-form mode while still intersecting whenever a grammar state exists.
- Screened-plan cache keys include the candidate token tuple, delta, universe,
  calibration id, budget, and layout, so the same grammar state with a different
  hidden state cannot reuse a stale candidate plan. The engine fast path does not
  use those keys yet.
- M5 training/evaluation: `screen.router` trains an MLP router from group
  marginals, `screen.artifacts` serializes full/static/group/router/low-rank
  screeners into verified bundles, and `bench.factorial` runs a screener x
  workload matrix with coverage and candidate-cost cells.
- M6 release tooling: `bench.release` writes hashed release manifests and
  Markdown reports for every implemented result schema, verifies hashes, and
  compares releases for B5 regressions. The SGLang adapter is a real general
  plugin: it slices draft block position 0 from the per-request grammar mask and
  keeps later positions on the full head. The vLLM DFlash adapter does the same
  for block position 0 with the exact worker shadow; later block positions stay
  full. Both paths are unit-tested and have real-engine runs (SGLang 0.5.23
  Qwen3.5-0.8B MTP; vLLM Qwen3.5-4B + 4B-DFlash).
- xgrammar adapter for JSON Schema and regex sessions, including B3 trajectory
  recording.
- B4 vLLM end-to-end runner (`draftslice bench e2e`) with structured, free-form,
  and DFlash cases; a diagnostic full-head+grammar-mask control
  (`DRAFTSLICE_VLLM_FULL_MASK=1`) separates acceptance effects from slicing.
- Numerical and shape tests for the torch and Triton kernels; fixed-clock B2
  performance runner.
- 2026-11-22 research closeout and routing experiments (all opt-in): an
  adapter-level workload router takes the stock path for all-free-form batches;
  a DFlash all-position grammar mask masks every block position under greedy
  sampling. The unfused mask wins at B=1 and loses at B>=8; the fused packed
  mask removes the high-batch penalty, so all-position masking is batch-gated
  (default 1, operator may raise to 8). A static free-form screener is a
  measured negative. The production routing experiment found that a blanket
  structured slice is a regression on 4B/B8 (0.767× mixed workload), so the
  default is built-in and DraftSlice is profile-gated; see
  `docs/production-architecture.md` §1–§2.

Evaluation policy: the objective is **server-level throughput / TTFT / ITL /
cost under quality and validity constraints**, not acceptance length alone.
`tau` is a tunable metric and may be deliberately traded for lower draft or
verification latency when the server operating point improves. Every such result
must report both `tau` and the server metric.

Still open (details in §18):

- wiring the offline plan/cost/scheduler/union/cache layer into the engine hot
  path under the measured profile table (default full);
- the DFlash per-position **slice** (all-position *masking* is implemented behind
  an opt-in, batch-gated flag; a per-position slice needs a sequential block
  sampler);
- measured profile cells for 4B B=1 tool, 35B-A3B tool, and DFlash-vs-MTP;
  a blanket structured slice is a measured negative on 4B/B8;
- SGLang EAGLE/DFlash hooks and an SGLang B4 runner (MTP step-0 is validated);
- calibrated/free-form end-to-end results with a positive server-level Pareto
  point and measured coverage (the first static point is a measured negative);
- multi-seed release runs on the reference hardware;
- cost-model calibration (MAPE) and scheduler validation on real traces.


## 1. What we are building

DraftSlice is a **draft-head planner and executor** for speculative decoding. It
answers one question before the draft head runs:

> Given the current draft hidden state, the grammar state, the batch shape, and a
> latency budget, which vocabulary rows must be scored, how should they be laid
> out, and which kernel should execute them?

The target verifier is never approximated. DraftSlice only changes the draft
path.

There are two product paths, and they must not be confused:

1. **Strict constrained slicing.** A grammar/schema defines an exact valid-token
   set `V`. The plan contains every token in `V`. This is the original DraftSlice
   claim: no valid token is dropped.
2. **Calibrated candidate screening.** A learned or static screener proposes a
   candidate set `A`, and only `A` is scored exactly. This is the CHESS-style path
   from the local research workspace. It is approximate with respect to the full
   head, but it can carry a finite-sample coverage guarantee. It may be applied
   to the full vocabulary or inside a constrained valid set.

The two paths are **designed** to share the same plan format, cache, cost model,
executor, and scheduler. This is the target architecture. In the current engine
fast path, the adapters construct exact `TokenSet`s and call the row kernel
directly; the offline plan/cost/scheduler/union/cache layer is not yet the
production path. The difference between the paths is the **guarantee**, not the
plumbing, once that wiring is complete.

### 1.1 Assumptions and non-goals

In scope for the first architecture:

- one draft head per request, on a single node, with a single GPU as the primary
  target;
- exact target verification, unchanged;
- grammar/schema compilation delegated to an existing backend;
- offline artifact building and online planning/execution;
- dense and batched draft execution, including tree nodes.

Explicitly out of scope for the first implementation:

- training a drafter or a target model at runtime;
- approximate or lossy target verification;
- tensor/pipeline/expert parallelism across GPUs (the interfaces should not
  prevent it, but the first cost model and kernels are single-GPU);
- multi-head or multi-model routing inside one step;
- a new grammar engine;
- a serving system of its own. DraftSlice is a library plus adapters.

## 2. Design principles

1. **Exactness is a mode, not a marketing word.** Every plan declares its
   candidate universe and guarantee. Strict mode must cover the valid set
   exactly. Calibrated mode must report its delta and calibration source.
2. **Plans are data.** A plan is a small, serializable, inspectable object. It can
   be cached, logged, replayed, and compared across kernels and engines.
3. **Logical candidates and executed rows are separate.** A strict plan may have
   a logical valid set of 40k tokens but execute the full head because slicing is
   not profitable. The plan records both.
4. **The cost model is first-class.** Scheduling decisions use a calibrated
   `(bytes, ALU, MMA, launch)` model, and the benchmark suite measures its error.
5. **Fallback is always available.** Unknown grammar state, oversized valid set,
   unsupported kernel, OOM, or CUDA-graph-capture failure must fall back to a
   correct path. Strict mode never fails open.
6. **The core is engine-agnostic.** vLLM and SGLang are adapters. Importing the
   core must not import either engine.
7. **Every layer is benchmarkable in isolation.** Head quality, plan quality,
   kernel latency, and end-to-end acceptance are separate measurements.

## 3. System context

```text
OFFLINE
  model weights + tokenizer
        |
        +--> PartitionBuilder  ---> vocabulary groups / leaf layout
        +--> ScreenerTrainer   ---> router / static / low-rank weights
        +--> Calibrator        ---> conformal quantiles, acceptance model
        |
        v
  ArtifactBundle (versioned, content-hashed)

ONLINE
  prompt + schema/grammar
        |
        v
  ConstraintSession ---- state transitions ----> valid token set V
        |                                              |
        |                                              v
        +---- draft hidden h ----> Screener -----> group scores
        |                                              |
        +----------------------> PlanCompiler <-------+
                                       |
                                       v
                                   HeadPlan / StepPlan
                                       |
                                       v
                                   Scheduler
                              (mode, C/mass, tree, kernel)
                                       |
                                       v
                                   Executor
                         (union rows, gather, exact rescore)
                                       |
                                       v
                                  draft tree
                                       |
                                       v
                            target verifier (unchanged, exact)
```

## 4. Execution modes and guarantees

A plan always names a **universe** and a **candidate set**:

- `universe = valid`: the grammar/schema-valid token set `V`.
- `universe = vocab`: the full vocabulary.
- `candidates = A`: tokens that will be scored exactly.
- `executed = E`: rows the kernel actually reads. `E` may be larger than `A`
  (for example, `E = full vocabulary` when the kernel falls back).

| Mode | Universe | Candidate set | Guarantee | May drop valid tokens? |
|---|---|---|---|---|
| `full` | vocab | all tokens | exact full head | n/a |
| `strict` | valid | `A = V` | every valid token is present | no |
| `constrained_calibrated` | valid | `A ⊆ V` | calibrated coverage over `V` | yes, with delta |
| `freeform_calibrated` | vocab | `A ⊆ vocab` | calibrated coverage over top-k | yes, with delta |

**Hard invariants**

- `strict`: `V ⊆ A`. If the planner cannot afford `A = V`, it must widen the
  execution to `full` or reject the latency target. It must not shrink `A`.
- `strict`: output is identical to full-head constrained decoding under the same
  sampler and seed (up to numerical tolerance and documented tie-breaking).
- `constrained_calibrated`: `A ⊆ V`. Invalid tokens are never proposed, even if
  the screener ranks them highly.
- `freeform_calibrated`: no validity constraint. The target verifier still
  guarantees final correctness; the guarantee is only about candidate coverage.
- All modes: once `A` is chosen, scoring inside `A` is exact. Any ordering error
  can only come from a token missing from `A`, not from approximate scores.

This last point is the separation principle: screen, then rescore exactly.

### Why strict and calibrated are not mixed silently

It is tempting to "use the grammar to shrink the vocabulary, then use a router to
shrink it further." That silently drops valid tokens and invalidates the strict
claim. The architecture requires the caller to choose a mode:

- If the valid set is small enough, use `strict`.
- If it is too large for the budget, either execute the full head or explicitly
  switch to `constrained_calibrated` and report the coverage target.
- The scheduler may recommend the switch, but the request configuration decides
  whether approximation is allowed.

## 5. Components

### 5.1 Artifact builder (offline)

Produces an `ArtifactBundle` from model weights and tokenizer metadata.

- `PartitionBuilder`: vocabulary partition. Initially leaf-contiguous k-means
  (`K` leaves) and token-frequency buckets. Later: supervised partitions.
- `ScreenerTrainer`: trains the MLP router, static priors, or low-rank screen.
  Training targets are group marginals, not hard top-1 labels.
- `LayoutBuilder`: builds the token-to-row permutation for leaf-contiguous
  storage and the reverse mapping.
- `Calibrator`: split-conformal quantiles for `C*`, acceptance-model parameters,
  and cost-model calibration residuals.
- `BundleWriter`: writes a manifest plus binary arrays, content-hashed.

Artifacts are immutable. A new training run produces a new `bundle_id`.

### 5.2 Constraint layer

DraftSlice does not reimplement grammars. It adapts existing engines:

- JSON Schema / tool calling: xgrammar or outlines.
- Regex / CFG: the engine's grammar backend.
- Tokenizer-specific vocabularies: validated against the model tokenizer.

Interface:

```python
class ConstraintSession(Protocol):
    def state_id(self) -> int: ...
    def valid_ids(self) -> TokenSet: ...
    def advance(self, token_id: int) -> None: ...
    def fork(self) -> "ConstraintSession": ...
    def is_final(self) -> bool: ...
```

`TokenSet` is a sorted, deduplicated set of native token ids with an optional
range representation. The constraint layer is responsible for correctness of
`V`; DraftSlice is responsible for not losing it.

### 5.3 Plan layer

The plan layer turns `(V, h, policy)` into a `HeadPlan` and a `StepPlan`.

```python
class GuaranteeKind(str, Enum):
    FULL = "full"
    EXACT_VALID_SET = "exact_valid_set"
    CALIBRATED = "calibrated"

@dataclass(frozen=True, slots=True)
class TokenSet:
    ids: tuple[int, ...]          # sorted, unique, native ids

@dataclass(frozen=True, slots=True)
class Guarantee:
    kind: GuaranteeKind
    universe: Literal["vocab", "valid"]
    delta: float | None = None
    calibration_id: str | None = None

@dataclass(frozen=True, slots=True)
class Execution:
    kind: Literal["full", "slice_gemv", "slice_gemm"]
    rows: TokenSet | None        # None means the full head
    layout: Literal["native", "leaf_contiguous", "packed"]

@dataclass(frozen=True, slots=True)
class PlanCost:
    static_bytes: int
    dynamic_bytes: int
    macs: int
    alu_ops: int
    launch_count: int
    predicted_us: dict[str, float]   # hardware profile -> estimate

@dataclass(frozen=True, slots=True)
class HeadPlan:
    plan_id: str
    state_id: int | None
    candidates: TokenSet
    execution: Execution
    guarantee: Guarantee
    cost: PlanCost
    source: str                      # constraint | router | static | fallback

@dataclass(frozen=True, slots=True)
class NodePlan:
    node_id: int
    plan: HeadPlan

@dataclass(frozen=True, slots=True)
class StepPlan:
    nodes: tuple[NodePlan, ...]
    union_rows: TokenSet           # deduplicated execution rows for the step
    kernel: Literal["auto", "slice_gemv", "slice_gemm", "full"]
    cost: PlanCost
```

The compiler entry points:

```python
class PlanCompiler(Protocol):
    def compile_strict(self, state, *, budget, policy) -> HeadPlan: ...
    def compile_calibrated(
        self, candidates: TokenSet, *, universe: TokenSet | None, delta: float,
        budget, policy, source, calibration_id,
    ) -> HeadPlan: ...
    def compile_screened(
        self, states, candidates: Sequence[TokenSet], *, delta: float,
        universe: Literal["valid", "vocab"], budget, policy, source,
    ) -> StepPlan: ...
    def compile_step(
        self, states, *, mode=None, screener=None, hidden_states=None,
        delta=None, universe: Literal["valid", "vocab"] = "valid",
        budget, policy,
    ) -> StepPlan: ...
```

A `Screener` is one protocol, `candidates(hidden_states) -> list[TokenSet]`.
`screen.registry` is the single name-to-implementation map shared by the
compiler and the benchmark. In constrained-calibrated mode the compiler enforces
`A := A ∩ V` and records `V` as the coverage universe; in free-form mode it
keeps `A`; in strict mode it **rejects** a screener rather than silently dropping
valid tokens. Empty or over-budget rows fall back to the full head unless the
budget forbids fallback.

`PlanPolicy` chooses the allowed modes, preference order, and fallback rules.
`Budget` carries `target_bytes`, `deadline_ms`, `target_delta`, `max_candidates`,
and `min_expected_tau`.

For a draft tree, states differ per node. `compile_step` must:

1. Group identical constraint states.
2. Compile one plan per distinct state (cacheable).
3. Compute the union of execution rows across nodes.
4. Execute the union once and scatter outputs to nodes.

This is where tree drafting gets its leverage: many nodes share a small set of
grammar states, and the union is much smaller than the sum of per-node sets.

### 5.4 Cache layer

There are four distinct caches. They must not be conflated:

| Cache | Key | Value | Scope |
|---|---|---|---|
| `ArtifactCache` | `bundle_id` | loaded tensors, manifest | process |
| `PlanCache` | `(bundle_id, tokenizer_hash, constraint_id, state_id, mode)` | `HeadPlan` | cross-request |
| `CandidateSetCache` | `(bundle_id, sorted group ids)` | `TokenSet` / packed index | cross-request |
| `PackedRowCache` | `(bundle_id, layout, rowset_hash)` | packed matrix / gather index | optional, memory-budgeted |

- `PlanCache` is the original "state -> plan" cache. It only applies to
  constraint-keyed plans. A router plan depends on `h` and is not state-keyed.
- `CandidateSetCache` is the equivalent for screened plans: many queries select
  the same leaf set, and all tree nodes in a step may share it.
- `PackedRowCache` is an optimization for hot row sets. It must be evictable and
  disabled by default until measured.
- Every cache exposes hit/miss/eviction/byte counters.
- Eviction is LRU with byte budgets. The current request is pinned.

### 5.5 Layout layer

```python
class VocabularyLayout(Protocol):
    token_to_row: Sequence[int]
    row_to_token: Sequence[int]
    def pack(self, ids: TokenSet) -> PackedRows: ...
    def segments(self, group_ids: Sequence[int]) -> list[slice]: ...
```

- `native`: row `i` is token `i`. Simplest; random gather.
- `leaf_contiguous`: rows of the same leaf are adjacent. Candidate sets become a
  small number of contiguous segments, which can improve gather efficiency on
  some kernels. Research shows this is not a free win; it must be measured.
- `packed`: a precomputed submatrix for a hot candidate set.

Layout must be a bijection. Every plan that uses a non-native layout records the
layout id in its `Execution`.

### 5.6 Kernels and executor

```python
class HeadKernel(Protocol):
    def supports(self, exec: Execution, batch: int, hints: KernelHints) -> bool: ...
    def run(self, hidden: Tensor, exec: Execution, layout: VocabularyLayout,
            out: Tensor | None = None) -> Tensor: ...
    def cost(self, exec: Execution, batch: int, hints: KernelHints) -> PlanCost: ...
```

Backends:

- `ReferenceKernel`: PyTorch-free gather + dot, CPU and CUDA. Always available.
- `SliceGEMVKernel`: batch 1; a row-block fused gather + dot. Several weight
  rows per program keep the load coalesced and amortise the launch.
- `SliceGEMMKernel`: batched; union rows across the batch, one tiled tensor-core
  GEMM, scatter. It is used for `B>=2` (including float32 through a legacy
  scalar-dot path).
- `FullKernel`: dense full head, used for fallback and baselines.
- `TreeKernel`: union rows across tree nodes, scatter to node outputs (planned,
  see §18.1).

The eager kernels are intentionally paired with **equal-effort baselines**:

| Path | Eager | CUDA graph | Purpose |
|---|---|---|---|
| full head | `TorchFullKernel` | `TorchFullKernel(use_cuda_graph=True)` | exact oracle and latency ceiling |
| gather + GEMM | `TorchGatherKernel` | same class with graphs | naive baseline |
| fused slice | `TritonSliceKernel` | same class with graphs | DraftSlice path |

A comparison that gives graph replay only to the fused path is invalid. The B2
runner therefore measures both sides in both modes.

#### 5.6.1 Shape dispatch

`kernels/tuning.py` maps `(rows, batch, width, dtype)` to a launch configuration.
The current regimes are:

| Rows | Batch | Kernel | Typical tile |
|---|---|---|---|
| `1..4096` | `1` | row-block GEMV | `BLOCK_M=8`, `BLOCK_D=512` |
| `8193..16384` | `1` | row-block GEMV | `BLOCK_M=16`, `BLOCK_D=512` |
| `>16384` | `1` | row-block GEMV | `BLOCK_M=8`, `BLOCK_D=256` |
| `1..1024` | `2..16` | tensor-core GEMM | `32x128`, 4 stages |
| `1..1024` | `32..64` | tensor-core GEMM | `16x64`, 3 stages |
| `1025..8192` | `2..16` | tensor-core GEMM | `64x256` then `64x128` |
| `1025..8192` | `32` | tensor-core GEMM | `32x128`, 3 stages |
| `1025..8192` | `64` | tensor-core GEMM | `64x64`, 4 stages |
| `>8192` | `32..64` | tensor-core GEMM | `32/64 x 64/128` |

The table is a conservative starting point, not a claim about another GPU. Every
configuration is checked against a shared-memory budget, and a launch that still
raises `OutOfResources` is retried with a smaller tile. The next step is a
runtime autotuner keyed by `(shape, dtype, device)` (§18.8).

#### 5.6.2 CUDA graphs and shape buckets

The candidate row set changes every grammar step, so graph capture is keyed by
the **shape bucket**, not by the exact token ids:

```text
key = (kernel, weight_ptr, batch, hidden, dtype, device, row_count)
```

The first call captures a graph with static hidden/row buffers. Later calls
replay it, copying a new hidden state or row index into the static buffers when
the source tensor identity or its in-place version changes. Consequences:

- a cached plan that reuses its row tensor replays with no copy;
- a new row set of the same size pays one device copy, which is still cheaper
  than eager launch overhead at small `m`;
- graph capture failure is non-fatal and falls back to eager execution;
- the returned tensor is owned by the graph entry and must be consumed before
  the next replay with the same key;
- entries are LRU-bounded.

Engine adapters should eventually capture per decode shape bucket and keep the
hidden workspace stable; until then, the copy path is the safe default.

Executor responsibilities:

1. Validate that `execution.rows` covers `candidates` for strict plans.
2. Select a kernel by hint, backend support, batch, layout, and hardware.
3. Deduplicate rows across batch/tree.
4. Run in `bf16`/`fp16` with `fp32` accumulation where supported.
5. Return logits for the logical candidate set, plus execution stats.
6. Fall back to `ReferenceKernel` or `FullKernel` on any failure.

Numerical contract: results must match a float32 reference within a documented
tolerance, and top-k ordering must match under a deterministic tie-break.

CUDA graphs are a first-class concern: candidate sets are dynamic. The design
uses shape buckets (for example `m` in powers of two) with a fixed maximum, and
falls back to eager execution outside the buckets. Graph capture success rate is
a benchmark metric.

### 5.7 Scheduler

The scheduler chooses, per step:

- execution mode (`strict`, `constrained_calibrated`, `freeform_calibrated`,
  `full`);
- budget (`C`, mass threshold, `lambda`, `m`);
- draft depth and tree width;
- verify width;
- kernel (`per_row` vs `union`);
- fallback if the deadline cannot be met.

```python
@dataclass(frozen=True, slots=True)
class StepContext:
    batch: int
    k: int
    tree_depth: int
    tree_width: int
    deadline_ms: float | None
    min_expected_tau: float
    mode_policy: PlanPolicy

@dataclass(frozen=True, slots=True)
class Schedule:
    step_plan: StepPlan
    tree_depth: int
    tree_width: int
    verify_width: int
    expected_tau: float
    predicted_us: float
    explanation: dict[str, object]
```

The acceptance model maps plan statistics to expected accepted length:

```python
class AcceptanceModel(Protocol):
    def predict(self, plan: HeadPlan, tree: TreeShape) -> AcceptanceEstimate: ...
```

Initial implementation: prefix-coverage proxy plus measured next-token agreement
(`sumC6`, `Δτ_prefix`) from the evaluation harness. Later: an online estimator
updated from accepted tokens.

Objective: maximize **server-level** throughput (or minimize TTFT/ITL) subject to
validity, quality, and a tail-latency budget. `tau` is a decision variable, not
an invariant: if reducing draft/verify work lowers server latency or raises
tokens/s, the scheduler may deliberately choose a lower-`tau` point. The plan and
result must report both `tau` and the server metric, plus the reason for the
trade. `min_expected_tau` is an optional constraint for deployments that need a
minimum acceptance floor.

Strict-mode constraints are checked before the objective. If the strict plan is
over budget, the scheduler must first reduce tree width/depth, then fall back to
`full`. It must never reduce `candidates` below `V`.

### 5.8 Cost model

One module, one interface, calibrated per hardware profile:

```python
class CostModel(Protocol):
    def predict(self, plan: HeadPlan, batch: int, hints: KernelHints) -> PlanCost: ...
```

The model is the executable form of the local framework notes:

```text
t = launch + max(bytes / BW(level), alu / P_alu, macs / P_mma)
```

- `bytes` is split into static (L2-resident if it fits) and dynamic (gather or
  streaming).
- Gather bandwidth is lower than peak HBM and depends on `m`, layout, and batch.
- Batch reuse is modeled by union rows, not by multiplying per-request traffic.
- Every hardware profile records how it was fitted and the residual error.

The planner uses predicted latency; the benchmark suite reports predicted vs
measured error. A model that is not calibrated for the target GPU must be marked
as such and must not be used to claim speedups.

For server planning, the model must also expose the **draft-head share of the
measured speculative step**. A head-only speedup is not a server speedup: the
same 10× head improvement can be 1% or 20% end-to-end depending on the model,
batch, k, engine overlap, and target-verification width. Every performance
report must state the measured share and the server metric, not only head
latency.

The share is not a constant across `k`: reducing draft depth removes whole head
evaluations, so a lower-`tau` configuration can remove more head work than the
per-call head time suggests. Conversely, a higher `k` can raise `tau` while
lowering server throughput. The scheduler must evaluate the joint `(k, source,
budget)` point on the server metric.

### 5.9 Integrations

```python
class DraftPathAdapter(Protocol):
    def begin_request(self, constraint, tokenizer) -> ConstraintSession: ...
    def plan_step(self, hidden, states, policy, budget) -> StepPlan: ...
    def execute(self, hidden, step_plan) -> Tensor: ...
    def on_accept(self, node_id: int, token_id: int) -> None: ...
```

- `vllm.py`: attach to the speculative-decoding proposer and draft LM head.
- `sglang.py`: attach to the draft path.
- The adapter owns engine-specific batching, CUDA graph buckets, and grammar
  state extraction.
- The core owns plan compilation, caching, execution, and scheduling.
- Engine versions are pinned in optional extras; integration tests are skipped
  when the engine is not installed.

The current engine fast path does **not** yet call this full protocol: it builds
exact `TokenSet`s from the worker shadow and invokes the row kernel directly. See
§0 for the boundary.

### 5.10 Observability

Every request can emit a `DraftStats` record:

```python
@dataclass
class DraftStats:
    mode: str
    plan_compile_us: float
    plan_cache_hit: bool
    candidate_set_cache_hit: bool
    candidates: int
    executed_rows: int
    fallback: str | None
    kernel: str
    predicted_us: float
    measured_us: float
    coverage_delta: float | None
    expected_tau: float
    accepted_tokens: int
```

Stats are useful for debugging, benchmarking, and online calibration. They are
not a public API until the format stabilizes.

## 6. Package layout

```text
src/draftslice/
  __init__.py
  cli.py
  errors.py

  plan/
    types.py            # TokenSet, HeadPlan, StepPlan, Guarantee, PlanCost
    compiler.py         # PlanCompiler, strict/calibrated/step compilation
    compose.py          # union, intersection, dedupe, coverage checks
    serialize.py        # JSON + binary plan encoding

  constraints/
    base.py             # ConstraintSession protocol
    xgrammar.py         # optional adapter
    outlines.py         # optional adapter

  screen/
    base.py             # Screener protocol
    router.py           # MLP group router
    static.py           # frequency / static rows
    lowrank.py          # SVD or trained low-rank screen

  layout/
    permutation.py      # native and leaf-contiguous layouts
    packing.py          # pack/unpack, segment metadata

  cache/
    plan_cache.py       # state -> HeadPlan
    candidate_cache.py  # group set -> TokenSet
    row_cache.py        # optional packed rows
    stats.py

  kernels/
    base.py             # HeadKernel protocol
    reference.py        # torch-free reference
    torch.py            # torch full/gather baselines
    triton.py           # fused slice kernels
    tuning.py           # shape -> launch configuration
    graph.py            # CUDA-graph capture/replay cache
    registry.py         # selection and fallback

  runtime/
    executor.py         # dispatch, dedupe, scatter, validation
    scheduler.py        # mode/budget/tree/kernel decisions
    acceptance.py       # acceptance model
    stats.py

  cost/
    model.py            # roofline + cache staircase
    profiles.py         # A100/H100/... calibrated profiles

  bench/
    schema.py           # result and config schemas
    datasets.py         # dataset adapters and hashes
    metrics.py          # accuracy, cost, latency, coverage
    harness.py          # runner protocol, oracle cache, result store
    runners/
      conformance.py
      head_accuracy.py
      kernel.py
      constraint.py
      e2e.py
    report.py           # tables, Pareto plots, compare
    cli.py              # `draftslice bench ...`

  integrations/
    vllm.py
    sglang.py
```

`bench/` is part of the package because the benchmark suite is a deliverable, not
an afterthought. Heavy dataset and engine dependencies remain optional extras.

Repository-level layout for the non-package parts:

```text
benchmarks/
  README.md          # logical dataset names; local roots come from the environment
  configs/           # B0-B4 TOML configs
  fixtures/          # small schemas, grammars, and synthetic cases
  manifests/         # dataset/artifact manifests (hashes, not data)

results/             # local run outputs; gitignored
artifacts/           # local artifact bundles; gitignored
```

The repository stores configs, fixtures, and manifests. It does not store
datasets, model weights, artifact tensors, or run outputs.

## 7. Data formats

### 7.1 Artifact bundle manifest

```json
{
  "schema_version": "artifact/1",
  "bundle_id": "sha256:...",
  "model": {
    "name": "Qwen3.5-4B",
    "revision": "...",
    "hidden_size": 2560,
    "vocab_size": 248320,
    "dtype": "bf16",
    "tokenizer_hash": "sha256:..."
  },
  "partition": {
    "kind": "leaf_kmeans",
    "groups": 1024,
    "token_to_group": "arrays/token_to_group.npy",
    "group_sizes": "arrays/group_sizes.npy",
    "layout": "arrays/layout.npy"
  },
  "screener": {
    "kind": "mlp_router",
    "weights": "arrays/router.safetensors",
    "hidden": 512,
    "calibration": "arrays/conformal.json"
  },
  "cost": {
    "profile": "a100",
    "calibration": "arrays/cost_fit.json"
  },
  "created_by": "draftslice-build",
  "git_commit": "..."
}
```

Binary arrays are stored next to the manifest. Every file has a hash. A mismatch
is a hard error; the bundle is rebuilt rather than silently used.

### 7.2 Plan serialization

```json
{
  "schema_version": "plan/1",
  "plan_id": "sha256:...",
  "state_id": 17,
  "candidates": {"encoding": "u32", "count": 6115, "file": "plans/17.u32"},
  "execution": {
    "kind": "slice_gemm",
    "rows": {"encoding": "u32", "count": 6115, "file": "plans/17.u32"},
    "layout": "leaf_contiguous"
  },
  "guarantee": {
    "kind": "exact_valid_set",
    "universe": "valid",
    "delta": null,
    "calibration_id": null
  },
  "cost": {
    "static_bytes": 5656576,
    "dynamic_bytes": 31308800,
    "macs": 15654400,
    "alu_ops": 12654,
    "launch_count": 1,
    "predicted_us": {"a100": 28.8}
  },
  "source": "constraint"
}
```

Small plans can embed ids inline. Large plans use a binary sidecar. The
`plan_id` is a content hash of the normalized plan, excluding timestamps.

### 7.3 Versioning and compatibility

- `artifact/1`, `plan/1`, `bench/1` are explicit schema versions.
- A runtime refuses to load an artifact with an unknown major version.
- The model/tokenizer hash is part of every cache key.
- Plans are portable across kernels only if layout and row ids are identical.
- A cost estimate is hardware-specific and is not part of `plan_id`.

## 8. Runtime lifecycle

```text
request start
  adapter -> ConstraintSession
  adapter -> ArtifactCache.get(bundle_id)

step:
  hidden h <- draft backbone
  states   <- constraint sessions for draft tree nodes
  options  <- PlanCompiler.compile_step(states, h, policy, budget)
  schedule <- Scheduler.choose(options, StepContext)
  logits   <- Executor.run(h, schedule.step_plan)
  tree     <- draft tree from logits
  target   <- verifier(tree)              # exact, unchanged
  accepted <- verifier result
  update constraint states and stats
```

The plan compiler may be called before the backbone finishes if the constraint
state is known; this overlaps plan compilation with the backbone. The executor
must not start before the hidden state is ready.

For a tree, the plan compiler groups states, compiles distinct plans, and
computes a union. The executor runs the union once. This is the main runtime
optimization beyond per-node slicing.

Plan compilation is on the critical path for dynamic screens. Three mitigations
are required:

- cache strict plans by grammar state across requests;
- overlap screen scoring with the draft backbone where the engine allows it;
- precompile plans for known schemas/tool definitions at load time, and keep a
  fast path for repeated states.

A Python-only plan compiler that takes longer than the head kernel is a failure
of the architecture, not a tuning problem. Compile time is a first-class
benchmark metric.

## 9. Scheduling policy

The default policy is conservative and explainable:

1. If no screener is configured, and a constraint exists, use `strict`.
2. If a strict plan fits the byte and latency budgets, use it.
3. If a strict plan does not fit, reduce tree width/depth first. If it still does
   not fit, use `full` and mark the deadline as missed.
4. If the caller allows calibrated mode and a screener exists, consider
   `constrained_calibrated` or `freeform_calibrated`.
5. Among feasible calibrated options, choose the one with the highest predicted
   `expected_tau / predicted_us` while meeting `target_delta`.
6. Prefer `per_row` at small batch and `union` at larger batch; the crossover is
   measured, not hardcoded.
7. Never choose a mode that violates the request's allowed guarantee set.

The scheduler returns an explanation object so benchmark reports can show why a
configuration was chosen.

## 10. Memory and cache budget

The artifact bundle is long-lived and shared. Plan and candidate caches are
byte-budgeted. Suggested defaults for a 4B model on an 80GB GPU:

- artifact bundle: resident, budgeted separately;
- plan cache: 64 MiB;
- candidate-set cache: 128 MiB;
- packed-row cache: disabled by default;
- per-request working set: bounded by `max_candidates`.

These are starting points, not tuned values. The benchmark suite measures hit
rate, eviction rate, and latency with cold and warm caches.

## 11. Failure handling

| Failure | Behavior |
|---|---|
| Unknown grammar state | `full` fallback, stats record |
| Grammar backend error | reject request or `full` fallback, never guess |
| Valid set too large | `full` fallback; strict plan stays correct |
| Candidate set too large | `full` or larger budget, never truncate strict |
| Unsupported kernel | `ReferenceKernel` |
| OOM | shrink tree/budget, then `full`; if still OOM, fail request |
| Triton `OutOfResources` | retry a smaller tile; if still failing, `ReferenceKernel` |
| CUDA graph bucket miss | eager execution, stats record |
| CUDA graph capture failure | eager execution, record the reason |
| Artifact hash mismatch | refuse to load, rebuild |
| Calibration drift | warn and recalibrate; do not silently claim coverage |
| Non-finite logits | fail the step; fall back to full head for diagnosis |

Strict mode is fail-closed with respect to coverage: a fallback may be slower,
but it must not drop a valid token.

## 12. Dependencies and packaging

Keep the core light:

```toml
dependencies = []

[project.optional-dependencies]
runtime     = ["numpy>=1.24", "torch>=2.2"]
kernels     = ["triton>=3.0"]
constraints = ["xgrammar>=0.1"]
bench       = ["transformers>=4.40", "datasets>=2.18", "scipy>=1.11"]
vllm        = ["vllm>=0.6"]
sglang      = ["sglang>=0.3"]
dev         = ["pytest>=8", "ruff>=0.6", "mypy>=1.8"]
```

The exact versions are pinned when the integration lands. The base install must
remain importable without torch, Triton, or an engine.

## 13. Milestones and definition of done

### M0 — public skeleton (done)

- Interfaces, CLI placeholder, CI, license.
- DoD: `pytest` and `ruff` pass on a clean checkout.

### M1 — plan core and conformance suite

- `TokenSet`, `HeadPlan`, `StepPlan`, `Guarantee`, `PlanCost`.
- Plan serialization v1.
- Constraint adapter protocol and one reference backend.
- Strict compiler and reference executor.
- `PlanCache` with metrics.
- CLI: `draftslice compile` writes a real plan.
- B0 conformance tests and B1 metric definitions.

DoD:

- Strict plans cover the valid set on every test grammar (100%).
- Plan round-trip preserves ids, layout, and guarantee.
- Reference executor matches full-head logits within tolerance.
- Cache hit/miss behavior is tested.
- `draftslice compile` works on the reference constraint fixture
  (`valid_ids` / `states`). JSON Schema and regex adapters are implemented via
  xgrammar; CFG remains unsupported until a backend is added.

### M2 — runtime, cost model, scheduler

- Layout layer, reference kernels, `PlanCost`, cost profiles.
- Scheduler v1 with strict/full modes (calibrated mode lands with the screener).
- Candidate-set cache and packed-row cache behind flags.
- Benchmark B2 (kernel microbench) and B3 (constraint plan quality).

Implemented so far: the roofline cost model with A100/H100 profiles, native and
grouped vocabulary layouts, candidate-set and packed-row caches, a deadline-aware
strict/full scheduler, the B2 kernel runner with fixed-clock support, and the B3
constraint replay runner with compiler-level plan-cache integration.

DoD:

- Kernel correctness on all supported shapes.
- Scheduler never violates strict coverage.
- Cost model predicts held-out microbench configs within a documented error
  target (initial target: median absolute percentage error ≤ 25%).
- Cold/warm cache behavior is reported.

### M3 — optimized kernels

- Triton `slice_gemv`, `slice_gemm`, union and tree executors.
- Shape buckets and CUDA graph support.
- Kernel autotuning and fallback.

Implemented so far: row-block GEMV and tensor-core slice GEMM with static shape
tables, shared-memory guard and `OutOfResources` fallback, CUDA-graph replay,
opt-in runtime autotuning, graph parity for baseline kernels, and a union/tree
executor with per-node scatter. Engine-side graph bucket ownership and a
tree-specific fused kernel remain.

DoD:

- Numerical equivalence with the reference kernel. **Passing for eager, graph,
  and union paths.**
- Measured speedup over the reference and full-head baselines at B=1/8/32/64.
  **Eager fused is competitive with gather from roughly `m>=1k` and wins clearly
  at large `m`. Graph replay removes the launch floor for both sides; fused wins
  from roughly `m>=4k`, while tiny-M graph replay is at parity with gather.**
- Graph capture success and bucket-miss behavior reported. **B2 now records
  captures, replays, fallbacks, and a disabled reason.**
- No regression in B1 accuracy.

### M4 — vLLM integration

- Adapter, grammar state extraction, draft-path hook.
- End-to-end constrained decoding on structured workloads.
- Benchmark B4.

DoD:

- Strict mode output matches full-head constrained decoding under greedy.
- Output validity is 100% on structured workloads.
- Server-level throughput, TTFT, ITL, and cost are reported per workload and
  batch, together with `tau` and the head-share decomposition. A result may trade
  `tau` for server latency; it must state that it did.
- No core import of vLLM in unit tests.

### M5 — screener and calibrated mode

- Partition builder, router trainer, static/low-rank screens.
- Split-conformal calibration and `constrained_calibrated` /
  `freeform_calibrated` modes.
- Full B1 accuracy matrix.

Implemented so far: deterministic partition builder, split-conformal
calibration, content-hashed artifact bundles, `compile_calibrated`, the
acceptance model, the screener registry, the group-marginal router trainer,
screener artifacts, screened-plan cache keys, the factorial evaluator, and the
constrained/free-form vLLM screener hook. Published multi-seed B1 numbers and
calibrated end-to-end tables remain release work.

DoD:

- Empirical coverage is within the reported confidence interval of nominal.
- Calibration and test splits are independent.
- On-policy and teacher-forced results are reported separately.
- Pareto curves are reproducible from the manifest.

### M6 — SGLang integration and release benchmark

- SGLang adapter.
- B1–B4 release run, multi-seed, report generator.
- CI/nightly/release gates.

Implemented so far: the SGLang general-plugin hook (draft block position 0,
validated end to end with Qwen3.5 built-in MTP), the DFlash block-position-0
slice in vLLM, the factorial B1 matrix, hashed release manifests and Markdown
reports for every result schema, B5 manifest comparison, and nightly/release
workflows. Multi-seed B1–B4 results are operator data.

DoD:

- One command reproduces each published table and figure. **`draftslice bench
  report` renders all implemented schemas; the published run inputs remain
  external.**
- Every result has a manifest, dataset hash, and code commit. **Hashes and the
  commit are recorded; dataset hashes are supplied in result metadata.**
- Negative results and failed configurations are reported. **Warnings and
  missing result kinds are rendered and flagged by `bench compare`.**

## 14. Risks and open questions

- **Grammar backend coupling.** xgrammar/outlines state APIs differ. Keep the
  adapter thin and test against real schemas early.
- **Dynamic shapes vs CUDA graphs.** Candidate sets change every step. Buckets
  may leave performance on the table; measure the capture rate.
- **Gather efficiency.** Leaf-contiguous layout is not automatically faster.
  Treat it as a hypothesis to measure, not a default.
- **Batch union complexity.** The optimal per-row/union crossover depends on
  `B`, `m`, and the kernel. Do not hardcode it.
- **Calibration drift.** On-policy distributions differ from teacher-forced
  data. Recalibrate on deployment data and monitor coverage.
- **Cost-model overfitting.** Fit on one set of hardware/configs and report error
  on held-out configs.
- **Engine API churn.** Pin versions and isolate adapters.
- **Strict-mode performance.** A large valid set may make slicing unprofitable;
  the fallback must be fast and clearly reported.
- **Tree planning.** State grouping and union execution need real grammar
  traces; synthetic states are not enough.
- **Head share / Amdahl.** A head-only speedup is bounded by the draft head's
  share of the measured server step. Parameter share is not step-time share;
  measure both and report the zero-head ceiling.
- **Tau/latency tradeoff.** Lowering `tau` can raise throughput when it removes
  draft/verify work, but it can also lower it by adding rounds. Validate any
  trade on the server metric, not on `tau` alone.
- **Default mode.** Strict is safe but may be slower; calibrated is faster but
  changes acceptance. The default must be explicit in config and docs.

## 15. Relationship to the local research workspace

The repository's public scope is the package and its interfaces. The ignored
`research/` workspace contains the exploratory code, data, and pilot results that
motivate this plan. The architecture deliberately turns the research findings
into contracts:

- the unified `(partition, screener, cost, allocation, guarantee)` abstraction;
- screen-then-exact-rescore;
- static vs dynamic bytes in the cost model;
- batch-dependent union execution;
- separate head-quality and kernel measurements;
- calibrated coverage as an explicit mode.

Pilot numbers must not be copied into public docs as benchmark results. They are
inputs to the benchmark design, not the report.

## 16. vLLM integration

The first vLLM integration is a **strict-mode draft-step-0** path. It is
registered as a vLLM general plugin and exposes two custom speculative methods:

- `draftslice_mtp` (wraps vLLM's `MTPSpeculator`);
- `draftslice_eagle3` (wraps vLLM's `EagleSpeculator`);
- `draftslice_dflash` (wraps `DFlashSpeculator`; block position 0 is sliced,
  later positions use the full head because their grammar state is sampled
  inside the same parallel block).

The worker-side xgrammar shadow is advanced by the tokens the target actually
sampled, so the draft sees the exact next grammar state (not the one-step-ahead
scheduler bitmask). Slicing is still **opt-in** (`DRAFTSLICE_VLLM_SLICE=1`) in the
current build: it is safe on supported eager/TP=1/unquantized paths, but it is a
regression on free-form and some workloads, and it cannot run inside a FULL
decode CUDA graph. The default is therefore a pass-through until per-workload
routing and graph/TP/quantized support land. The B4 runner and debug counters
record the fallback rate.

Entry point:

```toml
[project.entry-points."vllm.general_plugins"]
draftslice = "draftslice.integrations.vllm:register"
```

### 16.1 Hook points

The integration uses three vLLM extension points:

1. **Plugin registration**: `register_speculative_method` registers the custom
   config classes. `SpeculativeConfig` moves the custom name to `custom_method`
   and uses the built-in base method (`mtp` / `eagle3`) for draft-model setup.
2. **Grammar bridge**: vLLM sends a structured-output token bitmask to the
   worker. `install_grammar_bridge()` patches
   `StructuredOutputsWorker.apply_grammar_bitmask` to capture the bitmask,
   request ids, mask stride, and vocabulary size in a context variable.
3. **Speculator wrapper**: the mixin overrides `propose`, `sample_draft`, and
   `compute_draft_logits`. `propose` captures the grammar snapshot and the
   request-state mapping; `sample_draft` records the draft-step index;
   `compute_draft_logits` builds the sliced logits for draft step 0.

### 16.2 Draft logits construction

For each batch row with a captured valid set:

- group rows by identical valid-token set;
- call the Triton slice kernel (or the torch gather fallback);
- scatter the computed logits into a float32 `[batch, vocab]` tensor;
- fill every non-candidate column with `-inf`;
- rows without a valid set, with an empty set, or with an oversized set use the
  built-in full-head `compute_draft_logits` path.

The logits processor's `scale` and `soft_cap` are applied to the sliced values so
the draft distribution matches the full-head path.

`DRAFTSLICE_VLLM_FULL_MASK=1` is a diagnostic control, not a product mode: it
computes the full draft head and then applies the same exact grammar mask. It
separates the **acceptance** effect of grammar-aware drafting from the **latency**
effect of slicing. It is off by default.

When `DRAFTSLICE_VLLM_SCREENER` names a registry source, the adapter first asks
that source for candidates on the draft hidden states and then intersects them
with the exact valid set for each row (`A := A ∩ V`). Only candidate selection is
approximate; every proposed draft token is still grammar-valid and the target
verifier remains exact. Rows without an exact set, screener failures, and empty
intersections fall back to the full head. The counters
(`screener_calls`, `screener_rows`, `screener_dropped`, `screener_empty_rows`,
`screener_errors`) are exposed through `draftslice_stats()`.

### 16.3 Safety fallbacks

The integration falls back to the full draft head when any of the following is
true:

- `DRAFTSLICE_VLLM=0`;
- no grammar snapshot was captured;
- the draft step is not 0;
- the LM head's tensor-parallel shard layout cannot be proven from its metadata;
- the LM head is quantized;
- the weight dtype is neither bf16/fp16/fp32 nor a supported fp8/int8 quant;
- the logits processor has `logits_as_input=True`;
- the valid set is empty or larger than `DRAFTSLICE_VLLM_MAX_ROWS`;
- the configured screener cannot be built or raises on `candidates()`;
- the intersection of the screener candidates with the valid set is empty;
- any exception occurs during plan construction or kernel execution.

`DRAFTSLICE_VLLM_DEBUG=1` prints per-speculator call/slice/fallback counters to
stderr. `draftslice_stats()` exposes the same counters, the `fallback_reasons`
histogram, TP rank/shard fields, and `last_contract` (the concrete
`DraftContract` for the most recent step). Under TP>1 each rank scores only its
own global-token-id range; a row whose valid set has no token on that rank keeps
the full head but is masked to V before sampling, so a rank-local fallback can
never propose a grammar-invalid token. The full-head row set is derived from
*all* ranks' id ranges, so every rank enters vLLM's logits all-gather for the
same rows; a rank-local decision would deadlock the TP engine. LOCAL-argmax
reduction is supported through a `get_draft_top_tokens` override that slices the
head and reproduces vLLM's `(value, id)` TP reduction. A quantized head whose
scale layout cannot be mapped to rows keeps the full head with an exact mask.

### 16.4 Grammar-state timing and the worker shadow

The scheduler's grammar bitmask is computed **before** the target forward and is
one state ahead of the token the draft must propose after the target samples.
Using it directly drops valid draft tokens on a changing grammar, which is why
the first cut was experimental.

The current integration fixes this with a worker-side xgrammar shadow
(`vllm_shadow.py`):

- a matcher per request is compiled in the worker from the request's structured
  output parameters;
- `GPUModelRunner.postprocess_sampled` advances it by the tokens the target
  actually sampled (accepted drafts plus the bonus);
- for `spec_tokens>1`, a forked matcher is advanced by each sampled draft token,
  so every draft position gets its own exact valid set;
- with a reasoning parser, the shadow builds a per-request parser in the worker
  and stays inactive until the end-of-reasoning boundary is observed in sampled
  tokens; reasoning rows return no valid set (full head) and the grammar is only
  fed the post-boundary token suffix;
- valid sets are cached by grammar definition plus accepted-token prefix.

Supported rows therefore use an exact valid set. The target verifier remains
full-head exact: invalid drafts are rejected and trimmed by vLLM's structured
output path.

Slicing is still opt-in for **operational** reasons, not because the state is
missing:

- free-form rows have no grammar state and currently pay wrapper overhead;
- FULL decode CUDA graphs cannot read step-0 candidate ids at replay without a
  D2H sync (later-position sets are GPU-resident), so they use a static packed
  bitmask + fused mask instead of head-row slicing;
- a server-level router is not implemented yet (the adapter-level free-form
  router is).

A deployment that enables slicing should restrict it to structured workloads;
TP>1, quantized heads, reasoning parsers, local-argmax reduction and FULL graphs
all have concrete contracts. The B4 test compares constrained greedy output
against the full-head constrained baseline, `output_validity == 1.0`, and draft
acceptance/fallback rates.

`DRAFTSLICE_VLLM_SLICE=0` remains a safe pass-through with explicit counters;
`DRAFTSLICE_VLLM_SLICE=1` enables the shadow + slice path for supported rows.

### 16.5 Usage

```bash
# The plugin registers itself when vLLM loads.
vllm serve Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"draftslice_mtp","num_speculative_tokens":1}'
```

For a Python smoke test, use a real file (vLLM's spawned engine core cannot read
a script from stdin). The local MTP structured-output smoke test has been run
with the local model and a JSON schema; it produced valid JSON and the speculator
reported sliced calls.

Environment controls:

- `DRAFTSLICE_VLLM=0` disables slicing and keeps the wrapper as a pass-through.
- `DRAFTSLICE_VLLM_SLICE=1` enables the shadow + draft-step-0 slice path for
  supported rows; `0` keeps the safe pass-through.
- `DRAFTSLICE_VLLM_FULL_MASK=1` enables the diagnostic full-head + grammar-mask
  control (no slicing), for separating acceptance from latency.
- `DRAFTSLICE_VLLM_MAX_ROWS` caps the strict slice size (default 32768).
- `DRAFTSLICE_VLLM_SCREENER` names an opt-in `screen.registry` source (for
  example `static`, `group`, `lowrank`, `router`, `svd_lowrank`); candidates are
  intersected with the exact grammar set. Empty/unset means off.
- `DRAFTSLICE_VLLM_SCREENER_ARGS` is a JSON object passed to the screener
  factory (for example `'{"rows": [1, 2, 3]}'` for `static`).
- `DRAFTSLICE_VLLM_SCREENER_ARTIFACT` loads a verified `screener/1` bundle
  instead of constructing the policy from JSON; the bundle's calibration id and
  delta are reported in the stats.
- `DRAFTSLICE_VLLM_SCREENER_UNIVERSE` is `valid` (default) or `vocab`. `vocab`
  uses candidates directly on free-form rows and still intersects every row
  that has a grammar state.
- `DRAFTSLICE_VLLM_CUDA_GRAPH=1` enables the kernel-level graph replay cache. It
  is opt-in because capture must not run inside the engine's own graph capture;
  capture failure falls back to eager.
- `DRAFTSLICE_VLLM_ROUTER=1` routes an all-free-form batch through the stock
  speculator path without shadow/mapping bookkeeping. It removes the wrapper
  cost when no row has a grammar state and leaves structured rows unchanged.
- `DRAFTSLICE_VLLM_BLOCK_ALLPOS=1` enables DFlash all-position grammar masking
  under greedy sampling. It is batch-gated by
  `DRAFTSLICE_VLLM_BLOCK_ALLPOS_MAX_BATCH` (default 1: all-position at B=1,
  position-0 slice above that) and can be depth-capped with
  `DRAFTSLICE_VLLM_BLOCK_ALLPOS_DEPTH`.
- `DRAFTSLICE_VLLM_BLOCK_ALLPOS_FAST=1` applies the per-position bitmasks with
  one packed upload and one fused Triton launch instead of unpacking per row.
  It removes the B>=8 penalty of all-position masking; default-off for now.
- `DRAFTSLICE_VLLM_DETERMINISTIC=1` skips the draft head for rows whose valid
  set contains a single token. It is a negative result on the reference tool
  schemas (no such rows occur with xgrammar), kept as a research control.
- `DRAFTSLICE_VLLM_DEBUG=1` prints counters to stderr.

## 17. Fair baseline suite

### 17.1 Common policy interface

Every baseline implements:

```python
class CandidatePolicy(Protocol):
    name: str
    def candidates(self, hidden_states: Tensor) -> list[TokenSet]: ...
```

The runner receives the same hidden states, oracle top-k, `k`, dtype, and
hardware for every policy. It computes the same B1 metrics (`top1`,
`set_recall@k`, `prefix_coverage@k`, `OA`, `all@k`, `valid_recall`,
`valid_precision`, `m_mean`).

### 17.2 Implemented reference policies

| Policy | Class | Notes |
|---|---|---|
| full head | `FullPolicy` | oracle baseline |
| static rows | `StaticPolicy` | frequency / fixed row set |
| SVD low-rank | `LowRankPolicy.from_weight` | randomized SVD; reference implementation |
| trained low-rank | `LowRankPolicy.from_trained` | accepts `Vt`/`U` artifacts |
| cluster shortlist | `GroupPolicy` / `make_group_policy` | centroid dot, FlashHead-like |
| MLP router | `RouterPolicy` | DynaSpec/CHESS-like group shortlist |
| oracle group mass | `OracleGroupPolicy` | non-deployable upper bound |
| CFHead reference | `CFHeadPolicy` | tile-wise codebook + index map + top-k, follows the official repo |
| FlashHead reference | `FlashHeadPolicy` | balanced spherical k-means + top-p cluster probes |
| VocabTrim reference | `make_vocabtrim_policy` | static retained-vocabulary rows |

The CFHead/FlashHead/VocabTrim implementations are **reference
reimplementations** of the algorithms in
[`HigherIdeal/CFHead`](https://github.com/HigherIdeal/CFHead) (IEEE TC 2026),
adapted to the shared `CandidatePolicy` interface. They are not copies of the
official evaluation harness and should be labelled as reference implementations.
The official repository is research-only; the public DraftSlice package should
carry the attribution and citation in its baseline documentation.

CFHead/FlashHead artifact construction is expensive for a 248k vocabulary. Build
the policy once per weight/configuration and reuse it across B1 cases; do not
rebuild inside the per-query loop.

### 17.3 Runner

```python
from draftslice.bench.baseline_runner import HeadQualityCase, run_matrix
from draftslice.bench.baselines import (
    FullPolicy, StaticPolicy, LowRankPolicy, make_group_policy,
)
from draftslice.bench.official_baselines import (
    CFHeadPolicy, FlashHeadPolicy, make_vocabtrim_policy,
)

case = HeadQualityCase(hidden=H, truth=topk)
policies = [
    FullPolicy(vocab_size=V),
    make_vocabtrim_policy(retained_rows),
    LowRankPolicy.from_weight(weight, rank=128, top_m=4096),
    make_group_policy(weight, leaf_groups, top_c=16),
    CFHeadPolicy(weight, tile_size=8, cluster_size=4, top_k=8192),
    FlashHeadPolicy(weight, num_clusters=8016, num_probes=512),
]
results = run_matrix(policies, case, k=5)
```

Rules:

- every policy gets its own knob sweep;
- compare at iso-bytes or iso-latency, not only one point;
- label reference implementations;
- report `m_mean`, candidate bytes, and confidence intervals;
- never compare a tuned DraftSlice point to a baseline default.

### 17.4 End-to-end baseline design

The offline head-quality matrix is the first fair comparison. The end-to-end
matrix uses the same vLLM engine and workload with:

- full head / no spec;
- built-in `mtp`;
- built-in `eagle3` when a checkpoint is available;
- built-in `dflash` with `z-lab/Qwen3.5-4B-DFlash`;
- `draftslice_mtp`;
- `draftslice_eagle3`;
- optional static/low-rank draft heads once they are wired as speculators.

Report `tau`, throughput, TTFT, ITL, output validity, and task metric. A
head-only speedup is not an end-to-end claim.

### 17.5 Kernel-level fairness

The B2 runner now reports the same four dimensions for every implementation:

- eager latency;
- CUDA-graph replay latency, enabled for full, gather, and fused paths;
- effective weight bandwidth and speedup vs full and vs the naive gather;
- numerical max-abs difference against the torch gather reference.

A fused-vs-baseline claim must use the same mode on both sides. If graph replay is
enabled for DraftSlice, it is enabled for `TorchFullKernel` and
`TorchGatherKernel` too, and the result records capture/replay/fallback counts.
The tiny-M target is only meaningful under this rule.

## 18. Remaining design

The following pieces are the remaining architecture work. Some are implemented
offline but not wired into the engine hot path; others are not implemented yet.
Status is stated per item.

### 18.1 Union/tree execution

Per-node union execution and scatter are implemented (`runtime/union.py`). The
remaining work is a tree-specific fused kernel that writes directly to per-node
buffers instead of materializing the full union matrix, plus union-size and
kernel-crossover benchmarks at B=1/8/32/64 with real grammar traces.

### 18.2 Scheduler v2

The acceptance model, optional `min_expected_tau`, and `choose_best` are
implemented. What remains is fitting the prefix-coverage model on real B1/B4
traces, validating predicted `tau` against measured acceptance, adding server
metrics (throughput/TTFT/ITL/cost) to the objective, and wiring the scheduler
into an engine adapter. The scheduler must be allowed to choose a lower-`tau`
point when the server metric improves; `min_expected_tau` is an optional floor,
not a universal constraint.

### 18.3 Artifact builder and calibrated screener

The partition builder, split-conformal calibrator, artifact bundle,
`compile_calibrated`, the `screen.registry` (full/static/low-rank/group/router/
CFHead/FlashHead/VocabTrim), `compile_screened` / `compile_step(screener=...)`,
the router trainer, screener artifact bundles, the screened-plan cache, and the
factorial evaluator are implemented. The vLLM adapter has an opt-in constrained
screener hook (`DRAFTSLICE_VLLM_SCREENER` / `_ARTIFACT`) and a free-form
`universe=vocab` mode that still intersects rows with a grammar state. The
remaining work is a full multi-seed B1 matrix on the prescribed dataset mix and
published calibrated end-to-end numbers; the code paths exist but the public
repository does not ship datasets or result files.

### 18.4 vLLM hardening

- TP > 1 is supported: the adapter detects the rank's contiguous vocab range
  from the head's shard metadata, slices only that rank's weight rows, and
  scatters logits to global token-id columns. A rank with no local candidate
  takes a full-head fallback masked to V; the full-head row set is derived from
  all ranks' id ranges so the TP logits all-gather stays balanced. The custom
  method's config path and end-to-end slicing were verified under vLLM 0.34
  TP=2 (Qwen3.5-0.8B structured JSON, valid output, `sliced_calls > 0` on both
  ranks); the previous config-resolution failure no longer reproduces.
- Quantized heads: a quant config no longer rejects a bf16 head outright; fp8/int8
  weights are dequantized per candidate row (`quant_dequant.py`) and cached, so
  the same sliced kernels apply. Unknown scale layouts fail closed to a full head
  masked to V with a recorded reason.
- FULL decode graphs: a static packed bitmask is refreshed from the step-0 shadow
  in `propose()` before each replay and applied by the fused mask kernel; step≥1
  stays full head because its state is GPU-resident. Local-argmax reduction uses
  a `get_draft_top_tokens` override with the same `(value, id)` TP reduction as
  vLLM. Reasoning parsers gate the shadow on the observed reasoning boundary.
- Wire a **workload router** so structured/eager/TP=1 requests use the slice path
  and free-form/unsupported requests take the full path by default. The
  adapter-level "no grammar in this batch" branch is implemented
  (`DRAFTSLICE_VLLM_ROUTER`); the remaining piece is a **server-level** router
  that also sends short structured workloads with no acceptance gain (for
  example small JSON arrays or regex fills) to the built-in speculator.
- Add per-request calibration artifacts and coverage monitoring for calibrated
  modes.
- CUDA graph buckets for dynamic candidate sizes. The kernel-level
  `CudaGraphCache` exists; the adapter must still keep a stable hidden workspace
  and row buffer.
- Expose stats through the engine metrics path, including graph
  capture/replay/fallback counters and head-share/timing fields.
- Replace the fixed `spec_tokens` default with a server-level k/source choice
  that may trade `tau` for throughput or tail latency.

The constrained/free-form screener hook (`DRAFTSLICE_VLLM_SCREENER` /
`_ARTIFACT` / `_UNIVERSE`) is implemented; the remaining gap is production
routing, TP/quantized support, per-request calibration, and measured server
Pareto results.

### 18.5 DFlash

DFlash is a required future draft family. The official Qwen3.5-4B checkpoint is
`z-lab/Qwen3.5-4B-DFlash` (public HF ID); it has been downloaded to the operator
model root and a vLLM `method="dflash"` smoke test has passed with a 8-token
generation. vLLM 0.34 already ships `DFlashSpeculator` and
`DFlashDraftModel`; SGLang also supports it.

The DraftSlice adapter is registered as `draftslice_dflash`. Block position 0
is sliced with the exact worker-shadow valid set; positions 1..N-1 stay on the
full head because their grammar state depends on draft tokens sampled later in
the same block. This is a deliberate partial optimization: it never applies the
position-0 candidate set to later positions, so no valid later-position token is
dropped. The block path is validated by a real Qwen3.5-4B + 4B-DFlash run
(`block_step0_rows` increments, validity 1.0).

The adapter accounts for DFlash's **parallel block drafting**:

- each request emits a bonus query plus one query per draft token;
- the draft sample rows carry `sample_col = 0..N-1` and share a request state;
- the draft head produces logits for a whole block in one call, not one
  `sample_draft` call per position.

A single captured grammar bitmask is not enough to constrain every block
position. The implemented policy is:

1. position 0 uses the worker shadow's exact valid set;
2. positions 1..N-1 use the full draft head;
3. if the shadow has no exact set, the whole block uses the full head.

An opt-in all-position **mask** path (`DRAFTSLICE_VLLM_BLOCK_ALLPOS=1`) closes
most of the acceptance gap for greedy decoding: it masks each position with its
own exact set, takes the argmax to advance a forked matcher, and lets the engine
sample from the masked logits. It is batch-gated by default
(`..._MAX_BATCH=1`) because the per-position grammar/argmax work scales with
batch and loses above B=1; above the gate the adapter uses position-0 slicing.
The target verifier remains exact.

Per-position **slicing** for positions > 0 remains future work. Masking does not
reduce the head work: the block still computes the full head once. A slice would
need a grammar-aware sequential block sampler that reuses the engine's sampled
tokens; that serializes the block head and gives up the batched GEMM. The
longer-term target design is a `StepPlan` with one node per block position, each
node carrying its own `HeadPlan`.

### 18.6 SGLang

- The integration installs as a real `sglang.srt.plugins` entry point and patches
  `EagleDraftWorker.draft` / `FrozenKVMTPDraftWorker.draft` to publish the batch
  grammar state, and `LogitsProcessor._compute_lm_head` to slice draft block
  position 0 through the shared row kernel. Later positions use the full head.
- TP > 1 is supported: SGLang's `_compute_lm_head` returns rank-local logits
  before its own TP gather, so the hook slices local weight rows with local ids
  and returns local logits; no collective is needed inside the hook. Unsupported
  layouts, CPU weights, missing/terminated grammars, oversized valid sets, and
  errors still fall back to the original head with a recorded reason and a full
  `DraftContract`.
- End-to-end validated with SGLang 0.5.23 + Qwen3.5-0.8B and its built-in MTP
  (`speculative_algorithm="EAGLE"`, `speculative_draft_model_path` = target):
  step-0 slicing fires (`sliced_calls > 0`, candidates 2..443), structured JSON
  output is valid, and greedy output ids are identical with the hook on/off.
- The pure `screen_logits` helper remains available for hosts that prefer a
  post-head mask.

### 18.7 Release benchmark

- B1 baseline matrix on UltraChat-200K + ShareGPT and structured workloads.
- B2 fixed-clock kernel table with head-share and zero-head-bound reporting.
- B3 real grammar-state replay (the runner and xgrammar adapter exist).
- B4 end-to-end matrix with validity, tau, throughput, TTFT/ITL, cost, and
  quality (the runner exists; it is vLLM-only today).
- A tau/latency Pareto: sweep `k`, candidate budget/source, and draft depth;
  report the best server point, not only the best tau point. Sacrificing tau is
  allowed when throughput or tail latency improves.
- `draftslice bench report` builds a hashed manifest and Markdown report from
  any mix of B1-B4 results; `bench validate` and `bench compare` implement the
  B5 gate. Nightly/release workflows run the offline pipeline on synthetic data.
- Remaining: publish multi-seed results for the prescribed dataset mix and the
  reference GPU; add an SGLang B4 runner; the repository intentionally ships no
  datasets or result files.

### 18.8 Kernel work remaining

- A tree-specific fused union kernel; the current executor reuses the slice GEMM
  and scatters after the launch.
- Engine-owned shape buckets with stable hidden and row buffers.
- Exotic quantization scale layouts (block/MX) that cannot be mapped to
  candidate rows; common fp8/int8 rows already dequantize per row.
- A split-K or persistent tiny-M kernel only if graph replay is not available in
  the target engine.

### 18.9 State timing, graph mode, and the server objective

The original B4 sweep exposed a real blocker: using the scheduler bitmask
directly slices the draft head to the previous grammar state, drops valid draft
tokens, and lowers acceptance. The worker-side shadow in §16.4 now fixes this on
supported rows; the scheduler bitmask is no longer the draft state source.

What remains is operational, not a state-timing blocker:

- **Graph mode:** a FULL decode CUDA graph bypasses the Python hook at replay,
  but the adapter refreshes a static packed bitmask in `propose()` before replay;
  step 0 emits a `mask`/full contract and later positions stay full-head with a
  recorded reason. True per-row slicing inside the graph would require a D2H sync
  for step≥1 states, so it is deliberately not attempted.
- **TP/quantized heads:** TP>1 slices per-rank shards (vLLM) or local rows
  (SGLang, gathered by the engine afterwards); the vLLM path max-reduces the
  partial logits and agrees the full-head row set across ranks, because the
  engine samples on every rank. Quantized heads dequantize candidate rows. Both
  paths emit a contract and a fallback reason.
- **Routing:** free-form rows and some structured workloads regress if slicing
  is enabled globally, so the production default is built-in and DraftSlice is
  enabled only by a measured profile cell. The 4B/B8 routing experiment found
  0.767× for a blanket structured slice; the single-profile built-in path is the
  production default. See `docs/production-architecture.md` §1–§2.
- **Server objective:** acceptance length is not the product metric. The next
  evaluation must report throughput, TTFT/ITL, validity, and cost, and may
  deliberately trade `tau` for a lower-latency operating point. A head-only
  speedup or an acceptance-only improvement is not a server claim.
- **Head share:** the draft head is a model/engine/workload-dependent fraction of
  the step, not a constant. Measure it per deployment; do not import a
  parameter-share estimate as an e2e ceiling.
