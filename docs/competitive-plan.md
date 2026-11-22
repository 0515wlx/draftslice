# DraftSlice competitive plan

> **Production status 2026-11-22:** see
> [`production-architecture.md`](production-architecture.md). The routing
> experiments show that blanket structured slicing is a regression on
> Qwen3.5-4B/B8; built-in is the default and DraftSlice is a measured per-cell
> profile. Failed modes are archived in that file §1.

Status: internal planning document, updated 2026-11-22. It describes the current
gaps and the path to a defensible server-level advantage over static, low-rank,
tile/cluster screening, and router-shortlist draft heads. It does not publish
benchmark results.

2026-11-22 closeout + routing experiments (local report, opt-in code): the
adapter-level free-form router is implemented and recovers the free-form
regression; the fused DFlash all-position mask wins at B=1 and is neutral at
B=8, but only within the DFlash family; a static free-form screener is a
measured negative. TP>1, quantized heads, and FULL-graph step-0 masking are
code-complete and unit-tested; real-engine gates remain per cell, and a
2026-11-22 35B-A3B TP=2 `draftslice_mtp` run hung while built-in MTP TP=2 was
healthy, so MoE TP>1 is gated. The 4B/B8 routing experiment found that a blanket
structured slice is 0.767× on a mixed workload, so the production default is
built-in and DraftSlice is profile-gated. Results are in the local `runs/`
workspace.

Related documents:

- [architecture.md](architecture.md) — components and milestones.
- [benchmark.md](benchmark.md) — benchmark tiers, metrics, and protocols.

## 1. Executive summary

Status update (2026-11-22): the repository now has the plan core, a unified
candidate-source registry, an xgrammar backend, union/tree execution, router and
artifact tooling, vLLM/SGLang adapters, and a B4 runner. The remaining risk is
not “missing components”; it is **proof that the composed system beats a strong
fixed baseline at the server level**. The 2026-11-22 closeout adds an adapter
free-form router and a batch-gated DFlash all-position mask, and records a
negative static-screener result.

The missing pieces are:

1. engine wiring: the production fast path still bypasses the plan/cost/scheduler/
   union/cache layer; the profile table is not yet the hot-path decision source;
2. measured profile cells: 4B B=1 tool, 35B-A3B tool, and DFlash-vs-MTP; a
   blanket structured slice is a measured negative on 4B/B8;
3. a positive calibrated/free-form server result at iso-quality, not only an
   offline candidate-quality table (the first static point is negative);
4. a server-level tau/latency Pareto: choose `k`, draft depth, source, and budget
   per workload/batch, and allow a lower `tau` when throughput or tail latency
   improves;
5. measured head-share/zero-head bounds instead of parameter-share estimates;
6. multi-seed, manifest-backed release results on the prescribed datasets.

The strategic opening is **not** “a faster kernel for the same candidate set”.
It is:

> **Exact when constrained, calibrated when free-form, with one cost-aware
> execution contract and a server-level operating point.**

Existing methods usually pick one regime: static/low-rank compress the head,
cluster/tile methods screen the vocabulary, and router methods pick a shortlist.
DraftSlice can own the **contract** between the grammar state, the candidate set,
the kernel, the draft depth, and the server objective — and prove it where the
valid set changes every step or the best `tau` is not the best server point.

## 2. What is already in place

| Area | Current state | Can we claim advantage? |
|---|---|---|
| Plan types and strict coverage | Implemented and tested | Yes for coverage; needs real grammar corpus |
| Plan serialization and cache | Implemented; compiler-level cache wired | Engine fast path does not use it yet |
| Cost model | A100/H100 roofline; uncalibrated | No; MAPE and head-share calibration missing |
| Layout | native/grouped layout + packing | Not used by the engine kernels |
| Torch/Triton kernels | full, naive gather, tuned fused GEMV/GEMM, CUDA graph, union | Head-only yes; e2e bound depends on head share |
| Scheduler | deadline-aware strict/full + acceptance model | Implemented offline; not engine-wired |
| Screener / calibrated mode | registry, partition, conformal, router, artifacts, vLLM hook | Code exists; no positive free-form e2e at iso-quality |
| Union/tree executor | implemented with per-node scatter | No tree-specific fused kernel / engine use |
| Grammar backend | xgrammar JSON Schema + regex; shadow | Yes for supported schemas; CFG/corpus missing |
| Engine integration | vLLM MTP/EAGLE3/DFlash pos0 + allpos mask; SGLang MTP step0 | Code-complete for TP>1/quant/FULL-graph; real-engine gates per profile cell |
| End-to-end benchmark | B4 runner (vLLM), structured/free-form/DFlash | Exists; no multi-seed server release |
| Baselines | full/static/low-rank/group/router/CFHead/FlashHead/VocabTrim | Head quality yes; latency/Pareto incomplete |

## 3. The honest gap analysis

### 3.1 The kernel now covers small, medium, and large `m`

The first fused kernel had a `~70 us` floor at small `m`: launch overhead dominated
and the naive gather baseline could win. The current kernel suite addresses this
in two steps:

- a row-block GEMV for `B=1` and a shape-dispatched tensor-core GEMM for `B>=2`
  reduce eager time across `m=1..248k`;
- an optional CUDA-graph replay cache removes most of the launch cost for tiny
  and medium slices, with the same graph path offered to the full and gather
  baselines.

The honest regime split is:

- tiny `m`: graph replay is launch-bound; the fused and gather paths are close,
  and absolute latency depends on graph replay overhead;
- medium `m`: eager fused is competitive and graph replay wins clearly;
- large `m`: fused is bandwidth-bound and beats gather by a wide margin.

The remaining kernel gaps are the tree-specific fused union kernel and
engine-owned graph buckets. Union/tree execution and runtime autotuning are
implemented; the eager tiny-M floor is no longer the main issue.

### 3.2 The screener exists; the free-form advantage is unproven

The calibrated path is implemented: vocabulary partition, static/low-rank/group/
router screeners, candidate-set cache, split-conformal calibration, exact
rescoring inside the candidate set, and `constrained_calibrated` /
`freeform_calibrated` modes. Artifact bundles, router training, and the vLLM
screener hook exist.

What is missing is evidence:

- no positive free-form server result at iso-quality;
- a static-screen smoke test showed that dropping candidates can destroy `tau`;
- low-rank/router artifacts have only offline B1/factorial numbers;
- the engineering path is not wired to the plan/cost/scheduler layer.

Without a measured Pareto point, the project cannot claim a free-form advantage.

### 3.3 Union/tree executor exists; engine and tree-kernel use do not

Per-node union execution and scatter are implemented (`runtime/union.py`). The
architecture's runtime lever is:

- deduplicate candidate rows across batch items;
- deduplicate rows across draft-tree nodes;
- run one union kernel and scatter to per-request or per-node outputs.

What is missing is a tree-specific fused kernel that writes directly to per-node
buffers, union-size/crossover benchmarks with real grammar traces, and any engine
use of the union executor. Until then, measured speedups are per-plan and the
scheduler cannot trade tree width against kernel cost.

### 3.4 Grammar backend exists; real plan distribution is missing

The xgrammar adapter supports JSON Schema and regex, and B3 trajectory recording
exists. It still cannot answer at release quality:

- how large valid sets are for JSON Schema, tool calls, regex, SQL, and CFG;
- how often adjacent steps reuse the same grammar state;
- how much `PlanCache` helps;
- how often strict slicing falls back to full head;
- how much time plan compilation adds per step.

The next step is a real grammar replay corpus and state-hash stability, not a new
adapter.

### 3.5 B4 exists; server-level and multi-seed claims are missing

The B4 vLLM runner now covers structured, free-form, and DFlash cases. It reports
validity, `tau`, throughput, and greedy equality, and the worker shadow removes
the old state-timing blocker.

What is still missing:

- server-level TTFT/ITL/cost and head-share decomposition;
- a tau/latency Pareto that allows a lower `tau` when the server metric improves;
- TP>1, quantized, and CUDA-graph operating points: code-complete and
  unit-tested in the working tree; real-engine gates remain per profile cell;
- a measured profile table: 4B/B8 blanket structured slice is a regression
  (0.767× mixed workload), so built-in is the default;
- an SGLang B4 runner;
- multi-seed, manifest-backed release results.

Head-only latency is not the product metric; a `10×` head speedup can be a `1%`
or `20%` server speedup depending on the measured head share.

### 3.6 The cost model is not calibrated

The model predicts a full head faster than measured (the current A100 profile
uses a `2.0 TB/s` peak instead of the fitted effective bandwidth). Before the
scheduler uses predicted latency for decisions, it must be fitted on B2 data and
validated on held-out shapes. The benchmark gate is already defined: report MAPE
on disjoint shapes, not on the fit set.

### 3.7 The baseline suite exists; the latency matrix is partial

The candidate-policy suite now covers full, static top-M, SVD and trained
low-rank, CFHead-like tile/index screening, FlashHead-like cluster screening,
group/router shortlists, oracle group mass, and VocabTrim-style retained rows,
all over the same hidden states and oracle top-k. Each policy can be swept at
iso-bytes.

What is still missing is the end-to-end latency comparison: the screening cost of
each baseline must be measured with the same kernel effort as the DraftSlice path
and reported next to candidate quality. A policy-only table is not a latency
claim.

### 3.8 Artifact pipeline exists; calibrated e2e is missing

The `ArtifactBundle` builder, router training, conformal calibration, and layout
builder exist. The remaining gap is a positive calibrated end-to-end result with
measured coverage and a server-level Pareto point; the code paths are not the
blocker.

## 4. Where the advantage can come from

### 4.1 Contract advantage: exact constrained path

For grammar/schema-constrained decoding, DraftSlice can guarantee:

- every grammar-valid token is in the strict plan;
- target verification remains exact;
- plans are cached by grammar state across requests;
- constrained output is identical to full-head constrained decoding under greedy
  decoding.

This is a correctness and operability advantage over approximate draft-head
compression. It is especially valuable for tool calling and structured output,
where a dropped valid token is a quality bug, not just a slower draft.

**But** it only wins if the valid-set slice is actually cheaper than the full
head, if plan compilation/caching overhead is small, and if the saved head time is
a material share of the server step. Tiny-M latency and state replay are now
implemented; head-share analysis is the remaining gate.

### 4.2 Accuracy-per-byte advantage: calibrated screen + exact rescoring

The local research direction is clear: do not approximate all logits; estimate
the argmax/top-k posterior at group level, then rescore the candidate set
exactly. The defensible claims are:

- screen-then-exact-rescore has zero resolution error inside the candidate set;
- split-conformal gives finite-sample coverage for the chosen candidate set;
- a cost-normalized posterior threshold is a principled allocation rule;
- the router only affects candidate size, not the coverage guarantee.

This is a stronger story than "low-rank is close enough" or "top-1 usually
survives". It also composes with grammar: the universe is the valid set, not the
full vocabulary.

### 4.3 Scheduling advantage: plan cost is known before execution

Because a plan is data, the scheduler knows:

- exact candidate rows and bytes;
- static vs dynamic traffic;
- predicted latency by hardware profile;
- tree/batch union size;
- expected acceptance from a coverage model.

This enables per-request and per-tree-node decisions:

- strict vs calibrated vs full;
- `C`, mass threshold, or conformal `C*`;
- draft depth and tree width;
- per-row vs union kernel;
- deadline-aware fallback.

Existing methods often fix `C` or use a static budget. A scheduler that selects
the Pareto point per step is a real differentiator, provided the acceptance model
is measured rather than guessed.

The objective is the **server metric**, not `tau`: throughput, TTFT/ITL, and cost
under a quality/SLO constraint. A lower-`tau` plan is valid when it removes draft
or verification work and improves the server metric. The scheduler must report
both `tau` and the server result for every trade.

### 4.4 Cache advantage: grammar state and candidate-set reuse

Two caches matter and are not the same:

- **grammar state -> plan** for strict slicing;
- **selected group set -> candidate rows** for screening.

Structured generation revisits grammar states and leaf sets. A cache hit removes
plan compilation and row-set construction from the critical path. The current
engine fast path does not consume these caches yet; wiring them is part of the
system claim.

### 4.5 Kernel advantage: union/tree execution and tiny-M specialization

The kernel must win in three regimes:

1. **tiny M** (tens to hundreds of rows): launch-bound; graph replay or a
   persistent/specialized tiny kernel.
2. **medium M** (thousands): bandwidth/occupancy-bound; better tiling and
   vectorized loads.
3. **large M / batched tree**: union rows across batch/tree, tensor-core GEMM,
   one gather.

Eager fused/union kernels and graph replay exist; the remaining gap is a
tree-specific fused union kernel and engine-owned graph buckets.

## 5. Prioritized plan

### P0 — Make the strict path real and fast (mostly done)

1. **xgrammar adapter** — implemented for JSON Schema and regex, with B3 replay.
   Remaining: a real grammar corpus and stable cross-request state hashing.
2. **Plan cache integration** — compiler-level cache is wired and tested.
   Remaining: use it in the engine fast path and report hit rate/compile time.
3. **Union/tree executor** — per-node union execution and numerical tests are
   implemented. Remaining: a tree-specific fused kernel and engine use.
4. **Tiny-M kernel path** — fused GEMV/GEMM and CUDA-graph replay are implemented;
   eager `m<=1k` is ~67 µs, graph replay ~25 µs at fixed A100 clock. Remaining:
   engine-owned graph buckets.
5. **vLLM strict integration** — plugin, worker shadow, per-position forks,
   block-position-0 slice, TP>1 shard, quantized-row dequant, FULL-graph step-0
   mask, and local-argmax are code-complete and unit-tested; block-position-0
   slicing has real-engine runs. Remaining: the measured profile table, real-
   engine gates for each new path, and server-level tau/latency measurement.

**Exit gate:** on structured tool-call workloads, the strict path gives a
measurable server-level gain at B>=8 with 100% validity and output equality, and
reports head share plus the best tau/latency point. The current results are
eager/TP=1 and do not yet satisfy the full production gate.

### P1 — Calibrated screener and Pareto (implemented, evidence missing)

1. **Artifact builder** — partition, layout, hashed bundles, and document/session
   split support are implemented.
2. **Router/static/low-rank screeners** — registry, group-marginal router
   training, artifacts, and reference CFHead/FlashHead/VocabTrim policies exist.
3. **Calibrated modes** — `constrained_calibrated` / `freeform_calibrated`,
   split-conformal `C*`, candidate-set cache, and exact rescoring are implemented.
4. **Acceptance model and scheduler v2** — acceptance model, `choose_best`, and
   server-level objective hooks exist; validation on real traces is missing.
5. **B1 baseline matrix** — offline factorial matrix exists; a real multi-seed
   dataset run and an e2e latency/Pareto table are missing.

**Exit gate:** at `<=5%` of full-head bytes, DraftSlice matches or beats the best
baseline on top-1/top-5/OA on the primary mix, with coverage within the reported
interval, **and** a positive server-level free-form point (or an explicit negative
result).

### P2 — System-level advantage

1. **Union/tree kernel optimization** — union executor and eager kernels exist;
   tree-specific fused kernel and graph buckets remain.
2. **SGLang integration** — step-0 MTP hook validated; EAGLE/DFlash/TP>1 and a
   B4 runner remain.
3. **Quantization and multi-model support** — both still fall back; this is a
   production blocker.
4. **Server-level tau/latency routing** — choose `k`, draft depth, source, and
   budget per workload/batch; allow lower `tau` when the server metric improves.
5. **Nightly B1/B2/B3 and release B4** — workflows exist on synthetic data;
   real multi-seed results remain operator work.
6. **Artifact/result registry and one-command reproduction** — implemented for
   implemented schemas; datasets and result files remain external.

## 6. What to measure to prove an advantage

The claim must be a Pareto statement, not a single point.

### Head level

- top-1, top-5 set recall, OA, all-5;
- candidate bytes and MACs;
- median/p90 latency at fixed clocks;
- effective bandwidth;
- cost-model MAPE;
- calibration coverage vs nominal;
- on-policy vs teacher-forced gap.

### Constraint level

- strict valid-token recall (must be 1.0);
- plan size distribution by valid-set bucket;
- plan cache hit rate;
- compile time p50/p90;
- fallback rate;
- constrained calibrated validity.

### End-to-end

- tau and tau distribution;
- throughput, TTFT, ITL p50/p90/p99;
- output validity;
- task metric;
- speedup vs full-head and vs no-spec;
- head share of decode time;
- the best server point across the tau/latency sweep, including any deliberate
  tau reduction and the reason it improved the server metric.

### Fair comparison rules

- same model, tokenizer, dtype, hardware, batch, sampler, seed;
- same kernel effort where possible; label reference implementations;
- every method gets a knob sweep;
- report negative results and fallback rates;
- no "MACs only" speedup claims.

## 7. Concrete targets and kill criteria

These are project gates, not published results.

| Gate | Target | Kill / pivot condition |
|---|---|---|
| Strict validity | 100% valid tokens covered; constrained greedy output identical | Any valid-token drop is a stop-ship bug |
| Tiny-M latency | `m<=1k`: <=30 us; `m=8k`: <=60 us at fixed A100 | If `m<=1k` stays >60 us, defer strict e2e claims |
| Kernel vs gather | fused beats gather for `m>=8k`; union beats naive at B>=8 | If union loses at B>=8, revisit kernel before scheduler |
| Cost model | MAPE <=25% on held-out shapes | If worse, recalibrate before scheduling claims |
| Head quality | at <=5% bytes, match/beat best baseline on OA/top-5 | If worse, improve partition/router, not kernel |
| Calibration | empirical coverage within ±2pp of nominal | If not, fix split/leakage before claiming coverage |
| End-to-end | server-level win at B>=8 on structured; no free-form server regression; target >1.2x in the best structured configuration | If <1.1x at B>=8, head share/overhead is the bottleneck |
| Tau tradeoff | a lower-`tau` point is valid only when throughput/TTFT/ITL improves at fixed quality | Lower tau with no server gain is a regression |
| Reproducibility | every table from a manifest and one command | No manifest, no claim |

## 8. Claim discipline

Avoid claims that the data does not support:

- "2.75% MACs" is not a latency claim; gather and launch overhead matter.
- "Bounded error" from low-rank is not a useful top-1 certificate on this head;
  the local analysis found the interval covers the full vocabulary.
- "Coverage" requires an independent calibration split and an empirical check.
- "Full head is optimal at high batch" is kernel-dependent; report the union
  efficiency.
- "Head speedup" is not "end-to-end speedup"; report the head share and the
  zero-head bound.
- "Higher tau" is not automatically a server win; a lower-tau point can win on
  throughput or ITL. Report both sides of the trade.
- "Parameter share" is not "step-time share"; head-share claims must use measured
  latency, not weight counts.

### 8.1 Prior-art watchlist (2026-11)

These are the closest systems/papers for a submission; they must appear as
baselines or explicit related work:

- **TableTune** (arXiv:2610.38968) — measured batch/draft-length tables in vLLM.
- **APEX** (arXiv:2610.07780) — request-level drafter selection + depth control.
- **SpecSearch** (arXiv:2611.22296) — joint strategy/draft-length/precision
  search.
- **ForeSpark** (arXiv:2611.13442) — joint draft/verify budget scheduling.
- **Raptor** (arXiv:2611.25436), **DScale** (arXiv:2609.37532), **ASPIRE**
  (arXiv:2609.17943) — adaptive draft/verify budgets.
- **SlimSpec** (arXiv:2605.10453), **SpecVocab** (arXiv:2602.13836),
  **DynaSpec** (arXiv:2510.13847), **VocabTrim** (arXiv:2506.22694),
  **NarrowSpec** (arXiv:2611.23835), **Weights That Stay** (arXiv:2611.00506) —
  draft-head/vocabulary cost.
- **FactorSpec** (arXiv:2611.02297), **SchemaFill** (arXiv:2610.07086),
  **MobiInfer** (arXiv:2611.08496) — structured/tool-call speculative drafting.
- **vLLM grammar-aware draft sampling** (PR #47885) — worker shadow + grammar
  guidance; compare against the mask-only control.

The project should not claim novelty for the generic controller or for a static
`k` table; those are covered. The defensible claim is the **exact valid-set
contract + row slicing + unified candidate-source/guarantee** evaluated at the
server level against these systems.

## 9. Recommended next sequence

### Stage 0 — server-level upper bound

- Normalize existing paired runs into source x k x workload cells.
- Compute the best fixed configuration, a per-workload router, and a per-cell
  oracle.
- Compute the zero-head bound from measured full-head time and step time.
- If the oracle is within ~10% of the best fixed configuration, do not build a
  complex controller; invest in a different lever.

### Stage 1 — engine wiring and routing

- Wire the measured profile policy (enabled cell -> slice; otherwise built-in
  full). **Adapter-level "no grammar in this batch" routing is implemented**
  (`DRAFTSLICE_VLLM_ROUTER=1`), but the 4B/B8 routing experiment shows that a
  blanket structured slice is a regression; the profile table is the required
  gate. A server-level gateway is optional and model-dependent.
- Use a per-model `k` table; measure whether a lower-`tau` point improves the
  server metric. **The measured table is per model/workload/batch; a per-cell
  learned router overfits 1-5-repetition cells.**
- Report head share, TTFT/ITL, throughput, validity, and fallback rates.

### Stage 2 — production blockers

- TP>1 sharding, quantized LM-head path, CUDA-graph-safe slicing: code-complete
  and unit-tested; real-engine gates remain per cell. 35B-A3B TP=2
  `draftslice_mtp` hung after 4 structured requests while built-in MTP TP=2 was
  healthy, so MoE TP>1 is a blocker. Quantized rows dequantize per candidate
  row; unsupported scale layouts fail closed to a full head masked to V.
  FULL-graph step 0 uses a pre-replay static packed bitmask; step≥1 stays full
  head.
- SGLang EAGLE/DFlash and a B4 runner. **Still open.** SGLang MTP step-0 and
  local TP rows are validated.
- Engine metrics for capture/replay/fallback and head-share timings.

### Stage 3 — calibrated/free-form Pareto

- Low-rank/router artifacts at iso-quality and iso-memory, not only iso-bytes.
- Conformal coverage on an independent split.
- A positive server point, or an explicit negative result; do not ship a
  policy-only table as a latency claim. **A frequency-top-16k static screener on
  free-form at B=8 halved `tau` and throughput; explicit negative.**

### Stage 4 — DFlash all-position

- Mask all block positions sequentially with an advanced grammar matcher.
  **Implemented (greedy, exact shadow, opt-in) and batch-gated.**
- If the acceptance gain is real, design the corresponding per-position slice.
  **The acceptance gain is real (tau 2.7 -> 5.1 at B=1), but all-position
  masking loses above B=1 because the per-position grammar/argmax work scales
  with batch. A per-position slice remains deferred.**
- This is the only current path with a potential >1.2x head-share ceiling.
  **Measured 1.2-1.3x at B=1; no >1.2x server result at B>=8.**

### Stage 5 — release

- Multi-seed B1-B4, manifests, negative results, reproduction commands.
- Decide: strict-first product, calibrated-first product, or both.

## 10. The one-sentence strategy

**Win at the server level by owning the contract between grammar state, candidate
source, execution plan, and the tau/latency operating point — not by claiming a
head speedup that the measured step time does not support.**
