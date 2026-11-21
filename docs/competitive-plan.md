# DraftSlice competitive plan

Status: internal planning document. It describes what is missing and how to get
to a defensible advantage over static, low-rank, tile/cluster screening, and
router-shortlist draft heads. It does not publish benchmark results.

Related documents:

- [architecture.md](architecture.md) — components and milestones.
- [benchmark.md](benchmark.md) — benchmark tiers, metrics, and protocols.

## 1. Executive summary

The current repository has a solid **plan core** (strict compilation, cache,
serialization, reference executor), a **cost/layout/scheduler foundation**, and
working **torch/Triton kernels** with numerical and fixed-frequency performance
tests. That is necessary, but not yet sufficient to claim an advantage.

The missing pieces are:

1. a real grammar backend and constraint-state replay;
2. a screener (router/static/low-rank) and calibrated mode;
3. union/tree execution for batch and draft trees;
4. engine-owned CUDA-graph buckets for the target runtime (the kernel-level graph
   cache exists);
5. vLLM integration hardening and end-to-end acceptance measurements;
6. screening-cost-aware latency baselines and a Pareto benchmark at iso-bytes and
   iso-latency.

The strategic opening is **not** "a faster kernel for the same candidate set".
It is:

> **Exact when constrained, calibrated when free-form, with one cost-aware plan
> and union/tree execution.**

Existing methods usually pick one regime: static/low-rank compress the head,
cluster/tile methods screen the vocabulary, and router methods pick a shortlist.
DraftSlice can own the **contract** between the grammar state, the candidate set,
the kernel, and the scheduler, and prove it on structured workloads where the
valid set changes every step.

## 2. What is already in place

| Area | Current state | Can we claim advantage? |
|---|---|---|
| Plan types and strict coverage | Implemented and tested | Not yet; needs real grammar states |
| Plan serialization and cache | Implemented | Not yet; cache not wired into compiler |
| Cost model | A100/H100 profiles, roofline | Not yet; not calibrated against B2 |
| Layout | native/grouped layout + packing | Not yet; not used by kernels |
| Torch/Triton kernels | full, naive gather, tuned fused GEMV/GEMM, CUDA graph | Partially: eager wins from ~`m>=1k`; graph replay clears the launch floor |
| Scheduler | deadline-aware strict/full | Not yet; no acceptance model |
| Screener / calibrated mode | Not implemented | No |
| Union/tree executor | Not implemented | No |
| Grammar backend | Fixture only | No |
| Engine integration | vLLM draft-step-0 plugin | Partially: strict structured smoke test only |
| End-to-end benchmark | Not implemented | No |
| Baselines | full/static/low-rank/group/router/CFHead/FlashHead/VocabTrim policies | Head quality yes; latency matrix incomplete |

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

The remaining kernel gaps are union/tree execution, runtime autotuning, and
engine-owned graph buckets, not the eager tiny-M floor.

### 3.2 No screener means no free-form advantage

Strict grammar slicing only helps when a constraint exists and the valid set is
small enough. Free-form states still use the full head. To compete with
SpecVocab, CFHead, FlashHead, DynaSpec, and SlimSpec on free-form workloads,
DraftSlice needs the calibrated path:

- vocabulary partition and leaf layout;
- router or static/low-rank screener;
- candidate-set cache;
- split-conformal calibration;
- exact rescoring inside the candidate set;
- `constrained_calibrated` and `freeform_calibrated` modes.

Without this, the project is only a grammar-specific optimization and cannot be
compared on the main free-form Pareto.

### 3.3 No union/tree execution means batch and tree leverage is unrealized

The current kernels execute one plan at a time. The architecture's main runtime
advantage is:

- deduplicate candidate rows across batch items;
- deduplicate rows across draft-tree nodes;
- run one union kernel and scatter to per-request or per-node outputs.

Until this exists, the measured speedups are per-plan and understate the batch
and tree opportunity. It also means the scheduler cannot trade tree width against
kernel cost.

### 3.4 No real grammar backend means no real plan distribution

The reference fixture backend is useful for tests, but it cannot answer:

- how large valid sets are for JSON Schema, tool calls, regex, SQL, and CFG;
- how often adjacent steps reuse the same grammar state;
- how much `PlanCache` helps;
- how often strict slicing falls back to full head;
- how much time plan compilation adds per step.

These are the core empirical questions for the strict product. The next step is
an xgrammar adapter and a replay format for real trajectories.

### 3.5 No end-to-end measurement means no acceptance claim

Head-only latency is not the product metric. The target verifier and backbone may
dominate decode time. We need:

- vLLM integration on the draft path;
- strict-mode output equality with full-head constrained decoding;
- acceptance length (`tau`), throughput, TTFT, ITL, and output validity;
- head-share decomposition so a `8x` head speedup is not reported as an `8x`
  end-to-end speedup.

Without B4, the project cannot claim a speculative-decoding win.

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

### 3.8 Missing artifact/calibration pipeline

There is no `ArtifactBundle` builder, no router training, no conformal
calibration, and no layout builder CLI. The runtime pieces exist, but the
deployable artifact does not. This blocks every calibrated experiment.

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
head and if plan compilation/caching overhead is small. That is why tiny-M
latency and state replay are P0.

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
the Pareto point per step is a real differentiator, provided the acceptance
model is measured rather than guessed.

### 4.4 Cache advantage: grammar state and candidate-set reuse

Two caches matter and are not the same:

- **grammar state -> plan** for strict slicing;
- **selected group set -> candidate rows** for screening.

Structured generation revisits grammar states and leaf sets. A cache hit removes
plan compilation and row-set construction from the critical path. This is a
systems advantage that head-compression papers typically do not model.

### 4.5 Kernel advantage: union/tree execution and tiny-M specialization

The kernel must win in three regimes:

1. **tiny M** (tens to hundreds of rows): launch-bound; needs a graph/persistent
   or specialized tiny kernel.
2. **medium M** (thousands): bandwidth/occupancy-bound; needs better tiling and
   vectorized loads.
3. **large M / batched tree**: union rows across batch/tree, tensor-core GEMM,
   one gather.

The current kernel wins only in the medium/large per-plan regime. The next
kernel must reduce the fixed overhead and add union/tree dispatch.

## 5. Prioritized plan

### P0 — Make the strict path real and fast (next)

1. **xgrammar adapter**
   - compile JSON Schema / regex / CFG to state -> valid token set;
   - record real state transitions;
   - add a replay format for B3.
2. **Plan cache integration**
   - compiler checks `PlanCache` before building a plan;
   - cache key includes artifact/tokenizer/constraint version;
   - report hit rate and compile p50/p90.
3. **Union/tree executor**
   - deduplicate rows across batch and tree nodes;
   - one kernel launch, scatter to per-node outputs;
   - numerical tests against per-plan execution.
4. **Tiny-M kernel path**
   - specialized kernel for `m <= 256/1024`;
   - CUDA graph capture for fixed shape buckets where possible;
   - target: `<=30 us` for `m<=1k`, `<=60 us` for `m=8k` at fixed A100 clock.
5. **vLLM strict integration**
   - draft-path hook, grammar state extraction, strict plan execution;
   - greedy output equality and 100% validity tests;
   - measure head share, tau, throughput, TTFT, ITL.

**Exit gate:** on UltraChat-200K + ShareGPT and structured tool-call workloads,
strict mode gives a measurable end-to-end speedup at B>=8 with 100% validity and
output equality, and the head share is reported.

### P1 — Build the calibrated screener and prove the Pareto

1. **Artifact builder**
   - leaf/static partition builder;
   - layout permutation;
   - `ArtifactBundle` manifest with hashes;
   - calibration set split by document/session.
2. **Router/static/low-rank screeners**
   - start with the leaf router and static prior;
   - add low-rank as a baseline, not the main path;
   - train on group marginals, not hard top-1 labels.
3. **Calibrated modes**
   - `constrained_calibrated` and `freeform_calibrated`;
   - split-conformal `C*`;
   - candidate-set cache;
   - exact rescoring inside the candidate set.
4. **Acceptance model and scheduler v2**
   - map plan coverage to expected accepted length;
   - choose mode/budget/tree shape under a latency target;
   - validate on held-out traces.
5. **B1 baseline matrix**
   - static, SVD, trained low-rank, CFHead-like, FlashHead-like,
     DynaSpec-like, CHESS, DraftSlice;
   - iso-bytes and iso-latency Pareto;
   - per-domain and on-policy results.

**Exit gate:** at `<=5%` of full-head bytes, DraftSlice matches or beats the best
baseline on top-1/top-5/OA on the primary mix, with coverage within the reported
interval.

### P2 — System-level advantage

1. **Union/tree kernel optimization**
   - tensor-core union GEMM;
   - tree scatter;
   - shape buckets and graph capture;
   - B=1/8/32/64 sweeps.
2. **SGLang integration**
3. **Quantization and multi-model support**
4. **Nightly B1/B2/B3 and release B4**
5. **Artifact/result registry and one-command reproduction**

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
- head share of decode time.

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
| End-to-end | >1.2x at B>=8 on structured; no free-form regression | If <1.1x at B>=8, head share/overhead is the bottleneck |
| Reproducibility | every table from a manifest and one command | No manifest, no claim |

## 8. Claim discipline

Avoid claims that the data does not support:

- "2.75% MACs" is not a latency claim; gather and launch overhead matter.
- "Bounded error" from low-rank is not a useful top-1 certificate on this head;
  the local analysis found the interval covers the full vocabulary.
- "Coverage" requires an independent calibration split and an empirical check.
- "Full head is optimal at high batch" is kernel-dependent; report the union
  efficiency.
- "Head speedup" is not "end-to-end speedup"; report the head share.

## 9. Recommended six-week sequence

### Week 1

- xgrammar adapter and B3 replay format.
- Plan cache integration and compile-time metrics.
- Tiny-M kernel prototype and clock-fixed benchmark.

### Week 2

- Union/tree executor with numerical tests.
- vLLM strict adapter and greedy/validity tests.
- Cost-model recalibration from B2.

### Week 3

- End-to-end strict run on structured workloads.
- Head-share and tau decomposition.
- Fix the largest overhead source (launch, gather, or plan compile).

### Week 4

- Artifact builder and router/static screener.
- Candidate-set cache and calibrated mode.
- Conformal calibration on a held-out split.

### Week 5

- B1 baseline matrix (static, low-rank, CFHead-like, FlashHead-like,
  DynaSpec-like, CHESS).
- Pareto selection at iso-bytes and iso-latency.

### Week 6

- B4 end-to-end calibrated run.
- Release report, negative results, reproduction commands.
- Decide: strict-first product, calibrated-first product, or both.

## 10. The one-sentence strategy

**Win on structured and calibrated workloads by owning the contract between
grammar state, candidate plan, exact rescoring, and cost-aware scheduling — not
by trying to out-compress low-rank heads on their own best case.**
