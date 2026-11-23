# Lean exact-candidate slice

Status: 2026-11-23. This note documents the opt-in lean hot path
(`DRAFTSLICE_VLLM_LEAN=1`), its gate semantics, and the measured production
A/B. Raw runs and the full accounting live in the gitignored `runs/` workspace
(`runs/P2_P4_LEAN_SLICE_20261123.md`).

Conclusion:

> On the production chat/tool request shape (Qwen3.5-4B, B=1, k=8, eager,
> TP=1, bf16 MTP) the lean exact-candidate slice is **+6.1%** over built-in
> MTP (127.03 → 134.76 tok/s, single server), with identical greedy outputs.
> It is an opt-in throughput mode, not a new default, and it drops per-step
> `DraftContract` emission from the hot path.

## 1. Why a lean path

The draft LM head is **16.2%** of a production verify step at k=8
(0.744 ms/call, 10.5 head calls/step). The head is already bandwidth-optimal:
0.744 ms for a 1.27 GB read is ~1.72 TB/s, i.e. A100 HBM peak, so the only
head-side win is reading fewer rows.

Exact grammar candidates make that possible: 63% of draft calls have a valid
candidate set in the reference tool workload, and the sliced kernel costs
65 µs for 128 rows instead of 744 µs. The theoretical head saving is
`10.5 × 0.63 × (0.744 − 0.065) ≈ 4.5 ms/step`.

The original adapter captured none of it. It paid ~1.9 ms **per call** in
telemetry, shadow forking, per-call payload construction, `torch.full`, and a
blocking `.tolist()` after every draft position. That tax exceeded the 0.68 ms
per-call head saving, so the production A/B was ~1.00× (0.94× on a busy host).

## 2. Design

`DRAFTSLICE_VLLM_LEAN=1` changes only how candidate logits are produced:

1. **Cached device payload per candidate set.** `build_draft_logits_lean` keeps
   a padded int32 `[groups, width]` token matrix and row index on the GPU,
   keyed by the exact `(row, TokenSet identity, width, device)` mapping, with a
   2048-entry LRU. The grammar shadow already caches CPU `TokenSet`s by state,
   so the payload survives across steps and requests.
2. **Grouped rows.** Rows are bucketed by candidate width; identical grammar
   states across batch rows share one launch and one weight read. This is the
   plan layer's union benefit without compiling a `StepPlan`.
3. **Reused output buffer.** One `[rows, vocab]` fp32 buffer per shape, refilled
   with `-inf` per call. No per-call `torch.full`.
4. **One fused launch per group** through `run_rows_into`; full-head fallback
   rows use the stock head for those rows only.
5. **No hot-path telemetry.** Lean skips `DraftContract`/JSONL trace, policy
   recording, plan compile, screener, and cost-gate work. It still evaluates the
   profile cell once per verify step (`_decide_policy(record=False)`) when a
   table is configured, and applies its `row_mask`.
6. **Unchanged candidate generation.** The exact grammar shadow is untouched, so
   the candidate set and `tau` are identical to the non-lean slice.

The one-line control-path fix
`Execution(kind="full")` must not carry `rows` is kept globally; without it the
full-mask/control path raises.

## 3. Gate semantics

Two independent gates remain:

| gate | source | behaviour |
|---|---|---|
| profile cell | `DRAFTSLICE_PROFILE_TABLE` + `DRAFTSLICE_PROFILE_MODE=enforce` | resolved once per verify step; non-slice cells fall back to the stock head; only `action == "slice"` is supported by lean |
| cost model | `DRAFTSLICE_COST_CALIBRATION` | **calibrated** artifact predicts slice vs full per call group (with `unique_rows` and a per-call tax, default 150 µs) and may fall back; **uncalibrated** artifact disables slicing at startup |

The uncalibrated behaviour is deliberately artifact-level: the adapter sets
`_draftslice_slicing = False` during startup and records
`uncalibrated_cost`, so the stock speculator runs with no shadow/fork tax. It
must not be "fixed" back to a per-call full head.

Lean requires eager, TP=1, bf16/unquantized MTP, no graph replay, no scream
screener/policy plan mode. All other configurations keep the previous path.

## 4. Measured production A/B

Single vLLM server on one idle GPU (the host had other jobs, so only one engine
at a time), `routing_bench` tool-only, B=1, k=8, 45 s, seed 7, eager, greedy.

| arm | tok/s (reps) | median | vs built-in |
|---|---|---:|---:|
| built-in MTP | 127.8 / 127.7 / 128.6 / 127.1 / 127.0 / 125.6 / 125.4 / 124.8 / 125.6 | 127.03 | 1.000× |
| old non-lean slice | 119.2 / 120.1 / 120.9 / 120.3 / 119.9 / 120.0 | 120.02 | 0.945× |
| **lean exact slice** | 130.6 / 133.4 / 128.3 / 132.6 / 133.2 / 133.1 | 132.85 | **1.046×** |
| **lean exact slice (final)** | 134.9 / 133.8 / 134.8 | **134.76** | **1.061×** |

Correctness: greedy token ids equal to built-in on 6 tool cases × 3 reps,
validity 1.0, `tau` 7.33 (same as the non-lean slice), accept rate 0.916.

Formal accounting at k=8, B=1:

| quantity | value |
|---|---:|
| draft-head calls per verify step | 10.5 |
| full head per call | 0.744 ms |
| sliced head per call | 0.065 ms |
| exact-candidate coverage | 63% (1517/2408 calls) |
| no-valid-set fallback | 37% |
| built-in step | 47.4 ms |
| lean step | 57.7 ms |

The 16.2% theoretical head share becomes ~10.2% after 63% coverage, ~7.8%
realized after the per-call residual, and ~4.6% after paying the verification
cost of the higher `tau`: masking raises `tau` 5.04 → 7.33, and on this hybrid
GDN model the extra verified tokens cost ~3% of throughput. Roughly 3 points of
the head gain are spent on that side effect.

## 5. Limitations

- **Contract/telemetry.** Lean has no per-step `DraftContract`, trace, policy,
  plan, screener, or scheduler guarantees on the hot path. Those layers are not
  an engine contract (no vLLM/SGLang consumer); they are DraftSlice's own audit
  surface. A sampled (e.g. 1%) or counters-only path is the recommended
  compromise if production auditability is required.
- **Coverage.** 37% of draft positions have no exact candidate set (grammar
  termination / missing fork) and still read the full 1.27 GB head.
- **Masking coupling.** Exact candidates constrain the draft distribution and
  raise `tau`; a head speedup that does not change `tau` (e.g. int8 weights)
  would avoid the ~3% verify tax, but int8 alone lost 3.0% `tau` at 1.07%
  quantization error and is not productionized.
- **Configuration scope.** Lean is disabled for TP>1, quantized heads,
  CUDA-graph replay, DFlash/EAGLE3, and screener/plan modes; those keep the old
  path.
- **Prompt scope.** Only structured requests get candidates; free-form traffic
  falls back to the full head.
- **Batch.** B>1 is not measured; the lean builder launches per row group, not
  one batched multi-bucket kernel.
- **Per-position sync.** `sample_draft` still copies tokens to CPU after each
  draft position (~2.2 ms/step); removing it is future work.
- **Output-buffer aliasing.** One logits buffer per `(rows, vocab)` is reused.
  This is safe for greedy MTP because the sampler consumes it immediately, but
  paths that retain logits across calls must be excluded.
- **Measurement shape.** Do not benchmark with several servers on one host; that
  inflates the adapter tax and previously produced the 0.94× number.

## 6. Reproduction

```bash
# built-in
CUDA_VISIBLE_DEVICES=3 DRAFTSLICE_VLLM=0 \
  vllm serve Qwen/Qwen3.5-4B --served-model-name qwen --host 127.0.0.1 \
  --max-model-len 2048 --gpu-memory-utilization 0.72 --enforce-eager \
  --disable-log-requests --seed 0 --port 8030 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}'

# lean
CUDA_VISIBLE_DEVICES=3 DRAFTSLICE_VLLM=1 DRAFTSLICE_VLLM_SLICE=1 \
  DRAFTSLICE_VLLM_LEAN=1 vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --host 127.0.0.1 \
  --max-model-len 2048 --gpu-memory-utilization 0.72 --enforce-eager \
  --disable-log-requests --seed 0 --port 8031 \
  --speculative-config '{"method":"draftslice_mtp","num_speculative_tokens":8}'
```

Then run the tool-only `routing_bench` against `http://127.0.0.1:8031/v1`
three times, restarting the server between arms.

Unit tests: `tests/test_lean_logits.py` (payload reuse, output-buffer reuse,
exact match against the reference builder; no GPU required).
