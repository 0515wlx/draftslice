# DraftSlice design notes

Status: public, pre-alpha. This document describes the intended system. It
does not report benchmark results. The implementation plan is in
[architecture.md](architecture.md); the evaluation plan is in
[benchmark.md](benchmark.md).

## 1. Problem

Speculative decoding splits a decoding step into drafting and verification.
For EAGLE, DFlash, and MTP drafters, the draft cost is

```text
T_draft = T_backbone + T_head
```

For single-layer drafters the LM head can dominate the draft step. When the
output is constrained by a grammar or JSON schema, only a subset of the
vocabulary can be proposed at each grammar state. A conventional pipeline still
projects the hidden state to the full vocabulary and applies the mask
afterwards, paying for rows that can never be proposed.

## 2. Core idea

A grammar state maps to an exact valid-token set `S`. A **HeadPlan** is the
executable form of `S`:

```text
HeadPlan = (state_id, token_ids, row_layout, kernel, estimated_bytes)
```

DraftSlice executes only the rows in the plan. The target verifier remains
exact.

**Invariant:** for a constrained state, the plan contains every grammar-valid
token. No valid token is dropped. Free-form states fall back to a full head or
to an existing compressed draft head.

## 3. Components

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

- **Plan compiler**: grammar/schema -> FSM state -> valid ids -> HeadPlan.
- **Slice cache**: state -> plan; hot row lists and packed submatrices reuse
  across requests and decode steps.
- **Executor**: `slice_gemv` for small slices, `slice_gemm` for packed hot
  slices, full/compressed fallback for large valid sets.
- **Scheduler**: the plan cost is known before execution, so it can inform
  draft depth, tree width, verify width, and latency budgets.
- **Integrations**: vLLM and SGLang entry points (future milestones).

## 4. Scope

In scope:

- exact constrained row selection for the draft head (strict mode);
- optional calibrated candidate screening followed by exact rescoring, with the
  coverage target stated explicitly;
- caching plans and rows across requests/steps;
- draft-aware scheduling;
- composition with low-rank, subset, and sparse draft heads for free-form
  states.

Out of scope:

- target-head approximation;
- lossy speculative verification;
- replacing general draft-head compression;
- claiming exact valid-set coverage in calibrated mode. Calibrated mode reports
  its delta and empirical coverage; strict mode is the only mode that guarantees
  every valid token is present.

## 5. Correctness and distribution semantics

The target verifier accepts only tokens on a verified target path. DraftSlice
does not change that contract. In strict mode, the system guarantees **validity**
(only grammar-valid tokens are proposed) and preserves the exact valid-token set
used by the target. In calibrated mode it still proposes only valid tokens, but
the candidate set may omit valid tokens; the plan must report its coverage target
and empirical coverage. In both cases DraftSlice does not claim to sample the
grammar-conditional distribution `mu*`; local masking samples a projected
distribution. Papers and integrations must state which distribution is
intended.

## 6. Open questions

- Optimal slice layout for batch size 1 versus batched serving.
- Cache budget, eviction, and pinning policy for hot rows.
- Reusing plans on the target side for constrained verification.
- Scheduling head plans jointly with draft depth and tree width.
- Composition with low-rank and sparse draft heads.
