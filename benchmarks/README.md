# DraftSlice benchmarks

The benchmark design, metrics, and measurement protocols are defined in
[`docs/benchmark.md`](../docs/benchmark.md). This directory is for configs,
fixtures, and manifests; it never contains datasets or run outputs.

## Datasets

Datasets are referenced by logical names. The first measurement mix is:

- `eagle3-mix-v1`: UltraChat-200K (`train_sft`) + ShareGPT, used for EAGLE-3 /
  MTP head-quality and end-to-end runs.

A local data root is supplied at runtime, for example through
`DRAFTSLICE_DATA_ROOT`. Public configs and manifests must not contain absolute
dataset paths. They store the logical dataset name, version, split, and content
hash instead.

Public draft-model IDs used by the benchmark:

- `z-lab/Qwen3.5-4B-DFlash` (DFlash draft for `Qwen/Qwen3.5-4B`).

Local model roots are supplied through an operator environment variable such as
`DRAFTSLICE_MODEL_ROOT`; no absolute model path is stored in this repository.

## Status

The harness covers the conformance tests, the B1 metric definitions, an
offline B1 accuracy runner over cached candidate records, the B2 kernel runner,
the fair candidate-policy baseline matrix (including CFHead/FlashHead-style
reference policies), the B3 constraint replay, the B4 vLLM runner, the
screener factorial matrix, and release manifests/reports. The dataset adapters
and published multi-seed runs remain operator work.

## Release pipeline

A release is a set of result JSON files (any mix of `bench/1`,
`baseline_matrix/1`, `factorial/1`, `constraint_bench/1`, `kernel_bench/1`, and
`e2e/1`) plus a manifest:

```bash
# 1. Produce results, one process per method where needed.
draftslice bench kernel --gpu 0 --out results/kernel.json
draftslice bench factorial --input cases.pt --artifact artifacts/router \
  --out results/factorial.json
draftslice bench e2e --model "$DRAFTSLICE_MODEL_ROOT/Qwen3.5-4B" \
  --method mtp --out results/e2e_mtp.json
draftslice bench e2e --model "$DRAFTSLICE_MODEL_ROOT/Qwen3.5-4B" \
  --method draftslice_mtp --out results/e2e_draftslice_mtp.json

# 2. Render a hashed report (writes report.md and report.manifest.json).
draftslice bench report --name release-2026-11 --out runs/release/report.md \
  --input results/kernel.json --input results/factorial.json \
  --input results/e2e_mtp.json --input results/e2e_draftslice_mtp.json

# 3. Verify and compare against a previous release.
draftslice bench validate --manifest runs/release/report.manifest.json
draftslice bench compare --base runs/base/report.manifest.json \
  --candidate runs/release/report.manifest.json --out runs/release/compare.md
```

The nightly and release workflows run the engine-free version of this pipeline
(`benchmarks/smoke_pipeline.py`) so manifest/report regressions fail CI before a
GPU run starts.

## Fixed-frequency performance runs

Performance tests must run with a fixed SM clock. The test and CLI both lock the
clock with `nvidia-smi -lgc`, verify it, and unlock it on exit. If the lock
cannot be acquired, the run fails instead of reporting an unlocked number.

```bash
DRAFTSLICE_RUN_PERF=1 DRAFTSLICE_GPU=0 pytest tests/perf -q -s
draftslice bench kernel --gpu 0 --batch 1,8,32,64 \
  --rows 128,1024,4096,8192,16384 --out runs/kernel.json
```

The B2 runner measures eager and CUDA-graph replay for the full, naive-gather,
and fused slice paths. Use `--no-cuda-graph` to omit the replay rows and
`--kernels` to select implementations.
