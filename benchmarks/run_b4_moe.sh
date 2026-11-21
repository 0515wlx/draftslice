#!/usr/bin/env bash
# B4 matrix for the MoE draft-slice thesis (Qwen3.5-35B-A3B).
#
# Baseline: full (no spec) vs mtp vs draftslice_mtp (safe pass-through).
# Run k=1 and k=4 to expose the per-draft-position head cost.
#
# TP=2 is the practical baseline (the 35B-A3B bf16 checkpoint is ~72 GB).
# DraftSlice slicing itself currently requires TP=1; see the TP=1 note below.
#
# Usage:
#   GPUS=4,5 TP=2 ./benchmarks/run_b4_moe.sh
#   GPUS=4 TP=1 GPU_MEM=0.99 MAX_LEN=512 ./benchmarks/run_b4_moe.sh slice
set -euo pipefail

ROOT="${DRAFTSLICE_MODEL_ROOT:-/mnt/data0601/model}"
MODEL="${MOE_MODEL:-$ROOT/Qwen3.5-35B-A3B}"
GPUS="${GPUS:-4,5}"
TP="${TP:-2}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_LEN="${MAX_LEN:-1024}"
BATCHES="${BATCHES:-1,8,32}"
REPS="${REPS:-2}"
CASES="${CASES:-tool_call,json_array,chat,math}"
OUT_PREFIX="${OUT_PREFIX:-runs/b4_moe_tp${TP}}"
MODE="${1:-all}"

mkdir -p "$(dirname "$OUT_PREFIX")"

# Fixed SM clock for every GPU in the matrix; always release on exit.
nvidia-smi -i "$GPUS" -lgc 1410 >/dev/null
trap 'nvidia-smi -i "$GPUS" -rgc >/dev/null 2>&1 || true' EXIT

run_case() {
  local k="$1"
  local out="${OUT_PREFIX}_k${k}.json"
  echo "=== spec_tokens=${k} -> ${out}"
  CUDA_VISIBLE_DEVICES="$GPUS" draftslice bench e2e \
    --model "$MODEL" \
    --methods full,mtp,draftslice_mtp \
    --cases "$CASES" \
    --batch "$BATCHES" \
    --spec-tokens "$k" \
    --max-tokens 64 \
    --repetitions "$REPS" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --out "$out"
}

if [[ "$MODE" == "all" || "$MODE" == "k1" ]]; then
  run_case 1
fi
if [[ "$MODE" == "all" || "$MODE" == "k4" ]]; then
  run_case 4
fi

# TP=1 slice attempt (requires a nearly empty 80 GB GPU):
#   GPUS=4 TP=1 GPU_MEM=0.99 MAX_LEN=512 REPS=1 ./benchmarks/run_b4_moe.sh slice
if [[ "$MODE" == "slice" ]]; then
  echo "=== TP=1 slice smoke; requires an empty 80 GB GPU"
  CUDA_VISIBLE_DEVICES="$GPUS" draftslice bench e2e \
    --model "$MODEL" \
    --methods mtp,draftslice_mtp \
    --cases tool_call,json_array \
    --batch 1,8 \
    --spec-tokens 1 \
    --max-tokens 32 \
    --repetitions 1 \
    --max-model-len 512 \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --out "${OUT_PREFIX}_slice_smoke.json"
fi
