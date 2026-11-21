#!/usr/bin/env bash
# Create the three DraftSlice environments.
#
#   wlx_draftslice          main project env (no inference engine)
#   wlx_draftslice_vllm     vLLM 0.34.0
#   wlx_draftslice_sglang   SGLang 0.5.23
#
# vLLM and SGLang must stay separate: vllm 0.34 pins torch 2.14.0 and
# sglang 0.5.23 pins torch 2.14.1 (plus different xgrammar/flashinfer pins).
#
# Usage:
#   bash scripts/setup_envs.sh                 # all three
#   bash scripts/setup_envs.sh base            # main env only
#   bash scripts/setup_envs.sh vllm            # vLLM env only
#   bash scripts/setup_envs.sh sglang          # SGLang env only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
BASE_ENV="${BASE_ENV:-wlx_draftslice}"
VLLM_ENV="${VLLM_ENV:-wlx_draftslice_vllm}"
SGLANG_ENV="${SGLANG_ENV:-wlx_draftslice_sglang}"
TARGET="${1:-both}"

install_env() {
    local env_name="$1"
    local requirements="$2"
    echo "==> creating ${env_name} (python ${PYTHON_VERSION})"
    conda create -y -n "${env_name}" "python=${PYTHON_VERSION}"
    conda run -n "${env_name}" python -m pip install --upgrade pip setuptools wheel
    echo "==> installing ${requirements} into ${env_name}"
    conda run -n "${env_name}" python -m pip install -r "${ROOT}/${requirements}"
    echo "==> installing DraftSlice (editable) into ${env_name}"
    conda run -n "${env_name}" python -m pip install -e "${ROOT}[dev]"
    echo "==> pip check: ${env_name}"
    conda run -n "${env_name}" python -m pip check
}

case "${TARGET}" in
    both)
        install_env "${BASE_ENV}" "requirements-base.txt"
        install_env "${VLLM_ENV}" "requirements.txt"
        install_env "${SGLANG_ENV}" "requirements-sglang.txt"
        ;;
    base)
        install_env "${BASE_ENV}" "requirements-base.txt"
        ;;
    vllm)
        install_env "${VLLM_ENV}" "requirements.txt"
        ;;
    sglang)
        install_env "${SGLANG_ENV}" "requirements-sglang.txt"
        ;;
    *)
        echo "usage: $0 [both|base|vllm|sglang]" >&2
        exit 2
        ;;
esac

echo "done. Activate with: conda activate ${BASE_ENV} (or ${VLLM_ENV} / ${SGLANG_ENV})"
