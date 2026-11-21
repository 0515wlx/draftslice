#!/usr/bin/env bash
# Verify all three environments: pip metadata, smoke imports, and tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_ENV="${BASE_ENV:-wlx_draftslice}"
VLLM_ENV="${VLLM_ENV:-wlx_draftslice_vllm}"
SGLANG_ENV="${SGLANG_ENV:-wlx_draftslice_sglang}"

check_env() {
    local env_name="$1"
    local import_check="$2"
    echo "==> ${env_name}: pip check"
    conda run -n "${env_name}" python -m pip check
    echo "==> ${env_name}: smoke import"
    conda run -n "${env_name}" python -c "${import_check}"
    echo "==> ${env_name}: pytest"
    (cd "${ROOT}" && conda run -n "${env_name}" python -m pytest -q)
}

check_env "${BASE_ENV}" \
    "import torch, draftslice; from draftslice.plan.compiler import PlanCompiler; print('base torch', torch.__version__)"

check_env "${VLLM_ENV}" \
    "import torch, vllm; from draftslice.integrations.vllm import register; register(); import draftslice.integrations.vllm as m; assert hasattr(m, 'DraftSliceMTPConfig'); print('vllm', vllm.__version__, 'torch', torch.__version__)"

check_env "${SGLANG_ENV}" \
    "import torch, sglang; from draftslice.integrations.sglang import screen_logits; print('sglang', sglang.__version__, 'torch', torch.__version__)"

echo "all environments are consistent"
