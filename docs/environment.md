# Development environments

The project has two engine integrations, and their upstream pins are mutually
exclusive. This is not a local packaging bug:

| Constraint | vLLM 0.34.0 | SGLang 0.5.23 |
|---|---|---|
| `torch` | `==2.14.0` | `==2.14.1` |
| `xgrammar` | `==0.2.8` | `==0.2.7` |
| `flashinfer-python` | `==0.7.3` | `==0.7.2` |
| `compressed-tensors` | `==0.17.0` | `==0.18.0` |
| `tokenizers` | `>=0.23.3` | `==0.23.2` |
| `humming-kernels[cu13]` | `==0.1.19` | `==0.1.12` |
| `numba` | `==0.65.0` | `==0.65.1` |

No single Python environment can satisfy both metadata sets, so `pip check`
would always report conflicts. DraftSlice therefore uses one main environment
and two sibling engine environments, all Python 3.11:

| Environment | Contents | Use |
|---|---|---|
| `wlx_draftslice` | `torch`, numpy, pytest, ruff, jsonschema; no engine | plan/kernel/cache work and the engine-free tests |
| `wlx_draftslice_vllm` | vLLM 0.34.0 stack | B4 and the vLLM adapter |
| `wlx_draftslice_sglang` | SGLang 0.5.23 stack | SGLang boundary/engine work |

DraftSlice imports both engines lazily, so the same checkout works in all
three. Tests that need an absent engine are skipped by `importorskip`.

## Create all environments

```bash
bash scripts/setup_envs.sh          # base + vllm + sglang
bash scripts/setup_envs.sh base     # main env only
bash scripts/setup_envs.sh vllm     # vLLM env only
bash scripts/setup_envs.sh sglang   # SGLang env only
```

The script creates the environments, installs the top-level pins, installs the
project in editable mode, and runs `pip check` in each.

## Manual install

```bash
# Main / engine-free
conda create -n wlx_draftslice python=3.11 -y
conda activate wlx_draftslice
python -m pip install -U pip
python -m pip install -r requirements-base.txt
python -m pip install -e ".[dev]"
python -m pip check

# vLLM
conda create -n wlx_draftslice_vllm python=3.11 -y
conda activate wlx_draftslice_vllm
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pip check

# SGLang
conda create -n wlx_draftslice_sglang python=3.11 -y
conda activate wlx_draftslice_sglang
python -m pip install -U pip
python -m pip install -r requirements-sglang.txt
python -m pip install -e ".[dev]"
python -m pip check
```

## Exact locks

The `requirements*.txt` files are the human-readable top-level pins. For a
byte-for-byte reproducible environment, install the generated locks instead:

```bash
python -m pip install -r requirements-base.lock.txt       # main env
python -m pip install -r requirements.lock.txt            # vLLM env
python -m pip install -r requirements-sglang.lock.txt     # SGLang env
```

Locks are produced from a verified environment:

```bash
conda run -n wlx_draftslice python -m pip freeze --exclude-editable \
  > requirements-base.lock.txt
conda run -n wlx_draftslice_vllm python -m pip freeze --exclude-editable \
  > requirements.lock.txt
conda run -n wlx_draftslice_sglang python -m pip freeze --exclude-editable \
  > requirements-sglang.lock.txt
```

Do not hand-edit a lock. Re-generate it from an environment that passes
`pip check`, the smoke imports, and the project tests.

## Verification gates

```bash
bash scripts/verify_env.sh
```

or manually, per environment:

```bash
conda run -n wlx_draftslice_vllm python -m pip check
conda run -n wlx_draftslice_vllm python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
conda run -n wlx_draftslice_vllm python -m pytest -q

conda run -n wlx_draftslice_sglang python -m pip check
conda run -n wlx_draftslice_sglang python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
conda run -n wlx_draftslice_sglang python -m pytest -q
```

The reference hardware is A100-80GB with driver 580 / CUDA 13.0. Wheels are
Linux x86_64; do not copy a lock to another CUDA major version without
re-resolving it.

## Upgrading an engine

Upgrade one environment at a time:

1. Change the engine version only in `requirements.txt` or
   `requirements-sglang.txt`.
2. Reinstall that environment from scratch (or `pip install --upgrade`).
3. Run `pip check`, the smoke imports, and the test suite.
4. Regenerate that environment's lock file.
5. Never add the other engine to the same environment.
