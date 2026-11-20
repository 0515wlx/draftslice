# Contributing

DraftSlice is in a pre-alpha skeleton phase. Small, focused changes are easiest
to review.

## Development environment

```bash
conda create -n wlx_draftslice python=3.11 -y
conda activate wlx_draftslice
python -m pip install -e ".[dev]"
```

## Checks

```bash
ruff check .
pytest -q
```

## Pull requests

- Keep the target-verifier path exact; do not introduce lossy verification.
- Keep research code, data, and benchmark artifacts out of this repository.
- Add tests for new public interfaces.
- Describe the draft model family (EAGLE / DFlash / MTP) and the workload
  (schema / tool calling / free-form) affected by the change.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0.
