# Performance tests

These tests are skipped unless `DRAFTSLICE_RUN_PERF=1`. They require CUDA, torch,
Triton, and permission to change the GPU clock.

```bash
DRAFTSLICE_RUN_PERF=1 DRAFTSLICE_GPU=0 pytest tests/perf -q -s
```

The test:

- locks the selected GPU's SM clock to its max frequency with `nvidia-smi -lgc`;
- verifies the clock after locking;
- runs the full/gather/fused kernel matrix;
- verifies the clock did not move during the run;
- unlocks the clock on exit.

If the clock cannot be locked or verified, the test fails. A result without
`clock_fixed=true` is not a benchmark result.

Optional environment variables:

- `DRAFTSLICE_GPU`: GPU index (default `0`).
- `DRAFTSLICE_PERF_ITERATIONS`: timing iterations (default `100`).
