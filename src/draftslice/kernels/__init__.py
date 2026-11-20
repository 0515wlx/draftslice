"""Draft-head kernels.

Planned kernels:

- ``slice_gemv``: gather valid rows and compute a small GEMV.
- ``slice_gemm``: packed-submatrix GEMM for hot slices and batched drafts.
- ``full``: fallback for free-form or large-valid-set states.

No kernels are implemented in the pre-alpha skeleton.
"""

__all__: list[str] = []
