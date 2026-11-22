"""Fused application of xgrammar bitmasks to draft logits.

The all-position DFlash path needs one grammar mask per block position. The
straightforward implementation unpacks each xgrammar bitmask into a ``[V]``
bool vector on the GPU, which costs one host-to-device upload and one allocation
per (request, position). This module batches the packed bitmasks into a single
``[requests, V/32]`` tensor and applies them with one Triton launch per position.

The kernel writes ``-inf`` only where the bitmask bit is zero; rows whose
matcher could not produce a mask are filled with all-ones upstream, so they stay
on the full head and no valid token is dropped.
"""

from __future__ import annotations

from typing import Any

_KERNEL: Any = None


def _mask_kernel() -> Any:
    """Define the Triton kernel once; redefining it per call defeats its cache."""

    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(
            LOGITS,
            PACKED,
            ROWS,
            V: tl.constexpr,
            WORDS: tl.constexpr,
            STRIDE: tl.constexpr,
            BASE: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            row = tl.program_id(0)
            blk = tl.program_id(1)
            token = blk * BLOCK + tl.arange(0, BLOCK)
            word = token // 32
            shift = token % 32
            packed_word = tl.load(
                PACKED + row * WORDS + word, mask=token < V, other=0
            )
            valid = ((packed_word >> shift) & 1) == 1
            req = tl.load(ROWS + row)
            tl.store(
                LOGITS + BASE + req * STRIDE + token,
                float("-inf"),
                mask=(token < V) & (~valid),
            )

        _KERNEL = (triton, kernel)
    return _KERNEL


def apply_packed_mask(
    logits: Any,
    *,
    base: int,
    row_stride: int,
    packed: Any,
    rows: Any,
    vocab_size: int,
    block: int = 2048,
) -> bool:
    """Mask strided logits rows with a packed ``[A, V/32]`` int32 bitmask.

    ``logits`` is the full ``[batch*steps, V]`` tensor, ``base`` is the element
    offset of the block position, and ``row_stride`` is the stride between
    request groups. ``rows`` holds the group indices that ``packed``'s rows
    correspond to. Returns ``True`` when the Triton path was used.
    """

    import torch

    # The pinned uploader returns a flat tensor; restore its 2-D bitmask shape.
    groups = int(rows.numel())
    words = (int(vocab_size) + 31) // 32
    packed = packed.reshape(groups, words)

    try:
        triton, kernel = _mask_kernel()
    except ImportError:  # pragma: no cover - torch-only environment
        triton, kernel = None, None

    if triton is not None and logits.device.type == "cuda":
        kernel[(groups, triton.cdiv(vocab_size, block))](
            logits,
            packed,
            rows,
            int(vocab_size),
            words,
            int(row_stride),
            int(base),
            int(block),
            num_warps=4,
        )
        return True

    # Portable fallback: unpack on the device and mask one row at a time.
    shifts = torch.arange(32, device=packed.device, dtype=torch.int32)
    bits = ((packed.unsqueeze(2) >> shifts) & 1).to(torch.bool).reshape(
        packed.shape[0], -1
    )[:, :vocab_size]
    view = logits.reshape(-1, row_stride // vocab_size, vocab_size)
    reqs = rows.tolist()
    for index, req in enumerate(reqs):
        view[int(req), base // vocab_size].masked_fill_(~bits[index], float("-inf"))
    return False


__all__ = ["apply_packed_mask"]
