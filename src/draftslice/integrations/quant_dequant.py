"""Row-wise dequantization for quantized draft LM heads.

A sliced draft head only ever needs a few weight rows. Dequantizing those rows
into bf16 lets the existing gather+dot kernels work with fp8/int8 heads without
materializing (or permanently owning) a dequantized copy of the whole head.

The module is engine-agnostic: it reads the scale parameters that vLLM/SGLang
register on the head module and never imports an engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_QUANT_DTYPES: frozenset[str] = frozenset(
    {"float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "int8", "uint8"}
)


def is_quantized_weight(weight: Any) -> bool:
    dtype = getattr(weight, "dtype", None)
    if torch is None or dtype is None:
        return False
    return str(dtype).split(".")[-1] in _QUANT_DTYPES


@dataclass(frozen=True, slots=True)
class QuantizedHead:
    """Metadata needed to dequantize a subset of rows of one LM head."""

    weight: Any
    out_features: int
    in_features: int
    transposed: bool
    scale: Any | None
    scale_inv: Any | None
    out_dtype: Any

    def describe(self) -> dict[str, Any]:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "transposed": self.transposed,
            "scale_shape": None
            if self.scale is None or not hasattr(self.scale, "shape")
            else tuple(self.scale.shape),
            "scale_inv_shape": None
            if self.scale_inv is None or not hasattr(self.scale_inv, "shape")
            else tuple(self.scale_inv.shape),
            "out_dtype": str(self.out_dtype),
        }


def describe_quantized_head(lm_head: Any, weight: Any) -> QuantizedHead | None:
    """Return dequant metadata, or ``None`` for an unsupported quant scheme."""

    if torch is None or not is_quantized_weight(weight):
        return None
    in_features = getattr(lm_head, "embedding_dim", None)
    if in_features is None:
        return None
    try:
        in_features = int(in_features)
        rows = int(weight.shape[0])
        cols = int(weight.shape[1])
    except (TypeError, ValueError):
        return None
    if rows == in_features and cols != in_features:
        transposed, out_features = True, cols
    elif cols == in_features and rows != in_features:
        transposed, out_features = False, rows
    else:
        return None
    scale = getattr(lm_head, "weight_scale", None)
    scale_inv = getattr(lm_head, "weight_scale_inv", None)
    if scale is None and scale_inv is None:
        return None
    return QuantizedHead(
        weight=weight,
        out_features=out_features,
        in_features=in_features,
        transposed=transposed,
        scale=scale,
        scale_inv=scale_inv,
        out_dtype=torch.bfloat16,
    )


def _scale_for_rows(scale: Any, rows: np.ndarray, out_features: int, device: Any) -> Any | None:
    if scale is None:
        return None
    value = scale.to(device=device, dtype=torch.float32) if hasattr(scale, "to") else scale
    if not hasattr(value, "numel"):
        return float(value)
    if value.numel() == 1:
        return value.reshape(())
    flat = value.reshape(-1)
    if flat.numel() == out_features:
        return flat[rows.tolist()].unsqueeze(-1)
    return None


def dequantize_rows(head: QuantizedHead, rows: list[int] | np.ndarray) -> Any:
    """Return ``[len(rows), in_features]`` bf16 weight rows.

    Returns ``None`` for scale layouts we cannot prove safe; the caller then
    fails closed to the full head with an exact grammar mask.
    """

    if torch is None or not rows:
        return None
    index = np.asarray(rows, dtype=np.int64)
    if np.any(index < 0) or np.any(index >= head.out_features):
        return None
    if head.transposed:
        values = head.weight[:, index].t()
    else:
        values = head.weight[index]
    values = values.to(torch.float32)
    scale = _scale_for_rows(head.scale, index, head.out_features, values.device)
    scale_inv = _scale_for_rows(head.scale_inv, index, head.out_features, values.device)
    if scale is None and scale_inv is None:
        return None
    if scale is not None:
        values = values * scale
    if scale_inv is not None:
        values = values / scale_inv
    return values.to(head.out_dtype)


__all__ = [
    "QuantizedHead",
    "dequantize_rows",
    "describe_quantized_head",
    "is_quantized_weight",
]
