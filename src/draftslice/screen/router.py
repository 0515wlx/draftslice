"""Train the MLP group router from group marginals.

The router predicts a distribution over vocabulary groups from a hidden state;
the policy keeps the top ``C`` groups. Training targets are group marginals:
the probability mass each group carries under the target distribution. They can
be built from teacher logits or from observed token ids, which keeps the router
independent of a particular draft-head implementation.

This is a small reference trainer, not a claim about another paper's recipe.
Calibrate the resulting candidate set on a disjoint split before using
``delta`` in a plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..plan.types import TokenSet
from .base import ScreenError
from .policies import RouterPolicy


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ScreenError("torch is not installed") from exc
    return torch


def group_index(groups: tuple[tuple[int, ...], ...], *, vocab_size: int) -> Any:
    """Map every token id to its group id (``-1`` when unassigned)."""

    torch = _require_torch()
    if vocab_size <= 0:
        raise ScreenError("vocab_size must be positive")
    index = torch.full((vocab_size,), -1, dtype=torch.long)
    for group_id, group in enumerate(groups):
        if not group:
            raise ScreenError("groups must not be empty")
        ids = torch.tensor(list(group), dtype=torch.long)
        if int(ids.max()) >= vocab_size:
            raise ScreenError("group token is outside vocab_size")
        index[ids] = group_id
    return index


def group_mass_from_logits(
    logits: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    temperature: float = 1.0,
    normalize: bool = True,
) -> Any:
    """Sum softmax probability inside each group; shape ``[N, groups]``."""

    torch = _require_torch()
    if logits.ndim != 2:
        raise ScreenError("logits must be 2-D")
    if temperature <= 0:
        raise ScreenError("temperature must be positive")
    vocab_size = int(logits.shape[1])
    index = group_index(groups, vocab_size=vocab_size).to(logits.device)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return _accumulate(probs, index, len(groups), normalize=normalize)


def group_mass_from_tokens(
    tokens: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    vocab_size: int,
    weights: Any | None = None,
    normalize: bool = True,
) -> Any:
    """Accumulate observed token occurrences into group marginals.

    ``tokens`` is ``[N]`` or ``[N, P]``. With a 2-D tensor each row is one
    query and the columns are positions (for example a draft block). Rows with
    no assigned tokens produce an all-zero row unless ``normalize=False``.
    """

    torch = _require_torch()
    values = torch.as_tensor(tokens, dtype=torch.long)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ScreenError("tokens must be 1-D or 2-D")
    if values.numel() and (int(values.min()) < 0 or int(values.max()) >= vocab_size):
        raise ScreenError("token id is outside vocab_size")
    index = group_index(groups, vocab_size=vocab_size).to(values.device)
    mapped = index[values]
    batch, positions = mapped.shape
    if weights is None:
        mass = torch.zeros((batch, len(groups)), dtype=torch.float32, device=values.device)
        valid = mapped >= 0
        rows = torch.arange(batch, device=values.device)[:, None].expand_as(mapped)
        mass.index_put_(
            (rows[valid], mapped[valid]),
            torch.ones(int(valid.sum()), device=values.device),
            accumulate=True,
        )
    else:
        weight = torch.as_tensor(weights, dtype=torch.float32, device=values.device)
        if weight.ndim == 1:
            weight = weight[:, None]
        if weight.shape != (batch, positions):
            raise ScreenError("weights must have the same shape as tokens")
        mass = torch.zeros((batch, len(groups)), dtype=torch.float32, device=values.device)
        valid = mapped >= 0
        rows = torch.arange(batch, device=values.device)[:, None].expand_as(mapped)
        mass.index_put_(
            (rows[valid], mapped[valid]),
            weight[valid],
            accumulate=True,
        )
    if not normalize:
        return mass
    return _normalize(mass)


def _accumulate(probs: Any, index: Any, num_groups: int, *, normalize: bool) -> Any:
    torch = _require_torch()
    batch = int(probs.shape[0])
    valid = index >= 0
    mass = torch.zeros(
        (batch, num_groups), dtype=torch.float32, device=probs.device
    )
    count = int(valid.sum())
    if count:
        rows = torch.arange(batch, device=probs.device)[:, None].expand(batch, count)
        mass.index_put_(
            (rows, index[valid].unsqueeze(0).expand(batch, count)),
            probs[:, valid],
            accumulate=True,
        )
    return _normalize(mass) if normalize else mass


def _normalize(mass: Any) -> Any:
    torch = _require_torch()
    total = mass.sum(dim=-1, keepdim=True)
    safe = torch.where(total > 0, total, torch.ones_like(total))
    return mass / safe


@dataclass(frozen=True, slots=True)
class RouterTrainingReport:
    """Loss and split metadata for one router fit."""

    epochs: int
    n_train: int
    n_val: int
    num_groups: int
    hidden_dim: int
    initial_loss: float
    final_loss: float
    validation_loss: float | None
    zero_mass_rows: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "num_groups": self.num_groups,
            "hidden_dim": self.hidden_dim,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "validation_loss": self.validation_loss,
            "zero_mass_rows": self.zero_mass_rows,
            "seed": self.seed,
        }


def train_router(
    hidden: Any,
    group_mass: Any,
    groups: tuple[tuple[int, ...], ...],
    *,
    top_c: int,
    hidden_dim: int | None = None,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    validation_fraction: float = 0.2,
    weight_decay: float = 0.0,
    seed: int = 0,
    name: str = "mlp_router",
) -> tuple[RouterPolicy, RouterTrainingReport]:
    """Fit ``RouterPolicy`` weights against group marginals.

    The loss is soft-target cross entropy between the normalized targets and a
    softmax over group logits. ``hidden`` and ``group_mass`` must have the same
    number of rows; rows whose target mass is all zero are neutralized to a
    uniform target and counted in the report.
    """

    torch = _require_torch()
    if hidden.ndim != 2:
        raise ScreenError("hidden must be 2-D")
    if group_mass.ndim != 2:
        raise ScreenError("group_mass must be 2-D")
    if int(hidden.shape[0]) != int(group_mass.shape[0]):
        raise ScreenError("hidden and group_mass have different row counts")
    if int(group_mass.shape[1]) != len(groups):
        raise ScreenError("group_mass columns must match groups")
    if not groups:
        raise ScreenError("groups must not be empty")
    if top_c <= 0:
        raise ScreenError("top_c must be positive")
    if epochs <= 0:
        raise ScreenError("epochs must be positive")
    if batch_size <= 0:
        raise ScreenError("batch_size must be positive")
    if learning_rate <= 0:
        raise ScreenError("learning_rate must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ScreenError("validation_fraction must be in [0, 1)")

    n, dim = int(hidden.shape[0]), int(hidden.shape[1])
    num_groups = len(groups)
    width = int(hidden_dim or max(32, min(4 * dim, 4096)))
    device = hidden.device
    dtype = hidden.dtype if hidden.dtype.is_floating_point else torch.float32

    targets = torch.as_tensor(group_mass, dtype=torch.float32, device=device).clamp_min(0.0)
    zero_rows = int((targets.sum(dim=-1) <= 0).sum())
    if zero_rows:
        targets = targets.clone()
        targets[targets.sum(dim=-1) <= 0] = 1.0
    targets = targets / targets.sum(dim=-1, keepdim=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    n_val = 0
    if validation_fraction > 0 and n >= 2:
        n_val = max(1, int(round(n * validation_fraction)))
        n_val = min(n_val, n - 1)
    val_index = perm[:n_val]
    train_index = perm[n_val:]

    features = hidden.detach().to(device=device, dtype=dtype)
    w1 = (
        torch.empty(width, dim, dtype=torch.float32)
        .uniform_(-1.0 / math.sqrt(dim), 1.0 / math.sqrt(dim), generator=generator)
        .to(device=device)
    )
    w2 = (
        torch.empty(num_groups, width, dtype=torch.float32)
        .uniform_(-1.0 / math.sqrt(width), 1.0 / math.sqrt(width), generator=generator)
        .to(device=device)
    )
    w1.requires_grad_(True)
    w2.requires_grad_(True)
    optimizer = torch.optim.Adam([w1, w2], lr=learning_rate, weight_decay=weight_decay)

    def loss_for(index: Any) -> Any:
        index = index.to(device)
        logits = torch.nn.functional.gelu(features[index].float() @ w1.T) @ w2.T
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(targets[index] * log_probs).sum(dim=-1).mean()

    initial_loss = float(loss_for(train_index).detach().cpu())
    for _ in range(epochs):
        order = torch.randperm(int(train_index.numel()), generator=generator)
        shuffled = train_index[order]
        for start in range(0, int(shuffled.numel()), batch_size):
            batch = shuffled[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batch)
            loss.backward()
            optimizer.step()

    final_loss = float(loss_for(train_index).detach().cpu())
    validation_loss = None if n_val == 0 else float(loss_for(val_index.to(device)).detach().cpu())
    policy = RouterPolicy(
        groups=tuple(tuple(group) for group in groups),
        w1=w1.detach(),
        w2=w2.detach(),
        top_c=top_c,
        name=name,
    )
    report = RouterTrainingReport(
        epochs=epochs,
        n_train=int(train_index.numel()),
        n_val=n_val,
        num_groups=num_groups,
        hidden_dim=width,
        initial_loss=initial_loss,
        final_loss=final_loss,
        validation_loss=validation_loss,
        zero_mass_rows=zero_rows,
        seed=int(seed),
    )
    return policy, report


def candidate_tokens(
    policy: RouterPolicy,
    hidden: Any,
) -> list[TokenSet]:
    """Convenience wrapper around ``policy.candidates`` for trainer callers."""

    return policy.candidates(hidden)


__all__ = [
    "RouterTrainingReport",
    "candidate_tokens",
    "group_index",
    "group_mass_from_logits",
    "group_mass_from_tokens",
    "train_router",
]
