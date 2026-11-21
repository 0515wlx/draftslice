"""Registry for candidate sources.

The registry is the single place that maps a config string to a screener. Both
the plan compiler and the offline benchmark use it, so a policy cannot drift
between the two paths.

Built-ins cover the full head, static/VocabTrim, low-rank (SVD or trained),
group/cluster, MLP router, and the non-deployable oracle. CFHead/FlashHead/
VocabTrim are imported lazily from ``bench.official_baselines`` so importing the
core (or the registry) does not pull in torch or the reference artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .base import Screener
from .policies import (
    FullPolicy,
    GroupPolicy,
    LowRankPolicy,
    OracleGroupPolicy,
    RouterPolicy,
    StaticPolicy,
)

ScreenerFactory = Callable[..., Screener]
_REGISTRY: dict[str, ScreenerFactory] = {}


def register_screener(
    name: str,
    factory: ScreenerFactory,
    *,
    replace: bool = False,
) -> None:
    """Register ``factory`` under ``name``.

    ``replace=True`` is for tests and experiments; the default refuses to shadow
    a built-in silently.
    """

    if not name:
        raise ValueError("screener name must not be empty")
    if not callable(factory):
        raise TypeError("screener factory must be callable")
    if not replace and name in _REGISTRY:
        raise ValueError(f"screener {name!r} is already registered")
    _REGISTRY[name] = factory


def get_screener(name: str, **kwargs: Any) -> Screener:
    """Build the screener registered under ``name`` with ``kwargs``."""

    factory = _REGISTRY.get(name)
    if factory is None:
        available = ", ".join(available_screeners())
        raise KeyError(f"unknown screener {name!r}; available: {available}")
    return factory(**kwargs)


def available_screeners() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _lazy(attr: str) -> ScreenerFactory:
    """Call ``bench.official_baselines.<attr>(**kwargs)`` on first use."""

    def factory(**kwargs: Any) -> Screener:
        module = import_module("..bench.official_baselines", package=__package__)
        return getattr(module, attr)(**kwargs)

    return factory


register_screener("full", FullPolicy)
register_screener("static", StaticPolicy)
register_screener("group", GroupPolicy)
register_screener("router", RouterPolicy)
register_screener("lowrank", LowRankPolicy)
register_screener("oracle_group", OracleGroupPolicy)
register_screener("svd_lowrank", lambda **kw: LowRankPolicy.from_weight(**kw))
register_screener("trained_lowrank", lambda **kw: LowRankPolicy.from_trained(**kw))
register_screener("cfhead", _lazy("CFHeadPolicy"))
register_screener("flashhead", _lazy("FlashHeadPolicy"))
register_screener("vocabtrim", _lazy("make_vocabtrim_policy"))


__all__ = [
    "ScreenerFactory",
    "available_screeners",
    "get_screener",
    "register_screener",
]
