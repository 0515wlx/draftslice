"""Production configuration (``configs/production/*.toml``).

Layout follows ``docs/production-architecture.md`` §5:

```toml
[server]
method = "builtin"          # built-in is the default, never draftslice
profile_mode = "observe"    # off | observe | enforce
profile_table = ""

[adapter.vllm]
slice = false               # only an enabled cell may turn this on
router = true
max_rows = 32768
full_mask = false
screener = ""

[safety]
tp = "auto"
quant = "auto"
full_graph = "step0"
fail_closed = true

[observability]
trace_sample = 0.01
jsonl = ""
prometheus = "engine"
alarm_fallback_rate = 0.05
```

Environment variables overlay the file so a deployment can point at a different
table without rewriting the artifact (``DRAFTSLICE_PROFILE_TABLE``,
``DRAFTSLICE_PROFILE_MODE``, ...). ``fingerprint()`` covers the resolved config
and is copied into every :class:`~draftslice.runtime.trace.StepTrace`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profile import GRAPHS, METHODS, QUANTS

SCHEMA = "config/1"

PROFILE_MODES = frozenset({"off", "observe", "enforce"})
PROMETHEUS_SINKS = frozenset({"engine", "none"})


class ConfigError(ValueError):
    """The production configuration is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    method: str = "builtin"
    profile_mode: str = "observe"
    profile_table: str = ""

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ConfigError(f"invalid server.method: {self.method!r}")
        if self.profile_mode not in PROFILE_MODES:
            raise ConfigError(f"invalid server.profile_mode: {self.profile_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "profile_mode": self.profile_mode,
            "profile_table": self.profile_table,
        }


@dataclass(frozen=True, slots=True)
class EngineAdapterConfig:
    slice: bool = False
    router: bool = True
    max_rows: int = 32768
    full_mask: bool = False
    screener: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.slice, bool):
            raise ConfigError("adapter.slice must be a bool")
        if not isinstance(self.router, bool):
            raise ConfigError("adapter.router must be a bool")
        if not isinstance(self.full_mask, bool):
            raise ConfigError("adapter.full_mask must be a bool")
        if self.max_rows <= 0:
            raise ConfigError("adapter.max_rows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice,
            "router": self.router,
            "max_rows": self.max_rows,
            "full_mask": self.full_mask,
            "screener": self.screener,
        }


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    vllm: EngineAdapterConfig = field(default_factory=EngineAdapterConfig)
    sglang: EngineAdapterConfig = field(default_factory=EngineAdapterConfig)

    def for_engine(self, engine: str) -> EngineAdapterConfig:
        if engine == "vllm":
            return self.vllm
        if engine == "sglang":
            return self.sglang
        raise ConfigError(f"unknown engine: {engine!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"vllm": self.vllm.to_dict(), "sglang": self.sglang.to_dict()}


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    tp: int | str = "auto"
    quant: str = "auto"
    full_graph: str = "step0"
    fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.tp != "auto":
            if isinstance(self.tp, bool) or not isinstance(self.tp, int) or self.tp < 1:
                raise ConfigError("safety.tp must be 'auto' or a positive int")
        if self.quant not in QUANTS:
            raise ConfigError(f"invalid safety.quant: {self.quant!r}")
        if self.full_graph not in GRAPHS:
            raise ConfigError(f"invalid safety.full_graph: {self.full_graph!r}")
        if not isinstance(self.fail_closed, bool):
            raise ConfigError("safety.fail_closed must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "quant": self.quant,
            "full_graph": self.full_graph,
            "fail_closed": self.fail_closed,
        }


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    trace_sample: float = 0.01
    jsonl: str = ""
    prometheus: str = "engine"
    alarm_fallback_rate: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.trace_sample <= 1.0):
            raise ConfigError("observability.trace_sample must be in [0, 1]")
        if not (0.0 <= self.alarm_fallback_rate <= 1.0):
            raise ConfigError("observability.alarm_fallback_rate must be in [0, 1]")
        if self.prometheus not in PROMETHEUS_SINKS:
            raise ConfigError(
                f"invalid observability.prometheus: {self.prometheus!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_sample": self.trace_sample,
            "jsonl": self.jsonl,
            "prometheus": self.prometheus,
            "alarm_fallback_rate": self.alarm_fallback_rate,
        }


@dataclass(frozen=True, slots=True)
class ProductionConfig:
    """Resolved production configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    # -- construction -----------------------------------------------------
    @classmethod
    def default(cls) -> ProductionConfig:
        """Built-in method, observe mode, no table: the safe production default."""

        return cls()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProductionConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("config must be a mapping")
        schema = data.get("schema")
        if schema is not None and schema != SCHEMA:
            raise ConfigError(f"unsupported config schema {schema!r}")
        unknown = set(data) - {
            "schema",
            "server",
            "adapter",
            "safety",
            "observability",
        }
        if unknown:
            raise ConfigError(f"unknown config sections: {sorted(unknown)}")
        server = _section(data, "server")
        adapter = _section(data, "adapter")
        safety = _section(data, "safety")
        observability = _section(data, "observability")
        _reject_unknown(server, {"method", "profile_mode", "profile_table"}, "server")
        _reject_unknown(
            adapter,
            {"vllm", "sglang", "slice", "router", "max_rows", "full_mask", "screener"},
            "adapter",
        )
        _reject_unknown(
            safety, {"tp", "quant", "full_graph", "fail_closed"}, "safety"
        )
        _reject_unknown(
            observability,
            {"trace_sample", "jsonl", "prometheus", "alarm_fallback_rate"},
            "observability",
        )
        legacy_adapter = {
            key: adapter[key]
            for key in ("slice", "router", "max_rows", "full_mask", "screener")
            if key in adapter
        }
        vllm = _engine_adapter(adapter.get("vllm"), legacy_adapter, "adapter.vllm")
        sglang = _engine_adapter(
            adapter.get("sglang"), legacy_adapter, "adapter.sglang"
        )
        config = cls(
            server=ServerConfig(**server),
            adapter=AdapterConfig(vllm=vllm, sglang=sglang),
            safety=SafetyConfig(**safety),
            observability=ObservabilityConfig(**observability),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> ProductionConfig:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config {source}: {exc}") from exc
        data = _parse_toml(text, source)
        return cls.from_dict(data)

    def with_env(self, environ: Mapping[str, str] | None = None) -> ProductionConfig:
        """Return a copy with the supported environment overrides applied."""

        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(name)
            return value if value is not None and value != "" else None

        server = self.server
        method = get("DRAFTSLICE_SERVER_METHOD")
        mode = get("DRAFTSLICE_PROFILE_MODE")
        table = get("DRAFTSLICE_PROFILE_TABLE")
        if method is not None or mode is not None or table is not None:
            server = ServerConfig(
                method=method if method is not None else server.method,
                profile_mode=mode if mode is not None else server.profile_mode,
                profile_table=table if table is not None else server.profile_table,
            )

        vllm = self.adapter.vllm
        v_slice = get("DRAFTSLICE_VLLM_SLICE")
        v_router = get("DRAFTSLICE_VLLM_ROUTER")
        v_mask = get("DRAFTSLICE_VLLM_FULL_MASK")
        v_rows = get("DRAFTSLICE_VLLM_MAX_ROWS")
        v_screener = get("DRAFTSLICE_VLLM_SCREENER")
        if any(
            value is not None
            for value in (v_slice, v_router, v_mask, v_rows, v_screener)
        ):
            vllm = EngineAdapterConfig(
                slice=_env_bool(v_slice) if v_slice is not None else vllm.slice,
                router=_env_bool(v_router) if v_router is not None else vllm.router,
                max_rows=int(v_rows) if v_rows is not None else vllm.max_rows,
                full_mask=_env_bool(v_mask) if v_mask is not None else vllm.full_mask,
                screener=v_screener if v_screener is not None else vllm.screener,
            )

        observability = self.observability
        trace_sample = get("DRAFTSLICE_TRACE_SAMPLE")
        jsonl = get("DRAFTSLICE_JSONL")
        if trace_sample is not None or jsonl is not None:
            observability = ObservabilityConfig(
                trace_sample=(
                    float(trace_sample)
                    if trace_sample is not None
                    else observability.trace_sample
                ),
                jsonl=jsonl if jsonl is not None else observability.jsonl,
                prometheus=observability.prometheus,
                alarm_fallback_rate=observability.alarm_fallback_rate,
            )

        config = ProductionConfig(
            server=server,
            adapter=AdapterConfig(vllm=vllm, sglang=self.adapter.sglang),
            safety=self.safety,
            observability=observability,
        )
        config.validate()
        return config

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProductionConfig:
        return cls.default().with_env(environ)

    # -- validation / identity -------------------------------------------
    def validate(self) -> None:
        """Reject unsafe combinations; called by every loader."""

        if self.server.profile_mode == "enforce" and not self.server.profile_table:
            raise ConfigError(
                "server.profile_mode='enforce' requires server.profile_table"
            )
        if not self.safety.fail_closed:
            raise ConfigError(
                "safety.fail_closed=false is not allowed: every uncertainty must "
                "fall back to the full head / built-in path"
            )

    def profile_path(self, *, base: str | Path | None = None) -> Path | None:
        """Resolve the profile table path relative to ``base`` or the cwd."""

        table = self.server.profile_table
        if not table:
            return None
        path = Path(table)
        if path.is_absolute():
            return path
        return (Path(base) if base is not None else Path.cwd()) / path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "server": self.server.to_dict(),
            "adapter": self.adapter.to_dict(),
            "safety": self.safety.to_dict(),
            "observability": self.observability.to_dict(),
        }

    def fingerprint(self) -> str:
        """``sha256:<hex>`` of the resolved configuration."""

        payload = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True, ensure_ascii=True
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a table")
    return dict(value)


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {name}: {sorted(unknown)}")


def _engine_adapter(
    value: Any, legacy: Mapping[str, Any], name: str
) -> EngineAdapterConfig:
    if value is None:
        data = dict(legacy)
    elif isinstance(value, Mapping):
        _reject_unknown(
            value,
            {"slice", "router", "max_rows", "full_mask", "screener"},
            name,
        )
        data = {**legacy, **dict(value)}
    else:
        raise ConfigError(f"{name} must be a table")
    return EngineAdapterConfig(**data)


def _parse_toml(text: str, source: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "TOML configs require Python 3.11+ or the 'tomli' package"
            ) from exc
    try:
        return tomllib.loads(text)
    except Exception as exc:  # tomllib raises TOMLDecodeError
        raise ConfigError(f"invalid TOML in {source}: {exc}") from exc


def _env_bool(value: str) -> bool:
    text = value.strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"invalid boolean value: {value!r}")


__all__ = [
    "AdapterConfig",
    "ConfigError",
    "EngineAdapterConfig",
    "ObservabilityConfig",
    "ProductionConfig",
    "PROFILE_MODES",
    "PROMETHEUS_SINKS",
    "SCHEMA",
    "SafetyConfig",
    "ServerConfig",
]
