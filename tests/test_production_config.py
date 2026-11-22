from __future__ import annotations

import pytest

from draftslice.runtime.config import (
    AdapterConfig,
    ConfigError,
    EngineAdapterConfig,
    ObservabilityConfig,
    ProductionConfig,
    SafetyConfig,
    ServerConfig,
)


def test_production_default_is_builtin_observe() -> None:
    config = ProductionConfig.default()
    assert config.server.method == "builtin"
    assert config.server.profile_mode == "observe"
    assert config.server.profile_table == ""
    assert config.adapter.vllm.slice is False
    assert config.safety.fail_closed is True
    assert config.profile_path() is None
    config.validate()


def test_toml_load(tmp_path) -> None:
    path = tmp_path / "vllm_dense.toml"
    path.write_text(
        """
[server]
method = "draftslice_mtp"
profile_mode = "enforce"
profile_table = "profiles/4b.json"

[adapter.vllm]
slice = true
router = true
max_rows = 16384
full_mask = false
screener = ""

[safety]
tp = 2
quant = "bf16"
full_graph = "eager"
fail_closed = true

[observability]
trace_sample = 0.25
jsonl = "/var/log/draftslice/contracts.jsonl"
prometheus = "engine"
alarm_fallback_rate = 0.05
"""
    )
    config = ProductionConfig.load(path)
    assert config.server.profile_mode == "enforce"
    assert config.adapter.vllm.max_rows == 16384
    assert config.safety.tp == 2
    assert config.observability.trace_sample == 0.25
    assert config.profile_path(base=tmp_path) == tmp_path / "profiles/4b.json"


def test_legacy_flat_adapter_keys_apply_to_both_engines() -> None:
    config = ProductionConfig.from_dict(
        {"adapter": {"slice": True, "router": False, "max_rows": 512}}
    )
    assert config.adapter.vllm.slice is True
    assert config.adapter.sglang.slice is True
    assert config.adapter.vllm.router is False
    assert config.adapter.sglang.max_rows == 512


def test_env_overlay_precedence() -> None:
    config = ProductionConfig.default().with_env(
        {
            "DRAFTSLICE_PROFILE_TABLE": "profiles/4b.json",
            "DRAFTSLICE_PROFILE_MODE": "enforce",
            "DRAFTSLICE_VLLM_SLICE": "1",
            "DRAFTSLICE_VLLM_ROUTER": "0",
            "DRAFTSLICE_VLLM_MAX_ROWS": "4096",
            "DRAFTSLICE_TRACE_SAMPLE": "0.5",
            "DRAFTSLICE_JSONL": "/tmp/trace.jsonl",
        }
    )
    assert config.server.profile_mode == "enforce"
    assert config.server.profile_table == "profiles/4b.json"
    assert config.adapter.vllm.slice is True
    assert config.adapter.vllm.router is False
    assert config.adapter.vllm.max_rows == 4096
    assert config.observability.trace_sample == 0.5
    assert config.observability.jsonl == "/tmp/trace.jsonl"


def test_from_env_ignores_empty_values() -> None:
    config = ProductionConfig.from_env(
        {"DRAFTSLICE_PROFILE_MODE": "", "DRAFTSLICE_VLLM_SLICE": ""}
    )
    assert config.server.profile_mode == "observe"
    assert config.adapter.vllm.slice is False


def test_unsafe_settings_are_rejected() -> None:
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"server": {"profile_mode": "enforce"}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"safety": {"fail_closed": False}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"server": {"profile_mode": "on"}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"server": {"method": "speculative"}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"adapter": {"vllm": {"max_rows": 0}}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"safety": {"tp": 0}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"observability": {"trace_sample": 2.0}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"observability": {"prometheus": "otlp"}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"safety": {"full_graph": "always"}})


def test_unknown_keys_and_bad_toml_are_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"server": {"nope": 1}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"unknown": {}})
    with pytest.raises(ConfigError):
        ProductionConfig.from_dict({"schema": "config/2"})
    path = tmp_path / "bad.toml"
    path.write_text("[server\nmethod = 1")
    with pytest.raises(ConfigError):
        ProductionConfig.load(path)
    with pytest.raises(ConfigError):
        ProductionConfig.load(tmp_path / "missing.toml")
    with pytest.raises(ConfigError):
        ProductionConfig.default().with_env({"DRAFTSLICE_VLLM_SLICE": "maybe"})


def test_fingerprint_is_stable_and_sensitive() -> None:
    first = ProductionConfig.default()
    second = ProductionConfig.default()
    assert first.fingerprint().startswith("sha256:")
    assert first.fingerprint() == second.fingerprint()

    changed = ProductionConfig(
        server=ServerConfig(profile_table="profiles/4b.json"),
        adapter=AdapterConfig(vllm=EngineAdapterConfig(slice=True)),
        safety=SafetyConfig(),
        observability=ObservabilityConfig(),
    )
    assert changed.fingerprint() != first.fingerprint()
    assert changed.fingerprint() == changed.fingerprint()


def test_engine_selection() -> None:
    config = ProductionConfig.default()
    assert config.adapter.for_engine("vllm") is config.adapter.vllm
    assert config.adapter.for_engine("sglang") is config.adapter.sglang
    with pytest.raises(ConfigError):
        config.adapter.for_engine("trtllm")
