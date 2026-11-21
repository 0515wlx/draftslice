from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.sglang import (  # noqa: E402
    SglangIntegrationError,
    config_from_env,
    register,
    screen_logits,
)
from draftslice.plan import TokenSet  # noqa: E402
from draftslice.screen import StaticPolicy  # noqa: E402


def test_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTSLICE_SGLANG", "0")
    monkeypatch.setenv("DRAFTSLICE_SGLANG_SCREENER", "static")
    monkeypatch.setenv("DRAFTSLICE_SGLANG_UNIVERSE", "vocab")
    config = config_from_env()
    assert not config.enabled
    assert config.screener == "static"
    assert config.universe == "vocab"
    monkeypatch.setenv("DRAFTSLICE_SGLANG_UNIVERSE", "bad")
    with pytest.raises(SglangIntegrationError):
        config_from_env()


def test_screen_logits_intersects_valid_universe() -> None:
    logits = torch.randn(2, 8)
    hidden = torch.zeros(2, 4)
    policy = StaticPolicy(rows=(1, 3, 9))
    result = screen_logits(
        logits,
        hidden,
        [TokenSet([1, 2]), None],
        policy,
        universe="valid",
    )
    assert torch.equal(result[1], logits[1])
    assert result[0, 1].item() == logits[0, 1].item()
    assert result[0, 2].item() == float("-inf")
    assert result[0, 3].item() == float("-inf")
    assert torch.equal(result[1], logits[1])
    # The input tensor is not mutated.
    assert torch.isfinite(logits[0, 2]).item()


def test_screen_logits_freeform_keeps_candidates() -> None:
    logits = torch.randn(2, 8)
    hidden = torch.zeros(2, 4)
    policy = StaticPolicy(rows=(1, 3))
    result = screen_logits(logits, hidden, None, policy, universe="vocab")
    assert torch.isfinite(result[:, 1]).all()
    assert torch.isfinite(result[:, 3]).all()
    assert torch.isinf(result[:, 0]).all()


def test_screen_logits_rejects_bad_shapes() -> None:
    policy = StaticPolicy(rows=(1,))
    with pytest.raises(SglangIntegrationError):
        screen_logits(torch.randn(2, 8), torch.randn(3, 4), None, policy)
    with pytest.raises(SglangIntegrationError):
        screen_logits(torch.randn(2, 8), torch.randn(2, 4), None, policy, universe="bad")


def test_register_reports_missing_engine_surface() -> None:
    from draftslice.integrations import sglang as integration

    available, _ = integration._sglang_status()
    if available:
        pytest.skip("SGLang is fully installed in this environment")
    with pytest.raises(SglangIntegrationError):
        register()
