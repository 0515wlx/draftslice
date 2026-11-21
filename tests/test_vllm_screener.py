from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.vllm import (  # noqa: E402
    DraftSliceSpeculatorMixin,
    DraftSliceVllmStats,
    ScreenerConfig,
    VllmIntegrationError,
    _screen_valid_sets,
    _screener_config,
)
from draftslice.plan import TokenSet  # noqa: E402
from draftslice.screen import StaticPolicy  # noqa: E402


def test_screener_config_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER", raising=False)
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER_ARGS", raising=False)
    assert _screener_config() is None
    for value in ("", "0", "off", "none", "false"):
        monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", value)
        assert _screener_config() is None


def test_screener_config_parses_name_and_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", "static")
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_ARGS", '{"rows": [1, 3, 5]}')
    config = _screener_config()
    assert config == ScreenerConfig(name="static", args={"rows": [1, 3, 5]})


def test_screener_config_rejects_bad_json(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", "static")
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_ARGS", "[1, 2]")
    with pytest.raises(VllmIntegrationError):
        _screener_config()
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_ARGS", "{not json")
    with pytest.raises(VllmIntegrationError):
        _screener_config()


def test_screen_valid_sets_intersects_and_falls_back() -> None:
    valid = [TokenSet([1, 2, 3, 5]), None, TokenSet([7, 8]), TokenSet([4])]
    candidates = [
        TokenSet([1, 3, 9]),
        TokenSet([2]),
        TokenSet([1, 2]),
        TokenSet([4]),
    ]
    result = _screen_valid_sets(valid, candidates)
    assert result.sets[0] == TokenSet([1, 3])
    assert result.sets[1] is None
    # An empty intersection falls back to the full head, not to an empty plan.
    assert result.sets[2] is None
    assert result.sets[3] == TokenSet([4])
    assert result.rows == 3
    assert result.dropped == (len(valid[0]) - 2) + len(valid[2])
    assert result.empty_rows == 1


def test_screen_valid_sets_rejects_wrong_row_count() -> None:
    with pytest.raises(VllmIntegrationError):
        _screen_valid_sets([TokenSet([1])], [TokenSet([1]), TokenSet([2])])
    with pytest.raises(VllmIntegrationError):
        _screen_valid_sets([TokenSet([1])], [[1]])  # type: ignore[list-item]


def test_apply_screener_intersects_with_exact_sets() -> None:
    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = StaticPolicy(rows=(1, 3, 9))
    speculator._draftslice_screener_config = ScreenerConfig(name="static")
    speculator._draftslice_screener_ready = True

    hidden = torch.zeros((2, 4))
    valid = [TokenSet([1, 2, 3]), None]
    result = speculator._apply_screener(hidden, valid)

    assert result[0] == TokenSet([1, 3])
    assert result[1] is None
    stats = speculator._draftslice_stats
    assert stats.screener_name == "static"
    assert stats.screener_calls == 1
    assert stats.screener_rows == 1
    assert stats.screener_dropped == 1
    assert stats.screener_empty_rows == 0
    assert stats.screener_errors == 0


def test_apply_screener_failure_keeps_exact_sets() -> None:
    class Failing:
        name = "broken"

        def candidates(self, hidden_states):
            raise RuntimeError("boom")

    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = Failing()
    speculator._draftslice_screener_config = ScreenerConfig(name="broken")
    speculator._draftslice_screener_ready = True

    valid = [TokenSet([1, 2, 3])]
    result = speculator._apply_screener(torch.zeros((1, 4)), valid)

    assert result == valid
    stats = speculator._draftslice_stats
    assert stats.screener_calls == 0
    assert stats.screener_errors == 1
    assert "boom" in (stats.last_error or "")
    assert "screener_errors" in stats.to_dict()


def test_unknown_configured_screener_disables_the_hook(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", "does-not-exist")

    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = None
    speculator._draftslice_screener_config = None
    speculator._draftslice_screener_ready = False

    valid = [TokenSet([1, 2, 3])]
    assert speculator._apply_screener(torch.zeros((1, 4)), valid) == valid
    assert speculator._draftslice_stats.screener_errors == 0
    assert "does-not-exist" in (speculator._draftslice_stats.last_error or "")
