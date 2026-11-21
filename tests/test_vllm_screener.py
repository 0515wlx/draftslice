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
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER_ARTIFACT", raising=False)
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER_UNIVERSE", raising=False)
    assert _screener_config() is None
    for value in ("", "0", "off", "none", "false"):
        monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", value)
        assert _screener_config() is None


def test_screener_config_parses_name_and_kwargs(monkeypatch) -> None:
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER_ARTIFACT", raising=False)
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER_UNIVERSE", raising=False)
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER", "static")
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_ARGS", '{"rows": [1, 3, 5]}')
    config = _screener_config()
    assert config == ScreenerConfig(name="static", args={"rows": [1, 3, 5]})


def test_screener_config_parses_artifact_and_universe(monkeypatch) -> None:
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER", raising=False)
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_ARTIFACT", "/tmp/bundle")
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_UNIVERSE", "vocab")
    config = _screener_config()
    assert config == ScreenerConfig(
        name="", artifact="/tmp/bundle", universe="vocab"
    )
    monkeypatch.setenv("DRAFTSLICE_VLLM_SCREENER_UNIVERSE", "bad")
    with pytest.raises(VllmIntegrationError):
        _screener_config()


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


def test_screen_valid_sets_freeform_uses_candidates() -> None:
    valid = [None, None, TokenSet([1, 2])]
    candidates = [TokenSet([7, 9]), TokenSet(), TokenSet([2, 3])]
    result = _screen_valid_sets(valid, candidates, universe="vocab")
    assert result.sets == [TokenSet([7, 9]), None, TokenSet([2])]
    assert result.freeform_rows == 1
    assert result.rows == 1
    assert result.dropped == 1


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


def test_apply_screener_freeform_universe() -> None:
    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = StaticPolicy(rows=(1, 3, 9))
    speculator._draftslice_screener_config = ScreenerConfig(
        name="static", universe="vocab"
    )
    speculator._draftslice_screener_ready = True

    hidden = torch.zeros((2, 4))
    result = speculator._apply_screener(hidden, [None, TokenSet([1, 2])])
    assert result == [TokenSet([1, 3, 9]), TokenSet([1])]
    stats = speculator._draftslice_stats
    assert stats.screener_universe == "vocab"
    assert stats.screener_freeform_rows == 1
    assert stats.screener_rows == 1


def test_apply_screener_loads_artifact_and_records_calibration(
    monkeypatch, tmp_path
) -> None:
    from draftslice.screen import build_screener_bundle

    bundle = build_screener_bundle(
        StaticPolicy(rows=(1, 3)),
        vocab_size=16,
        hidden_size=4,
        calibration={"delta": 0.05, "calibration_id": "cal-42"},
    )
    bundle.write(tmp_path / "artifact")

    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = None
    speculator._draftslice_screener_config = None
    speculator._draftslice_screener_ready = False
    monkeypatch.setenv(
        "DRAFTSLICE_VLLM_SCREENER_ARTIFACT", str(tmp_path / "artifact")
    )
    monkeypatch.delenv("DRAFTSLICE_VLLM_SCREENER", raising=False)

    result = speculator._apply_screener(
        torch.zeros((1, 4)), [TokenSet([1, 2, 3])]
    )
    assert result == [TokenSet([1, 3])]
    assert speculator._draftslice_screener_config is not None
    assert speculator._draftslice_screener_config.name == "static"
    stats = speculator._draftslice_stats
    assert stats.screener_calibration_id == "cal-42"
    assert stats.screener_delta == 0.05


def test_compute_draft_logits_freeform_screener(monkeypatch) -> None:
    from types import SimpleNamespace

    from draftslice.integrations import vllm_logits
    from draftslice.kernels.torch import TorchGatherKernel

    monkeypatch.setattr(vllm_logits, "_default_kernel", lambda: TorchGatherKernel())

    class FakeBase:
        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return hidden_states @ self.weight.T

    class Speculator(DraftSliceSpeculatorMixin, FakeBase):
        pass

    vocab, dim, batch = 16, 4, 2
    torch.manual_seed(1)
    weight = torch.randn(vocab, dim)
    hidden = torch.randn(batch, dim)
    speculator = Speculator.__new__(Speculator)
    speculator.weight = weight
    speculator.vocab_size = vocab
    speculator.model = SimpleNamespace(
        lm_head=SimpleNamespace(tp_size=1, quant_config=None, weight=weight),
        logits_processor=SimpleNamespace(
            logits_as_input=False, scale=1.0, soft_cap=None
        ),
    )
    speculator._lm_head_weight = lambda: weight
    speculator._draftslice_enabled = True
    speculator._draftslice_slicing = True
    speculator._draftslice_graph_replay = False
    speculator._draftslice_dummy = False
    speculator._draftslice_snapshot = None
    speculator._draftslice_shadow = None
    speculator._draftslice_idx_mapping = None
    speculator._draftslice_idx_mapping_np = None
    speculator._draftslice_spec_step = 0
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = StaticPolicy(rows=(1, 3))
    speculator._draftslice_screener_config = ScreenerConfig(
        name="static", universe="vocab"
    )
    speculator._draftslice_screener_ready = True

    logits = speculator.compute_draft_logits(hidden, 0)
    expected = torch.full((batch, vocab), float("-inf"))
    expected[:, [1, 3]] = hidden @ weight[[1, 3]].T
    torch.testing.assert_close(logits, expected)
    assert speculator._draftslice_stats.screener_freeform_rows == batch


def test_block_valid_sets_only_slice_position_zero() -> None:
    class Shadow:
        entries = {0: object()}

        def has(self, state):
            return state in self.entries

        def valid_ids(self, state, *, step=0):
            return TokenSet([1, 2]) if step == 0 else None

    class Bare(DraftSliceSpeculatorMixin):
        pass

    speculator = Bare.__new__(Bare)
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_shadow = Shadow()
    speculator._draftslice_idx_mapping = torch.tensor([0, 0, 1, 1])
    speculator._draftslice_draft_step = torch.tensor([0, 1, 0, 1])

    result = speculator._block_valid_sets(torch.zeros((4, 4)))
    assert result == [TokenSet([1, 2]), None, None, None]
    assert speculator._draftslice_stats.block_step0_rows == 1

    speculator._draftslice_draft_step = None
    assert speculator._block_valid_sets(torch.zeros((4, 4))) is None


def test_compute_draft_logits_block_mode_keeps_later_positions_full(monkeypatch) -> None:
    from types import SimpleNamespace

    from draftslice.integrations import vllm_logits
    from draftslice.kernels.torch import TorchGatherKernel

    monkeypatch.setattr(vllm_logits, "_default_kernel", lambda: TorchGatherKernel())

    class Shadow:
        entries = {0: object()}

        def has(self, state):
            return state in self.entries

        def valid_ids(self, state, *, step=0):
            return TokenSet([1, 3]) if step == 0 else None

    class FakeBase:
        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return hidden_states @ self.weight.T

    class Speculator(DraftSliceSpeculatorMixin, FakeBase):
        _draftslice_block = True

    vocab, dim = 16, 4
    torch.manual_seed(2)
    weight = torch.randn(vocab, dim)
    hidden = torch.randn(2, dim)
    speculator = Speculator.__new__(Speculator)
    speculator.weight = weight
    speculator.vocab_size = vocab
    speculator.model = SimpleNamespace(
        lm_head=SimpleNamespace(tp_size=1, quant_config=None, weight=weight),
        logits_processor=None,
    )
    speculator._lm_head_weight = lambda: weight
    speculator._draftslice_enabled = True
    speculator._draftslice_slicing = True
    speculator._draftslice_graph_replay = False
    speculator._draftslice_dummy = False
    speculator._draftslice_snapshot = None
    speculator._draftslice_shadow = Shadow()
    speculator._draftslice_idx_mapping = torch.tensor([0, 0])
    speculator._draftslice_idx_mapping_np = None
    speculator._draftslice_draft_step = torch.tensor([0, 1])
    speculator._draftslice_spec_step = 0
    speculator._draftslice_stats = DraftSliceVllmStats()
    speculator._draftslice_screener = None
    speculator._draftslice_screener_config = None
    speculator._draftslice_screener_ready = True

    logits = speculator.compute_draft_logits(hidden, 0)
    expected_first = torch.full((vocab,), float("-inf"))
    expected_first[[1, 3]] = hidden[0] @ weight[[1, 3]].T
    torch.testing.assert_close(logits[0], expected_first)
    torch.testing.assert_close(logits[1], hidden[1] @ weight.T)


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
