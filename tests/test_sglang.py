from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.integrations.sglang import (  # noqa: E402
    SglangIntegrationError,
    _plan_enabled,
    config_from_env,
    register,
    screen_logits,
)
from draftslice.plan import TokenSet  # noqa: E402
from draftslice.screen import StaticPolicy  # noqa: E402


def test_plan_mode_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("DRAFTSLICE_SGLANG_PLAN", raising=False)
    assert not _plan_enabled()
    monkeypatch.setenv("DRAFTSLICE_SGLANG_PLAN", "telemetry")
    assert _plan_enabled()
    monkeypatch.setenv("DRAFTSLICE_SGLANG_PLAN", "off")
    assert not _plan_enabled()


def test_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DRAFTSLICE_SGLANG", "0")
    monkeypatch.setenv("DRAFTSLICE_SGLANG_SCREENER", "static")
    monkeypatch.setenv("DRAFTSLICE_SGLANG_UNIVERSE", "vocab")
    monkeypatch.setenv("DRAFTSLICE_SGLANG_MAX_ROWS", "4096")
    config = config_from_env()
    assert not config.enabled
    assert config.screener == "static"
    assert config.max_rows == 4096
    monkeypatch.setenv("DRAFTSLICE_SGLANG_UNIVERSE", "bad")
    # The engine hook no longer uses a universe switch; the variable is
    # ignored rather than silently changing grammar semantics.
    assert config_from_env().max_rows == 4096
    monkeypatch.setenv("DRAFTSLICE_SGLANG_MAX_ROWS", "bad")
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


def test_grammar_candidates_replicate_topk_and_intersect() -> None:
    from draftslice.integrations import sglang as sg

    class Grammar:
        def __init__(self, allowed, terminated=False):
            self.allowed = set(allowed)
            self.terminated = terminated

        def is_terminated(self):
            return self.terminated

        def allocate_vocab_mask(self, vocab_size, batch_size, device):
            return torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            )

        def fill_vocab_mask(self, mask, idx):
            for token in self.allowed:
                mask[idx, token // 32] |= 1 << (token % 32)

    hidden = torch.zeros(2, 4)
    state = {
        "step": 0,
        "grammars": [Grammar({1, 2}), Grammar({3})],
        "hidden_states": hidden,
    }
    result = sg._grammar_candidates(
        state,
        rows=4,
        vocab_size=8,
        max_rows=64,
        device=torch.device("cpu"),
        screener=None,
    )
    # topk=2: each request's set is repeated for its branches.
    assert result == [
        TokenSet([1, 2]),
        TokenSet([1, 2]),
        TokenSet([3]),
        TokenSet([3]),
    ]

    result = sg._grammar_candidates(
        state,
        rows=4,
        vocab_size=8,
        max_rows=64,
        device=torch.device("cpu"),
        screener=StaticPolicy(rows=(1,)),
    )
    assert result == [TokenSet([1]), TokenSet([1]), None, None]


def test_grammar_candidates_reject_later_steps_and_oversize() -> None:
    from draftslice.integrations import sglang as sg

    class Grammar:
        def __init__(self, allowed):
            self.allowed = set(allowed)

        def is_terminated(self):
            return False

        def allocate_vocab_mask(self, vocab_size, batch_size, device):
            return torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            )

        def fill_vocab_mask(self, mask, idx):
            for token in self.allowed:
                mask[idx, token // 32] |= 1 << (token % 32)

    state = {
        "step": 1,
        "grammars": [Grammar({1})],
        "hidden_states": torch.zeros(1, 4),
    }
    assert (
        sg._grammar_candidates(
            state,
            rows=1,
            vocab_size=8,
            max_rows=64,
            device=torch.device("cpu"),
            screener=None,
            step=1,
        )
        is None
    )
    state["step"] = 0
    assert (
        sg._grammar_candidates(
            state,
            rows=1,
            vocab_size=8,
            max_rows=0,
            device=torch.device("cpu"),
            screener=None,
            step=0,
        )
        is None
    )


def test_tp_head_slices_local_rows_and_returns_local_logits(monkeypatch) -> None:
    from types import SimpleNamespace

    from draftslice.integrations import sglang as sg

    class Grammar:
        def is_terminated(self):
            return False

        def allocate_vocab_mask(self, vocab_size, batch_size, device):
            return torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            )

        def fill_vocab_mask(self, mask, idx):
            for token in (1, 3, 9, 11):
                mask[idx, token // 32] |= 1 << (token % 32)

    weight = torch.zeros(8, 4)
    monkeypatch.setattr(sg, "_sliceable_weight", lambda lm_head: weight)
    seen: dict[str, object] = {}

    def fake_build(hidden, w, candidates, *, vocab_size, full_logits, max_slice_rows):
        seen["candidates"] = candidates
        seen["vocab_size"] = vocab_size
        return torch.full((hidden.shape[0], vocab_size), -1.0), None

    monkeypatch.setattr(sg, "build_draft_logits", fake_build)
    monkeypatch.setattr(
        sg,
        "_ORIGINAL_LM_HEAD",
        lambda self, hidden, lm_head, bias=None, *, out=None: torch.ones(
            (hidden.shape[0], 8)
        ),
    )
    head = SimpleNamespace(
        weight=weight,
        num_embeddings=16,
        tp_size=2,
        org_vocab_size=16,
        num_embeddings_padded=16,
        num_embeddings_per_partition=8,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=8,
            org_vocab_end_index=16,
            padded_org_vocab_start_index=8,
            added_vocab_start_index=16,
            added_vocab_end_index=16,
            num_org_elements_padded=8,
        ),
    )
    hidden = torch.zeros(1, 4)
    state = {
        "step": 0,
        "grammars": [Grammar()],
        "hidden_states": hidden,
        "config": sg.SglangConfig(enabled=True),
    }
    token = sg._DRAFT_STATE.set(state)
    try:
        sg._patched_compute_lm_head(
            type("Processor", (), {"vocab_size": 16})(), hidden, head
        )
    finally:
        sg._DRAFT_STATE.reset(token)
    # Global ids 9/11 map to local rows 1/3 on rank 1; ids 1/3 are not owned.
    assert seen["candidates"] == [TokenSet([1, 3])]
    assert seen["vocab_size"] == 8
    assert sg._STATS["tp_size"] == 2


def test_step0_hook_uses_sliced_logits(monkeypatch) -> None:
    from draftslice.integrations import sglang as sg

    class Grammar:
        def is_terminated(self):
            return False

        def allocate_vocab_mask(self, vocab_size, batch_size, device):
            return torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            )

        def fill_vocab_mask(self, mask, idx):
            mask[idx, 1 // 32] |= 1 << (1 % 32)
            mask[idx, 3 // 32] |= 1 << (3 % 32)

    weight = torch.zeros(8, 4)
    monkeypatch.setattr(sg, "_sliceable_weight", lambda lm_head: weight)
    seen = {}

    def fake_build(hidden, w, candidates, *, vocab_size, full_logits, max_slice_rows):
        seen["candidates"] = candidates
        return torch.full((hidden.shape[0], vocab_size), -1.0), None

    monkeypatch.setattr(sg, "build_draft_logits", fake_build)
    monkeypatch.setattr(
        sg,
        "_ORIGINAL_LM_HEAD",
        lambda self, hidden, lm_head, bias=None, *, out=None: torch.ones(
            (hidden.shape[0], 8)
        ),
    )
    hidden = torch.zeros(2, 4)
    state = {
        "step": 0,
        "grammars": [Grammar(), Grammar()],
        "hidden_states": hidden,
        "config": sg.SglangConfig(enabled=True),
    }
    token = sg._DRAFT_STATE.set(state)
    try:
        logits = sg._patched_compute_lm_head(
            type("Processor", (), {"vocab_size": 8})(),
            hidden,
            type("Head", (), {"weight": weight})(),
        )
    finally:
        sg._DRAFT_STATE.reset(token)
    assert seen["candidates"] == [TokenSet([1, 3]), TokenSet([1, 3])]
    assert torch.equal(logits, torch.full((2, 8), -1.0))
    assert state["step"] == 1
    assert state["sliced_calls"] == 1


def test_step0_hook_falls_back_after_position_zero(monkeypatch) -> None:
    from draftslice.integrations import sglang as sg

    calls = {"full": 0}

    def original(self, hidden, lm_head, bias=None, *, out=None):
        calls["full"] += 1
        return torch.zeros((hidden.shape[0], 8))

    monkeypatch.setattr(sg, "_ORIGINAL_LM_HEAD", original)
    state = {"step": 1, "config": sg.SglangConfig(enabled=True), "grammars": []}
    token = sg._DRAFT_STATE.set(state)
    try:
        out = sg._patched_compute_lm_head(
            object(), torch.zeros(2, 4), object(), None
        )
    finally:
        sg._DRAFT_STATE.reset(token)
    assert calls["full"] == 1
    assert out.shape == (2, 8)


def test_step0_hook_real_kernel_masks_logits(monkeypatch) -> None:
    from draftslice.integrations import sglang as sg
    from draftslice.integrations import vllm_logits
    from draftslice.kernels.torch import TorchGatherKernel

    monkeypatch.setattr(vllm_logits, "_default_kernel", lambda: TorchGatherKernel())

    class Grammar:
        def is_terminated(self):
            return False

        def allocate_vocab_mask(self, vocab_size, batch_size, device):
            return torch.zeros(
                (batch_size, (vocab_size + 31) // 32), dtype=torch.int32
            )

        def fill_vocab_mask(self, mask, idx):
            for token in (1, 3):
                mask[idx, token // 32] |= 1 << (token % 32)

    torch.manual_seed(3)
    weight = torch.randn(8, 4)
    hidden = torch.randn(2, 4)
    monkeypatch.setattr(sg, "_sliceable_weight", lambda lm_head: weight)
    monkeypatch.setattr(
        sg,
        "_ORIGINAL_LM_HEAD",
        lambda self, h, lm_head, bias=None, *, out=None: h @ weight.T,
    )
    state = {
        "step": 0,
        "grammars": [Grammar(), Grammar()],
        "hidden_states": hidden,
        "config": sg.SglangConfig(enabled=True),
    }
    token = sg._DRAFT_STATE.set(state)
    try:
        logits = sg._patched_compute_lm_head(
            type("Processor", (), {"vocab_size": 8})(),
            hidden,
            type("Head", (), {"weight": weight})(),
        )
    finally:
        sg._DRAFT_STATE.reset(token)

    expected0 = torch.full((8,), float("-inf"))
    expected0[[1, 3]] = hidden[0] @ weight[[1, 3]].T
    expected1 = torch.full((8,), float("-inf"))
    expected1[[1, 3]] = hidden[1] @ weight[[1, 3]].T
    torch.testing.assert_close(logits[0], expected0)
    torch.testing.assert_close(logits[1], expected1)


def test_install_patches_the_engine_classes() -> None:
    pytest.importorskip("sglang")
    from draftslice.integrations import sglang as sg

    sg.install()
    sg.install()
    assert sg._PATCHED
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPDraftWorker

    assert EagleDraftWorker.draft is sg._patched_draft
    assert FrozenKVMTPDraftWorker.draft is sg._patched_draft
    assert LogitsProcessor._compute_lm_head is sg._patched_compute_lm_head
    assert DFlashWorkerV2.forward_batch_generation is sg._patched_dflash_generation


def test_dflash_hook_stays_full_and_records_the_negative(monkeypatch) -> None:
    from draftslice.integrations import sglang as sg

    monkeypatch.setenv("DRAFTSLICE_SGLANG", "1")
    calls = {"n": 0}

    def original(self, batch, *args, **kwargs):
        calls["n"] += 1
        return "full"

    monkeypatch.setattr(sg, "_ORIGINAL_DFLASH", original)
    before = sg._STATS["dflash_fallback_steps"]
    out = sg._patched_dflash_generation(object(), object())
    assert out == "full"
    assert calls["n"] == 1
    assert sg._STATS["dflash_fallback_steps"] == before + 1
    assert sg._STATS["dflash_reason"] == "dflash_no_exact_position"
    assert sg._STATS["action"] == "builtin"


def test_plan_step_builds_and_caches_slice_plans() -> None:
    from draftslice.integrations import sglang as sg
    from draftslice.runtime.policy import PolicyDecision
    from draftslice.runtime.reasons import FallbackReason

    decision = PolicyDecision(
        action="slice",
        mode="strict",
        profile_mode="enforce",
        reason=FallbackReason.NO_CELL,
        row_mask=(True,),
    )
    first = sg._plan_step(
        [TokenSet([1, 2, 3])], decision, rows=1, vocab=64, width=8
    )
    second = sg._plan_step(
        [TokenSet([1, 2, 3])], decision, rows=1, vocab=64, width=8
    )
    assert first is not None and second is not None
    assert first.plan_cache_miss
    assert second.plan_cache_hit
    assert sg._STATS["plan_cache_hits"] >= 1
    assert sg._STATS["plan_compile_us"] >= 0.0


def test_plugin_entry_point_is_installed() -> None:
    pytest.importorskip("sglang")
    from importlib.metadata import entry_points

    names = {
        entry.name
        for entry in entry_points(group="sglang.srt.plugins")
    }
    assert "draftslice" in names


def test_register_reports_missing_engine_surface() -> None:
    from draftslice.integrations import sglang as integration

    available, _ = integration._sglang_status()
    if available:
        pytest.skip("SGLang is fully installed in this environment")
    with pytest.raises(SglangIntegrationError):
        register()
