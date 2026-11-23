from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")

from draftslice.integrations import vllm as integration  # noqa: E402


def test_plan_mode_defaults_off(monkeypatch) -> None:
    from draftslice.integrations.vllm import _plan_enabled

    monkeypatch.delenv("DRAFTSLICE_VLLM_PLAN", raising=False)
    assert not _plan_enabled()
    monkeypatch.setenv("DRAFTSLICE_VLLM_PLAN", "enforce")
    assert _plan_enabled()
    monkeypatch.setenv("DRAFTSLICE_VLLM_PLAN", "off")
    assert not _plan_enabled()


def test_register_is_idempotent_and_registers_methods() -> None:
    integration.register()
    integration.register()
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        registered_speculative_methods,
    )

    methods = set(registered_speculative_methods())
    assert "draftslice_mtp" in methods
    assert "draftslice_eagle3" in methods
    assert integration.DraftSliceMTPConfig.base_method == "mtp"
    assert integration.DraftSliceEagle3Config.base_method == "eagle3"


def test_speculator_classes_are_mixins() -> None:
    integration.register()
    assert issubclass(integration.DraftSliceMTPSpeculator, integration.DraftSliceSpeculatorMixin)
    assert issubclass(
        integration.DraftSliceEagle3Speculator,
        integration.DraftSliceSpeculatorMixin,
    )


def test_full_decode_cudagraph_is_detected() -> None:
    from types import SimpleNamespace

    from vllm.config.compilation import CUDAGraphMode

    from draftslice.integrations.vllm import _decode_uses_full_cudagraph

    def config(mode):
        return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))

    assert _decode_uses_full_cudagraph(config(CUDAGraphMode.FULL))
    assert _decode_uses_full_cudagraph(config(CUDAGraphMode.FULL_DECODE_ONLY))
    assert _decode_uses_full_cudagraph(config(CUDAGraphMode.FULL_AND_PIECEWISE))
    assert not _decode_uses_full_cudagraph(config(CUDAGraphMode.PIECEWISE))
    assert not _decode_uses_full_cudagraph(config(CUDAGraphMode.NONE))
    assert not _decode_uses_full_cudagraph(SimpleNamespace())
    assert not _decode_uses_full_cudagraph(None)


def test_graph_mode_uses_static_packed_mask(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch
    from vllm.config.compilation import CUDAGraphMode

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self, config) -> None:
            self.vllm_config = config
            self.vocab_size = 16
            self.max_num_reqs = 4
            self.device = torch.device("cpu")

        def capture(self):
            return "captured"

        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return torch.zeros(hidden_states.shape[0], 16)

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    full = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    )
    speculator = Speculator(full)
    assert speculator._draftslice_graph_replay
    assert speculator._draftslice_slicing
    assert speculator._draftslice_graph_packed is not None
    assert speculator.capture() == "captured"

    hidden = torch.zeros(1, 8)
    speculator._draftslice_spec_step = 0
    logits = speculator.compute_draft_logits(hidden, 0)
    assert speculator._draftslice_stats.graph_mask_calls == 1
    assert speculator._draftslice_stats.last_contract["mode"] == "mask"
    assert speculator._draftslice_stats.last_contract["execution"]["kind"] == "full"
    assert logits.shape == (1, 16)

    speculator._draftslice_spec_step = 1
    speculator.compute_draft_logits(hidden, 1)
    assert speculator._draftslice_stats.graph_replay_calls == 1
    assert (
        speculator._draftslice_stats.last_contract["fallback_reason"]
        == "graph_later_step"
    )

    piecewise = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    )
    eager = Speculator(piecewise)
    assert not eager._draftslice_graph_replay
    assert eager._draftslice_slicing


def test_tp_head_detection_and_contract(monkeypatch) -> None:
    from types import SimpleNamespace

    import numpy as np
    import torch

    from draftslice.integrations import vllm as integration
    from draftslice.plan import TokenSet

    class Base:
        pass

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator.__new__(Speculator)
    weight = torch.randn(8, 4)
    head = SimpleNamespace(
        tp_size=2,
        tp_rank=1,
        org_vocab_size=16,
        num_embeddings_padded=16,
        num_embeddings_per_partition=8,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=8,
            org_vocab_end_index=16,
            added_vocab_start_index=16,
            added_vocab_end_index=16,
            num_org_elements_padded=8,
        ),
    )
    speculator.model = SimpleNamespace(lm_head=head, logits_processor=None)
    speculator.vocab_size = 16
    speculator.num_speculative_steps = 4
    speculator._lm_head_weight = lambda: weight
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    speculator._draftslice_idx_mapping_np = np.array([3])

    view = speculator._slice_head()
    assert view is not None
    _, shard = view
    assert (shard.tp_size, shard.tp_rank) == (2, 1)
    assert speculator._draftslice_stats.tp_size == 2
    assert speculator._draftslice_stats.tp_shard_rows == 8

    speculator._emit_contract(
        mode="strict",
        valid_sets=[TokenSet([9, 11])],
        logits_stats=SimpleNamespace(rows=1, candidate_total=2),
        vocab_size=16,
    )
    contract = speculator._draftslice_stats.last_contract
    assert contract is not None
    assert contract["serving"]["source"] == "draftslice_tp2"
    assert contract["execution"]["kind"] == "slice_gemm"
    assert contract["guarantee"]["scope"] == "mu_proj"


def test_custom_config_resolves_at_tp2() -> None:
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        get_custom_speculator_config_class,
        get_method_config_class,
    )

    integration.register()
    custom = get_custom_speculator_config_class("draftslice_mtp")
    assert custom is integration.DraftSliceMTPConfig
    resolved = get_method_config_class(
        SimpleNamespace(method="mtp", custom_method="draftslice_mtp")
    )
    assert resolved is custom
    assert resolved.base_method == "mtp"
    assert custom.speculator_class.endswith("DraftSliceMTPSpeculator")


def test_mask_allpos_contract_uses_full_execution() -> None:
    from draftslice.integrations import vllm as integration

    class Speculator(integration.DraftSliceSpeculatorMixin):
        pass

    speculator = Speculator.__new__(Speculator)
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    speculator._emit_contract(mode="mask_allpos", execution_kind="full")
    contract = speculator._draftslice_stats.last_contract
    assert contract is not None
    assert contract["mode"] == "mask_allpos"
    assert contract["execution"]["kind"] == "full"


def test_local_argmax_reduction_slices(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()
            self.use_local_argmax_reduction = True

        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return torch.arange(16, dtype=torch.float32).reshape(1, 16)

        def get_draft_top_tokens(self, hidden_states, spec_step_idx=0):
            return torch.tensor([999])

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    assert speculator._draftslice_slicing
    assert speculator._draftslice_local_argmax
    # No grammar state: the shared sliced path falls back to the full head,
    # and the override still returns greedy ids in the vLLM format.
    tokens = speculator.get_draft_top_tokens(torch.randn(1, 4))
    assert tokens.tolist() == [15]
    assert speculator._draftslice_stats.fallback_reasons["no_grammar_state"] == 1


def test_grammar_bridge_is_idempotent() -> None:
    from draftslice.integrations.vllm_grammar import install_grammar_bridge

    install_grammar_bridge()
    install_grammar_bridge()


def test_router_skips_free_form_batches(monkeypatch) -> None:
    from types import SimpleNamespace

    import numpy as np

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")
    monkeypatch.setenv("DRAFTSLICE_VLLM_ROUTER", "1")

    calls: list[str] = []

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()

        def propose(self, input_batch, *args, **kwargs):
            calls.append("propose")
            return "proposed"

        def sample_draft(self, *args, **kwargs):
            calls.append("sample")
            return "sampled"

        def compute_draft_logits(self, hidden_states, spec_step_idx):
            calls.append("logits")
            return "logits"

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    input_batch = SimpleNamespace(idx_mapping_np=np.array([0, 1]))
    assert speculator.propose(input_batch) == "proposed"
    assert speculator._draftslice_route_full
    assert speculator._draftslice_stats.routed_full_steps == 1
    assert speculator.sample_draft(None, None, None, None, None, None, None) == "sampled"
    assert speculator.compute_draft_logits(None, 0) == "logits"
    assert calls == ["propose", "sample", "logits"]


def test_batch_has_grammar_reads_shadow_entries(monkeypatch) -> None:
    from types import SimpleNamespace

    import numpy as np

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")
    monkeypatch.setenv("DRAFTSLICE_VLLM_ROUTER", "1")

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    speculator._draftslice_shadow = SimpleNamespace(entries={3: object()})
    assert speculator._batch_has_grammar(
        SimpleNamespace(idx_mapping_np=np.array([3, 4]))
    )
    assert not speculator._batch_has_grammar(
        SimpleNamespace(idx_mapping_np=np.array([4, 5]))
    )


def test_batch_has_grammar_ignores_empty_snapshot(monkeypatch) -> None:
    from types import SimpleNamespace

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    speculator._draftslice_shadow = None
    batch = SimpleNamespace(idx_mapping_np=None, req_ids=["a", "b"])
    monkeypatch.setattr(
        integration,
        "current_grammar_snapshot",
        lambda: SimpleNamespace(structured_req_ids=()),
    )
    assert not speculator._batch_has_grammar(batch)
    monkeypatch.setattr(
        integration,
        "current_grammar_snapshot",
        lambda: SimpleNamespace(structured_req_ids=("b",)),
    )
    assert speculator._batch_has_grammar(batch)


def test_block_allpos_masks_each_position_and_forks_greedy(monkeypatch) -> None:
    torch = pytest.importorskip("torch")


    from draftslice.integrations import vllm as integration

    class FakeShadow:
        entries = {0: object()}

        def __init__(self) -> None:
            self.forks: list[tuple[int, int]] = []

        def has(self, state: int) -> bool:
            return state in self.entries

        def begin_block(self, state):
            return {"pos": 0, "state": state}

        def block_mask(self, work, *, vocab_size=None, device=None):
            allowed = {
                0: [0, 3],
                1: [1, 4],
            }[work["pos"]]
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            mask[allowed] = True
            return mask

        def block_advance(self, work, token):
            self.forks.append((work["pos"], token))
            work["pos"] += 1
            return True

    class Speculator(integration.DraftSliceSpeculatorMixin):
        pass

    speculator = Speculator.__new__(Speculator)
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    shadow = FakeShadow()
    speculator._draftslice_shadow = shadow
    speculator._draftslice_idx_mapping = torch.tensor([0, 0])
    speculator._draftslice_draft_step = torch.tensor([0, 1])
    speculator.num_speculative_steps = 2
    speculator.temperature = torch.tensor([0.0])

    logits = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )
    out = speculator._compute_block_allpos(
        torch.zeros(2, 2), 0, lambda hidden, step: logits.clone()
    )
    assert out is not None
    # Position 0: only ids 0 and 3 survive; 3 is the argmax.
    assert torch.isinf(out[0, 1]) and out[0, 1] < 0
    assert torch.isinf(out[0, 4]) and out[0, 4] < 0
    assert float(out[0, 3]) == 3.0
    # Position 1: only ids 1 and 4 survive; 1 has the higher value.
    assert torch.isinf(out[1, 0]) and out[1, 0] < 0
    assert torch.isinf(out[1, 3]) and out[1, 3] < 0
    assert float(out[1, 1]) == 4.0
    assert shadow.forks == [(0, 3), (1, 1)]
    assert speculator._draftslice_stats.block_allpos_rows == 2


def test_block_allpos_respects_batch_gate(monkeypatch) -> None:
    torch = pytest.importorskip("torch")


    from draftslice.integrations import vllm as integration

    monkeypatch.delenv("DRAFTSLICE_VLLM_BLOCK_ALLPOS_MAX_BATCH", raising=False)

    class FakeShadow:
        entries = {0: object()}

        def has(self, state: int) -> bool:
            return True

        def begin_block(self, state):
            return object()

    class Speculator(integration.DraftSliceSpeculatorMixin):
        pass

    speculator = Speculator.__new__(Speculator)
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    speculator._draftslice_shadow = FakeShadow()
    speculator._draftslice_idx_mapping = torch.tensor([0, 0, 0, 0])
    speculator._draftslice_draft_step = torch.tensor([0, 1, 0, 1])
    speculator.num_speculative_steps = 2
    speculator.temperature = torch.tensor([0.0])

    # Two requests is above the default gate of one; the caller must fall back
    # to the position-0 path rather than pay the all-position cost.
    assert (
        speculator._compute_block_allpos(
            torch.zeros(4, 2), 0, lambda hidden, step: torch.zeros(4, 6)
        )
        is None
    )
    assert speculator._draftslice_stats.block_allpos_fallbacks == 1


def test_policy_observe_hold_returns_builtin_logits(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace()

        def compute_draft_logits(self, hidden_states, spec_step_idx):
            return torch.arange(
                hidden_states.shape[0] * 16, dtype=torch.float32
            ).reshape(hidden_states.shape[0], 16)

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    speculator._draftslice_policy_hold = "mode_observe"
    logits = speculator.compute_draft_logits(torch.randn(2, 4), 0)
    # Byte-identical to the built-in path, with the observe reason recorded.
    assert logits.tolist() == Base().compute_draft_logits(torch.zeros(2, 4), 0).tolist()
    assert speculator._draftslice_stats.fallback_reasons["mode_observe"] == 1
    assert speculator._draftslice_stats.last_contract["mode"] == "full"


def test_policy_enforce_masks_only_hitting_rows(monkeypatch) -> None:
    from types import SimpleNamespace

    from draftslice.integrations import vllm as integration
    from draftslice.plan import TokenSet
    from draftslice.runtime.policy import ProfilePolicy
    from draftslice.runtime.profile import ProfitCell, build_table

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(
                    model="/models/Qwen3.5-4B",
                    dtype=SimpleNamespace(name="bfloat16"),
                    quantization=None,
                ),
                speculative_config=SimpleNamespace(method="draftslice_mtp"),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
            )

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    cell = ProfitCell(
        model="Qwen3.5-4B",
        engine="vllm",
        method="draftslice_mtp",
        workload="tool_call",
        batch_bucket="2-8",
        k=8,
        tp=1,
        quant="bf16",
        graph="eager",
        valid_bucket="65-512",
        action="strict_slice",
        expected_gain=1.24,
        measured_p99_ratio=1.02,
        manifest_id="b1",
        enabled=True,
    )
    policy = ProfilePolicy(
        build_table([cell], table_id="t"),
        profile_mode="enforce",
        engine="vllm",
        model="Qwen3.5-4B",
        method="draftslice_mtp",
    )
    speculator = Speculator()
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    speculator._draftslice_adapter_policy = integration.AdapterPolicy(
        policy=policy, profile_mode="enforce", table_hash="sha256:x"
    )
    speculator.num_speculative_steps = 8
    speculator._draftslice_row_workloads = ["tool_call", "free"]

    decision = speculator._decide_policy([TokenSet(range(200)), None])
    assert decision is not None and decision.executes_slice
    assert decision.row_mask == (True, False)
    assert speculator._draftslice_stats.cell_hit
    assert speculator._draftslice_stats.action == "slice"
    assert speculator._draftslice_stats.policy_slice_steps == 1


def test_plan_session_builds_vllm_step_plans(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from draftslice.integrations import vllm as integration
    from draftslice.plan import TokenSet
    from draftslice.runtime.policy import PolicyDecision
    from draftslice.runtime.reasons import FallbackReason

    monkeypatch.delenv("DRAFTSLICE_COST_CALIBRATION", raising=False)

    class Base:
        def __init__(self) -> None:
            self.vllm_config = SimpleNamespace(
                model_config=SimpleNamespace(
                    model="/models/Qwen3.5-4B",
                    dtype=SimpleNamespace(name="bfloat16"),
                    quantization=None,
                ),
                speculative_config=SimpleNamespace(method="draftslice_mtp"),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
            )

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    speculator = Speculator()
    speculator.vocab_size = 64
    speculator._draftslice_idx_mapping_np = [7, 8]
    speculator._draftslice_stats = integration.DraftSliceVllmStats()
    decision = PolicyDecision(
        action="slice",
        mode="strict",
        profile_mode="enforce",
        reason=FallbackReason.NO_CELL,
        row_mask=(True, False),
    )
    hidden = torch.zeros(2, 8)
    first = speculator._plan_step(
        [TokenSet([1, 2, 3]), TokenSet([4, 5])], decision, hidden
    )
    second = speculator._plan_step(
        [TokenSet([1, 2, 3]), TokenSet([4, 5])], decision, hidden
    )
    assert first is not None and second is not None
    assert first.sliced_rows == 1 and first.full_rows == 1
    assert second.plan_cache_hit
    speculator._record_policy(
        decision, rows=2, candidates=3, plan_result=second
    )
    stats = speculator.draftslice_stats()
    assert stats["plan_cache_hit"]
    assert stats["plan_cache_hits"] >= 1
    assert stats["candidate_cache_hit"]
    assert stats["scheduler_reason"] == "strict_slice"
