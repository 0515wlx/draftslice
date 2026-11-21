from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")

from draftslice.integrations import vllm as integration  # noqa: E402


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


def test_graph_mode_construction_disables_slicing(monkeypatch) -> None:
    from types import SimpleNamespace

    from vllm.config.compilation import CUDAGraphMode

    from draftslice.integrations import vllm as integration

    monkeypatch.setenv("DRAFTSLICE_VLLM_SLICE", "1")

    class Base:
        def __init__(self, config) -> None:
            self.vllm_config = config

        def capture(self):
            return "captured"

    class Speculator(integration.DraftSliceSpeculatorMixin, Base):
        pass

    full = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    )
    speculator = Speculator(full)
    assert speculator._draftslice_graph_replay
    assert not speculator._draftslice_slicing
    assert speculator.capture() == "captured"

    piecewise = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    )
    eager = Speculator(piecewise)
    assert not eager._draftslice_graph_replay
    assert eager._draftslice_slicing


def test_grammar_bridge_is_idempotent() -> None:
    from draftslice.integrations.vllm_grammar import install_grammar_bridge

    install_grammar_bridge()
    install_grammar_bridge()
