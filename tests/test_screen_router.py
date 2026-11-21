from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from draftslice.screen import (  # noqa: E402
    ScreenError,
    group_index,
    group_mass_from_logits,
    group_mass_from_tokens,
    train_router,
)

GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7))


def test_group_index_marks_unassigned_tokens() -> None:
    index = group_index(((0, 2), (4,)), vocab_size=6)
    assert index.tolist() == [0, -1, 0, -1, 1, -1]


def test_group_mass_from_logits_sums_probability() -> None:
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mass = group_mass_from_logits(logits, GROUPS)
    assert mass.shape == (1, 4)
    assert mass[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert mass.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_group_mass_from_tokens_uses_weights() -> None:
    tokens = torch.tensor([[0, 2], [4, 6]])
    weights = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
    mass = group_mass_from_tokens(tokens, GROUPS, vocab_size=8, weights=weights)
    assert mass[0, 0].item() == pytest.approx(0.5)
    assert mass[0, 1].item() == pytest.approx(0.5)
    assert mass[1, 2].item() == pytest.approx(2 / 3)
    assert mass[1, 3].item() == pytest.approx(1 / 3)


def test_group_mass_from_tokens_handles_empty_rows() -> None:
    mass = group_mass_from_tokens(
        torch.tensor([[0, 1], [1, 1]]), ((0,), (2,)), vocab_size=4
    )
    # Row 1 has no assigned token; normalization leaves it at zero.
    assert mass[1].tolist() == [0.0, 0.0]


def test_train_router_learns_separable_groups() -> None:
    torch.manual_seed(0)
    n = 32
    hidden = torch.eye(4).repeat(n // 4, 1)
    targets = torch.eye(4).repeat(n // 4, 1)
    policy, report = train_router(
        hidden,
        targets,
        GROUPS,
        top_c=1,
        hidden_dim=16,
        epochs=120,
        learning_rate=0.05,
        batch_size=8,
        validation_fraction=0.25,
        seed=3,
    )
    assert report.final_loss < report.initial_loss
    assert report.validation_loss is not None
    assert report.validation_loss < report.initial_loss
    assert report.n_train + report.n_val == n
    candidates = policy.candidates(hidden)
    expected = [GROUPS[index] for index in targets.argmax(dim=-1).tolist()]
    assert [candidate.ids for candidate in candidates] == [
        tuple(sorted(group)) for group in expected
    ]


def test_train_router_is_seed_deterministic() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(16, 5)
    targets = torch.rand(16, 4)
    first, _ = train_router(
        hidden, targets, GROUPS, top_c=2, hidden_dim=8, epochs=10, seed=7
    )
    second, _ = train_router(
        hidden, targets, GROUPS, top_c=2, hidden_dim=8, epochs=10, seed=7
    )
    torch.testing.assert_close(first.w1, second.w1)
    torch.testing.assert_close(first.w2, second.w2)


def test_train_router_rejects_bad_shapes() -> None:
    with pytest.raises(ScreenError):
        train_router(torch.zeros(3, 3), torch.zeros(2, 4), GROUPS, top_c=1)
    with pytest.raises(ScreenError):
        train_router(torch.zeros(2, 3), torch.zeros(2, 3), GROUPS, top_c=1)
    with pytest.raises(ScreenError):
        train_router(torch.zeros(1, 3), torch.zeros(1, 4), GROUPS, top_c=1, epochs=0)
