import pytest

from rrs.experiments.hallway import build_baselines
from rrs.experiments.model_audit import sample_model


def test_sampled_model_uses_actual_reward_and_successor_discount():
    data = build_baselines()
    option = data.options["shortest_path"]
    pi = tuple(tuple(float(a == action) for a in range(4)) for action in option.policy)
    result = sample_model(data.positions, pi, option.stopping, (3, 1), 0, samples=16)
    assert result["reward_mean"] == pytest.approx(
        -sum(0.99**t for t in range(4)), abs=1e-12
    )
    hallway = data.nonterminal_indices.index(data.positions.index((3, 7)))
    assert result["successor_mean"][hallway] == pytest.approx(0.99**6, abs=1e-12)
    assert result["longest_rollout"] == 6
    terminal = sample_model(data.positions, pi, option.stopping, (6, 10), 0, samples=16)
    assert terminal["reward_mean"] == 0.0
    assert terminal["successor_mean"] == [0.0] * 72
    assert terminal["longest_rollout"] == 0


def test_rollout_limit_reports_nontermination():
    data = build_baselines()
    pi = ((1.0, 0.0, 0.0, 0.0),) * 73
    with pytest.raises(RuntimeError, match="max_steps"):
        sample_model(
            data.positions, pi, (False,) * 73, (1, 1), 0, samples=2, max_steps=5
        )
