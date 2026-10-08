import pytest

from rrs.envs import four_room_reference
from rrs.experiments.four_room_model_audit import sample_model


def test_sampled_reward_and_successor_discounts_and_forced_first_action(monkeypatch):
    positions = ((4, 1), (3, 1), (6, 2), (9, 7))
    transitions = {
        (4, 1): ((3, 1), -2, False),
        (3, 1): ((6, 2), 1, False),
        (6, 2): ((9, 7), 1, True),
    }
    monkeypatch.setattr(four_room_reference, "step", lambda p, a, rng: transitions[p])
    pi, beta = ((1, 0, 0, 0),) * 4, (1, 0, 1, 1)
    result = sample_model(positions, pi, beta, (4, 1), 0, samples=4, gamma=0.5)
    assert result["reward_mean"] == -1.5
    assert result["successor_mean"] == [0, 0, 0.25]
    assert result["reward_se"] == 0 and result["successor_se"] == [0, 0, 0]
    assert result["longest_rollout"] == 2
    assert sample_model(positions, pi, beta, (4, 1), 0, samples=4, gamma=0.5) == result
    target = sample_model(positions, pi, beta, (6, 2), 0, samples=4)
    assert target["reward_mean"] == 1 and target["successor_mean"] == [0, 0, 0]
    terminal = sample_model(positions, pi, beta, (9, 7), 0, samples=4)
    assert terminal["reward_mean"] == 0 and terminal["longest_rollout"] == 0
    with pytest.raises(RuntimeError, match="max_steps"):
        sample_model(positions, pi, beta, (4, 1), 0, samples=4, max_steps=1)
