from collections import deque

import jax
import jax.numpy as jnp
import pytest

from rrs.envs import reference
from rrs.experiments.hallway import build_baselines, option_rollout


@pytest.fixture(scope="module")
def baselines():
    return build_baselines()


def reference_rollout(data, name, start):
    """Independent trajectory and discounted-sum oracle using only Python dynamics."""
    option = data.options[name]
    indices = {position: i for i, position in enumerate(data.positions)}
    position = start
    route, rewards = [start], []
    if position == (6, 10):
        return route, rewards
    for _ in data.positions:
        action = option.policy[indices[position]]
        position, reward, done = reference.step(position, action)
        route.append(position)
        rewards.append(reward)
        if done or option.stopping[indices[position]]:
            return route, rewards
    pytest.fail(f"Nonterminating {name} rollout from {start}")


def test_feature_order_omits_goal_without_shifting_later_states(baselines):
    data = baselines
    assert len(data.nonterminal_indices) == 72
    assert len(data.features) == 73
    for feature, state in enumerate(data.nonterminal_indices):
        assert data.features[state] == tuple(float(i == feature) for i in range(72))
    assert data.features[data.positions.index((6, 10))] == (0.0,) * 72
    assert data.features[data.positions.index((6, 13))][-1] == 1.0


def test_start_values_routes_and_stopping_states(baselines):
    data = baselines
    start = data.positions.index((3, 1))
    rr = data.solutions["reward_respecting"]
    assert rr.values[start] == pytest.approx(0.99**11, abs=1e-12, rel=0)
    sp = data.solutions["shortest_path"]
    assert sp.values[start] == pytest.approx(
        -sum(0.99**t for t in range(6)), abs=1e-12, rel=0
    )
    rr_route, rr_rewards = reference_rollout(data, "reward_respecting", (3, 1))
    sp_route, sp_rewards = reference_rollout(data, "shortest_path", (3, 1))
    assert rr_route[-1] == sp_route[-1] == (3, 7)
    assert rr_rewards == [0.0] * 12
    assert sp_rewards == [-1.0] * 4 + [0.0] * 2
    assert {s for s, stop in zip(data.positions, rr.stopping, strict=True) if stop} == {
        (row, col) for row in range(1, 5) for col in (3, 4)
    } | {(3, 7), (6, 10)}
    assert {s for s, stop in zip(data.positions, sp.stopping, strict=True) if stop} == {
        (3, 7),
        (6, 10),
    }


@pytest.mark.parametrize("name", ["reward_respecting", "shortest_path"])
def test_subtask_values_and_optimality_from_every_state(baselines, name):
    data = baselines
    subtask, solution = data.subtasks[name], data.solutions[name]
    assert solution.bellman_residual <= 1e-12
    index = {p: i for i, p in enumerate(data.positions)}
    residuals = []
    for state, position in enumerate(data.positions):
        route, rewards = reference_rollout(data, name, position)
        if data.terminal[state]:
            assert solution.values[state] == 0.0
            assert solution.greedy_actions[state] == ()
            continue
        cumulants = rewards if name == "reward_respecting" else [-1.0] * len(rewards)
        final = index[route[-1]]
        value = sum(0.99**t * c for t, c in enumerate(cumulants))
        value += 0.99 ** (len(rewards) - 1) * subtask.stopping_values[final]
        assert value == pytest.approx(solution.values[state], abs=1e-12, rel=0)

        # Independently reconstruct every Bellman action using Python dynamics.
        action_values = []
        for action in range(4):
            next_position, reward, done = reference.step(position, action)
            next_index = index[next_position]
            c = reward if name == "reward_respecting" else -1.0
            continuation = 0.99 * solution.values[next_index]
            if subtask.can_stop[next_index]:
                continuation = max(subtask.stopping_values[next_index], continuation)
            action_values.append(c + (0.0 if done else continuation))
        residuals.append(abs(max(action_values) - solution.values[state]))
        assert solution.greedy_actions[state] == tuple(
            a
            for a, q in enumerate(action_values)
            if abs(q - max(action_values)) <= 1e-12
        )
        if name == "reward_respecting":
            # Eq. 9 and the Eq. 2 optimum agree for this fixed zero-weight experiment.
            assert solution.stopping[state] == (
                subtask.stopping_values[state] >= solution.values[state]
            )
    assert max(residuals) <= 1e-12


@pytest.mark.parametrize(
    "name", ["reward_respecting", "shortest_path", "up", "down", "left", "right"]
)
def test_exact_models_against_all_independent_rollouts(baselines, name):
    data = baselines
    exact = data.models[name]
    index = {p: i for i, p in enumerate(data.positions)}
    for state, position in enumerate(data.positions):
        route, rewards = reference_rollout(data, name, position)
        expected_reward = sum(0.99**t * r for t, r in enumerate(rewards))
        expected_features = tuple(
            0.99 ** len(rewards) * x for x in data.features[index[route[-1]]]
        )
        assert exact.rewards[state] == pytest.approx(expected_reward, abs=1e-12, rel=0)
        assert exact.successors[state] == pytest.approx(
            expected_features, abs=1e-12, rel=0
        )
        actual = option_rollout(data, data.options[name], position)
        assert list(actual.positions) == route
        assert list(actual.rewards) == rewards

    linear = exact.as_linear_model(data.nonterminal_indices)
    rewards, successors = jax.jit(jax.vmap(linear.predict))(jnp.array(data.features))
    assert rewards.tolist() == pytest.approx(exact.rewards, abs=1e-6, rel=0)
    assert jnp.max(jnp.abs(successors - jnp.array(exact.successors))) <= 1e-6


def test_shortest_path_against_breadth_first_search(baselines):
    data = baselines
    for position in data.positions:
        if position in ((3, 7), (6, 10)):
            continue
        queue = deque([(position, 0)])
        visited = {position}
        while queue:
            state, distance = queue.popleft()
            if state in ((3, 7), (6, 10)):
                break
            for action in range(4):
                successor = reference.step(state, action)[0]
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, distance + 1))
        _, rewards = reference_rollout(data, "shortest_path", position)
        assert len(rewards) == distance


def test_option_initiation_and_primitive_masking_at_the_goal(baselines):
    data = baselines
    for name in data.options:
        _, rewards = reference_rollout(data, name, (3, 7))
        assert len(rewards) >= 1
        route = option_rollout(data, data.options[name], (6, 10))
        assert route.positions == ((6, 10),)
        assert route.rewards == route.actions == ()
    start = data.positions.index((3, 1))
    hallway_feature = data.nonterminal_indices.index(data.positions.index((3, 7)))
    assert data.models["reward_respecting"].successors[start][
        hallway_feature
    ] == pytest.approx(0.99**12, abs=1e-12, rel=0)
    assert data.models["shortest_path"].successors[start][
        hallway_feature
    ] == pytest.approx(0.99**6, abs=1e-12, rel=0)
    assert data.models["reward_respecting"].rewards[start] == 0.0
    assert data.models["shortest_path"].rewards[start] == pytest.approx(
        -3.940399, abs=1e-12, rel=0
    )
