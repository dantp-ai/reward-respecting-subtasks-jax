from collections import deque

import pytest

from rrs.envs import reference, two_room
from rrs.experiments.two_room import build_model, greedy_route
from rrs.rl.exact import DeterministicModel, value_iteration


def test_delayed_reward_and_terminal_bootstrap():
    # State 0 reaches state 1 for zero reward, then terminates for reward 2.
    # The terminal successor deliberately points back to a positive-valued state.
    model = DeterministicModel(((1,), (0,)), ((0.0,), (2.0,)), ((False,), (True,)))
    result = value_iteration(model, gamma=0.5)
    assert result.values == (1.0, 2.0)
    assert result.greedy_actions == ((0,), (0,))
    assert result.bellman_residual == 0.0


def test_maximum_and_all_tied_actions():
    model = DeterministicModel(((0, 0, 0),), ((-1.0, 2.0, 2.0),), ((True, True, True),))
    result = value_iteration(model, gamma=0.99)
    assert result.values == (2.0,)
    assert result.greedy_actions == ((1, 2),)


def test_continuing_self_loop_and_reported_final_residual():
    model = DeterministicModel(((0,),), ((1.0,),), ((False,),))
    result = value_iteration(model, gamma=0.5, tolerance=1e-10)
    assert result.values[0] == pytest.approx(2.0, abs=2e-10)
    residual = abs(1.0 + 0.5 * result.values[0] - result.values[0])
    assert result.bellman_residual == residual
    assert residual <= 1e-10
    with pytest.raises(RuntimeError, match="converge"):
        value_iteration(model, gamma=0.5, max_iterations=1)


def test_zero_discount_uses_only_immediate_rewards():
    model = DeterministicModel(((1,), (1,)), ((-1.0,), (5.0,)), ((False,), (False,)))
    assert value_iteration(model, gamma=0.0).values == (-1.0, 5.0)


@pytest.mark.parametrize("gamma", [-0.1, 1.0, float("nan")])
def test_invalid_discount_is_rejected(gamma):
    model = DeterministicModel(((0,),), ((0.0,),), ((True,),))
    with pytest.raises(ValueError, match="gamma"):
        value_iteration(model, gamma=gamma)


def test_two_room_optimum_against_independent_safe_distance():
    params = two_room.default_params()
    positions, model = build_model(params)
    result = value_iteration(model, gamma=0.99)
    start = positions.index((3, 1))
    assert len(positions) == 73
    assert result.values[start] == pytest.approx(0.99**17, abs=1e-12, rel=0)
    assert result.values[positions.index((6, 10))] == 0.0
    assert result.bellman_residual <= 1e-12

    # Breadth-first search uses only the independent Python oracle, not the
    # JAX transition table or Bellman backup, to check the safe route length.
    distances = {(3, 1): 0}
    queue = deque([(3, 1)])
    while queue:
        position = queue.popleft()
        for action in range(4):
            next_position, reward, _ = reference.step(position, action)
            if reward < 0 or next_position in distances:
                continue
            distances[next_position] = distances[position] + 1
            queue.append(next_position)
    assert distances[(6, 10)] == 18
    route, actions, rewards = greedy_route(params, positions, result)
    assert route[0] == (3, 1)
    assert route[-1] == (6, 10)
    assert (3, 7) in route
    assert len(actions) == 18
    assert rewards == [0.0] * 17 + [1.0]
    assert sum(0.99**t * reward for t, reward in enumerate(rewards)) == pytest.approx(
        result.values[start], abs=1e-12, rel=0
    )
    for position, action, next_position in zip(route, actions, route[1:]):
        assert reference.step(position, action)[0] == next_position

    # Independently rebuild the entire model, including terminal flags.
    expected_rows = [
        [reference.step(position, action) for action in range(4)]
        for position in positions
    ]
    assert model.next_states == tuple(
        tuple(positions.index(item[0]) for item in row) for row in expected_rows
    )
    assert model.rewards == tuple(
        tuple(item[1] for item in row) for row in expected_rows
    )
    assert model.terminated == tuple(
        tuple(item[2] for item in row) for row in expected_rows
    )
