"""Independent semantic cases precede exhaustive and transformed comparisons."""

import jax
import jax.numpy as jnp
import pytest

from rrs.envs import reference, two_room


@pytest.fixture(scope="module")
def params():
    return two_room.default_params()


def outcome(position, action, params):
    state = two_room.State(jnp.array(position, dtype=jnp.int32))
    next_state, observation, reward, terminated = two_room.step(
        jax.random.key(0), state, action, params
    )
    assert jnp.array_equal(observation, next_state.position)
    return tuple(next_state.position.tolist()), float(reward), bool(terminated)


def test_geometry_and_reset(params):
    assert params.walls.shape == (8, 15)
    assert int((~params.walls).sum()) == 73
    assert int(params.penalties.sum()) == 20
    assert tuple(params.start.tolist()) == (3, 1)
    assert tuple(params.goal.tolist()) == (6, 10)
    assert two_room.legal_positions(params) == reference.legal_positions()
    for seed in (0, 17):
        state = two_room.reset(jax.random.key(seed), params)
        assert tuple(two_room.observe(state).tolist()) == (3, 1)


@pytest.mark.parametrize(
    "position, action, expected",
    [
        ((3, 1), 0, ((2, 1), 0.0, False)),
        ((3, 1), 1, ((4, 1), 0.0, False)),
        ((3, 2), 2, ((3, 1), 0.0, False)),  # leaving penalty
        ((3, 1), 3, ((3, 2), -1.0, False)),  # entering penalty
        ((3, 2), 3, ((3, 3), -1.0, False)),  # staying in penalty
        ((1, 2), 0, ((1, 2), -1.0, False)),  # blocked in penalty
        ((3, 6), 3, ((3, 7), 0.0, False)),
        ((3, 7), 3, ((3, 8), 0.0, False)),
        ((3, 8), 2, ((3, 7), 0.0, False)),
        ((3, 7), 2, ((3, 6), 0.0, False)),
        ((3, 7), 0, ((3, 7), 0.0, False)),
        ((3, 7), 1, ((3, 7), 0.0, False)),
        ((5, 10), 1, ((6, 10), 1.0, True)),
        ((6, 9), 3, ((6, 10), 1.0, True)),
        ((6, 11), 2, ((6, 10), 1.0, True)),
    ],
)
def test_semantics(params, position, action, expected):
    assert outcome(position, action, params) == expected
    assert reference.step(position, action) == expected


@pytest.mark.parametrize("row", [1, 2, 4, 5, 6])
@pytest.mark.parametrize("column, action", [(6, 3), (8, 2)])
def test_every_dividing_wall_segment(params, row, column, action):
    position = (row, column)
    assert outcome(position, action, params) == (position, 0.0, False)


@pytest.mark.parametrize("column", [*range(1, 7), *range(8, 14)])
def test_top_and_bottom_borders(params, column):
    reward = -1.0 if 2 <= column <= 5 else 0.0
    assert outcome((1, column), 0, params) == ((1, column), reward, False)
    assert outcome((6, column), 1, params) == ((6, column), 0.0, column == 10)


@pytest.mark.parametrize("row", range(1, 7))
def test_left_and_right_borders(params, row):
    assert outcome((row, 1), 2, params) == ((row, 1), 0.0, False)
    assert outcome((row, 13), 3, params) == ((row, 13), 0.0, False)


@pytest.mark.parametrize("action", range(4))
def test_terminal_is_absorbing_without_repeated_goal_reward(params, action):
    assert outcome((6, 10), action, params) == ((6, 10), 0.0, True)
    assert reference.step((6, 10), action) == ((6, 10), 0.0, True)


def test_all_nonterminal_transitions_against_independent_reference(params):
    comparisons = 0
    for position in reference.legal_positions():
        if position == (6, 10):
            continue
        for action in range(4):
            assert outcome(position, action, params) == reference.step(position, action)
            comparisons += 1
    assert comparisons == 288


def test_jit_vmap_and_combination_preserve_all_transitions(params):
    pairs = [(p, a) for p in reference.legal_positions() for a in range(4)]
    expected = [reference.step(p, a) for p, a in pairs]
    keys = jax.random.split(jax.random.key(42), len(pairs))
    states = two_room.State(jnp.array([p for p, _ in pairs], dtype=jnp.int32))
    actions = jnp.array([a for _, a in pairs], dtype=jnp.int32)
    compiled = jax.jit(two_room.step)
    for (position, action), wanted in zip(pairs, expected, strict=True):
        state, observation, reward, done = compiled(
            keys[0], two_room.State(jnp.array(position)), action, params
        )
        assert (tuple(state.position.tolist()), float(reward), bool(done)) == wanted
        assert jnp.array_equal(observation, state.position)
    batched = jax.vmap(two_room.step, in_axes=(0, 0, 0, None))
    for function in (batched, jax.jit(batched)):
        state, observations, rewards, dones = function(keys, states, actions, params)
        assert jnp.array_equal(state.position, jnp.array([x[0] for x in expected]))
        assert jnp.array_equal(observations, state.position)
        assert rewards.tolist() == [x[1] for x in expected]
        assert dones.tolist() == [x[2] for x in expected]
