"""Distribution-level checks against independent geometry and hand cases."""

import math
import random
from collections import Counter

import jax
import jax.numpy as jnp
import pytest

from rrs.envs import four_room as env, four_room_reference as reference
from rrs.representations.tabular import one_hot


@pytest.fixture(scope="module")
def params():
    return env.default_params()


def aggregate(branches):
    result = {}
    for p, position, reward, done in zip(
        *jax.tree.map(lambda x: x.tolist(), branches), strict=True
    ):
        key = tuple(position), reward, done
        result[key] = result.get(key, 0.0) + p
    return result


def reference_distribution(position, action):
    return {
        (o.position, o.reward, o.terminated): o.probability
        for o in reference.distribution(position, action)
    }


def test_figure_six_geometry_and_separate_representation(params):
    positions = env.legal_positions(params)
    assert positions == reference.legal_positions()
    assert params.walls.shape == (13, 13)
    assert len(positions) == 104
    assert int(params.penalties.sum()) == 12
    assert env.HALLWAYS == ((6, 2), (3, 6), (7, 9), (10, 6))
    for row, col in env.HALLWAYS:
        assert not params.walls[row, col]
        assert not params.penalties[row, col]
    for seed in (0, 17):
        state = env.reset(jax.random.key(seed), params)
        assert env.observe(state).tolist() == [4, 1]
    assert params.goal.tolist() == [9, 7]
    features = jnp.array([p for p in positions if p != (9, 7)])
    assert one_hot(params.start, features).shape == (103,)
    assert float(one_hot(params.start, features).sum()) == 1.0
    assert float(one_hot(params.goal, features).sum()) == 0.0


@pytest.mark.parametrize(
    "position,action,expected",
    [
        (
            (1, 1),
            0,
            {
                ((1, 1), 0.0, False): 7 / 9,
                ((2, 1), 0.0, False): 1 / 9,
                ((1, 2), 0.0, False): 1 / 9,
            },
        ),
        (
            (6, 2),
            0,
            {
                ((5, 2), 0.0, False): 2 / 3,
                ((6, 2), 0.0, False): 2 / 9,
                ((7, 2), -1.0, False): 1 / 9,
            },
        ),
        (
            (7, 3),
            0,
            {
                ((7, 3), -1.0, False): 2 / 3,
                ((8, 3), -1.0, False): 1 / 9,
                ((7, 2), -1.0, False): 1 / 9,
                ((7, 4), -1.0, False): 1 / 9,
            },
        ),
        (
            (7, 2),
            0,
            {
                ((6, 2), 0.0, False): 2 / 3,
                ((8, 2), -1.0, False): 1 / 9,
                ((7, 1), 0.0, False): 1 / 9,
                ((7, 3), -1.0, False): 1 / 9,
            },
        ),
        (
            (9, 8),
            2,
            {
                ((9, 7), 1.0, True): 2 / 3,
                ((8, 8), 0.0, False): 1 / 9,
                ((10, 8), 0.0, False): 1 / 9,
                ((9, 9), 0.0, False): 1 / 9,
            },
        ),
    ],
)
def test_hand_calculated_merged_outcomes(params, position, action, expected):
    assert reference_distribution(position, action) == pytest.approx(
        expected, abs=1e-15
    )
    actual = aggregate(env.outcomes(env.State(jnp.array(position)), action, params))
    assert actual == pytest.approx(expected, abs=1e-7)


def test_all_state_action_distributions_match_independent_reference(params):
    compiled = jax.jit(env.outcomes)
    for position in reference.legal_positions():
        for action in range(4):
            state = env.State(jnp.array(position))
            eager = env.outcomes(state, action, params)
            optimized = compiled(state, action, params)
            for a, b in zip(eager, optimized, strict=True):
                assert jnp.array_equal(a, b)
            actual = aggregate(eager)
            assert actual == pytest.approx(
                reference_distribution(position, action), abs=1e-7
            )
            assert sum(actual.values()) == pytest.approx(1.0, abs=1e-7)


def test_sampling_is_seeded_and_eager_jit_vmap_agree(params):
    pairs = [(p, a) for p in reference.legal_positions() for a in range(4)]
    keys = jax.random.split(jax.random.key(61), len(pairs))
    states = env.State(jnp.array([p for p, _ in pairs]))
    actions = jnp.array([a for _, a in pairs])
    batch = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    result = batch(keys, states, actions, params)
    compiled = jax.jit(env.step)
    for i, (position, action) in enumerate(pairs):
        state = env.State(jnp.array(position))
        eager = env.step(keys[i], state, action, params)
        actual = compiled(keys[i], state, action, params)
        for a, b, c in zip(
            jax.tree.leaves(eager),
            jax.tree.leaves(actual),
            jax.tree.leaves(result),
            strict=True,
        ):
            assert jnp.array_equal(a, b)
            assert jnp.array_equal(a, c[i])
        _, observation, reward, done = eager
        outcome = tuple(observation.tolist()), float(reward), bool(done)
        assert outcome in reference_distribution(position, action)
    for other in (
        batch(keys, states, actions, params),
        jax.jit(batch)(keys, states, actions, params),
    ):
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(result), jax.tree.leaves(other), strict=True
            )
        )


def test_sampled_probabilities_match_reference_including_blocked_mass(params):
    count = 18_000
    keys = jax.random.split(jax.random.key(6100), count)
    sample = jax.jit(jax.vmap(env.step, in_axes=(0, None, None, None)))
    for position in ((1, 1), (6, 2), (9, 8), (9, 3)):
        for action in range(4):
            _, observations, rewards, terminated = sample(
                keys, env.State(jnp.array(position)), action, params
            )
            counts = Counter(
                (tuple(p), r, done)
                for p, r, done in zip(
                    observations.tolist(),
                    rewards.tolist(),
                    terminated.tolist(),
                    strict=True,
                )
            )
            expected = reference_distribution(position, action)
            assert counts.keys() <= expected.keys()
            for outcome, p in expected.items():
                assert abs(counts[outcome] / count - p) <= 0.005 + 6 * math.sqrt(
                    p * (1 - p) / count
                )


def test_goal_absorbs_with_zero_reward_and_no_reset(params):
    for action in range(4):
        assert reference_distribution((9, 7), action) == pytest.approx(
            {((9, 7), 0.0, True): 1.0}
        )
        for seed in range(8):
            state, observation, reward, done = env.step(
                jax.random.key(seed), env.State(params.goal), action, params
            )
            assert state.position.tolist() == observation.tolist() == [9, 7]
            assert float(reward) == 0 and bool(done)


def test_python_ticket_sampler_and_invalid_inputs():
    class Ticket:
        def __init__(self, value):
            self.value = value

        def randrange(self, stop):
            assert stop == 9
            return self.value

    for action in range(4):
        counts = Counter(reference.step((9, 3), action, Ticket(i)) for i in range(9))
        assert {key: n / 9 for key, n in counts.items()} == pytest.approx(
            reference_distribution((9, 3), action)
        )
    first, second = random.Random(60), random.Random(60)
    assert [reference.step((4, 1), 3, first) for _ in range(20)] == [
        reference.step((4, 1), 3, second) for _ in range(20)
    ]
    with pytest.raises(ValueError, match="traversable"):
        reference.distribution((0, 0), 0)
    with pytest.raises(ValueError, match="Action"):
        reference.step((4, 1), 4, first)
