import jax
import jax.numpy as jnp
import pytest

from rrs.envs import reference, two_room
from rrs.experiments.hallway import build_baselines
from rrs.experiments.model_learning import (
    behavior_step,
    experience_step,
    freeze_options,
    initialize,
    option_tables,
    train_block,
)
from rrs.experiments.option_learning import training_problem
from rrs.rl.learning import initial_weights


@pytest.fixture(scope="module")
def setup():
    data, problem = build_baselines(), training_problem()
    options = freeze_options(data, initial_weights(72, 4))
    return data, problem, options


def test_frozen_options_keep_stochastic_policy_and_feature_order(setup):
    data, _, options = setup
    pi, beta = option_tables(options, data)
    assert jnp.array_equal(pi[4], jnp.full((73, 4), 0.25))
    assert jnp.array_equal(beta[4], jnp.ones(73))
    for model in (0, 1, 2, 3, 5):
        name = ("up", "down", "left", "right", "reward_respecting", "shortest_path")[
            model
        ]
        for s in data.nonterminal_indices:
            assert int(jnp.argmax(pi[model, s])) == data.options[name].policy[s]
            assert float(beta[model, s]) == data.options[name].stopping[s]


def test_all_models_learn_from_one_transition_with_own_ratios(setup):
    _, problem, options = setup
    state = initialize(0, problem)
    # Right from S enters a penalty. Initial learned pi is uniform; RIGHT pi=1.
    state, errors = experience_step(state, 3, state.key, state.key, problem, options)
    feature = problem.observations.tolist().index([3, 1])
    weights = state.learners.model.reward_weights[:, feature]
    assert weights[:5].tolist() == pytest.approx([0.0, 0.0, 0.0, -0.4, -0.1], abs=1e-6)
    assert float(errors.reward[3]) == -1.0
    assert state.environment.position.tolist() == [3, 2]  # stopping does not reset


def test_goal_reward_learned_before_behavior_reset(setup):
    _, problem, options = setup
    state = initialize(0, problem)._replace(
        environment=two_room.State(jnp.array([6, 9]))
    )
    state, _ = experience_step(state, 3, state.key, state.key, problem, options)
    feature = problem.observations.tolist().index([6, 9])
    assert float(state.learners.model.reward_weights[3, feature]) == pytest.approx(0.4)
    assert jnp.array_equal(
        state.learners.model.successor_weights[3], jnp.zeros((72, 72))
    )
    assert state.environment.position.tolist() == [3, 1]
    assert int(state.episodes) == 1


def test_eager_sequence_matches_scan_and_reference_environment(setup):
    _, problem, options = setup
    state = initialize(1000, problem)
    position = (3, 1)
    for _ in range(12):
        _, action_key, _, _ = jax.random.split(state.key, 4)
        action = int(jax.random.randint(action_key, (), 0, 4))
        position, _, done = reference.step(position, action)
        position = (3, 1) if done else position
        state, _ = behavior_step(state, problem, options)
        assert tuple(state.environment.position.tolist()) == position
    compiled = train_block(initialize(1000, problem), problem, options, steps=12)
    for a, b in zip(
        jax.tree.leaves(state.learners), jax.tree.leaves(compiled.learners), strict=True
    ):
        assert jnp.allclose(a, b, atol=1e-6)
    assert jnp.array_equal(
        jax.random.key_data(state.key), jax.random.key_data(compiled.key)
    )


def test_short_seeded_training_is_finite_reproducible_and_batched(setup):
    _, problem, options = setup
    initial = initialize(1000, problem)
    first = train_block(initial, problem, options, steps=128)
    second = train_block(initial, problem, options, steps=128)
    batched = jax.vmap(
        lambda seed: train_block(initialize(seed, problem), problem, options, steps=128)
    )(jnp.array([1000, 1001]))
    for a, b, c in zip(
        first.learners.model, second.learners.model, batched.learners.model, strict=True
    ):
        assert jnp.array_equal(a, b)
        assert jnp.isfinite(c).all()
        assert jnp.allclose(a, c[0], atol=1e-6)
        assert not jnp.array_equal(c[0], c[1])
    assert bool(jnp.any(first.learners.model.reward_weights != 0))
    assert jnp.array_equal(first.learners.reward_trace, jnp.zeros((6, 72)))
    assert jnp.array_equal(first.learners.successor_trace, jnp.zeros((6, 72, 72)))
