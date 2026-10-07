import jax
import jax.numpy as jnp
import pytest

from rrs.envs import four_room, four_room_reference
from rrs.experiments.four_room_options import (
    PARAMETERS,
    behavior_step,
    experience_step,
    initialize,
    train_block,
    training_problem,
)
from rrs.representations.tabular import one_hot
from rrs.rl.learning import actor_critic_update, softmax_policy
from rrs.rl.subtasks import stopping_value


@pytest.fixture(scope="module")
def problem():
    return training_problem()


def deterministic(problem):
    return problem._replace(
        params=problem.params._replace(motion_probabilities=jnp.eye(4))
    )


def test_features_and_shared_transition_have_four_distinct_targets(problem):
    problem = deterministic(problem)
    assert problem.observations.shape == (103, 2)
    assert problem.observations[problem.hallway_features].tolist() == [
        list(p) for p in four_room.HALLWAYS
    ]
    state = initialize(7000, problem)._replace(
        environment=four_room.State(jnp.array([3, 5]))
    )
    updated, info = experience_step(state, 3, state.key, state.key, problem)
    assert info.delta.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert info.beta.tolist() == [True] * 4
    assert updated.environment.position.tolist() == [
        3,
        6,
    ]  # Hallway stopping does not reset.
    assert int(updated.episodes) == 0
    feature = problem.observations.tolist().index([3, 5])
    assert updated.weights.critic[:, feature].tolist() == pytest.approx(
        [0.0, 0.05, 0.0, 0.0]
    )


def test_every_option_gets_its_own_preupdate_ratio_and_requested_action(problem):
    # Request UP at H1, but force a DOWN outcome into a penalty cell.
    probability = jnp.eye(4).at[0].set(jax.nn.one_hot(1, 4))
    problem = problem._replace(
        params=problem.params._replace(motion_probabilities=probability)
    )
    state = initialize(7000, problem)._replace(
        environment=four_room.State(jnp.array([6, 2]))
    )
    x = one_hot(state.environment.position, problem.observations)
    state = state._replace(
        weights=state.weights._replace(actor=state.weights.actor.at[0, 0].set(2 * x))
    )
    updated, info = experience_step(state, 0, state.key, state.key, problem)
    assert updated.environment.position.tolist() == [7, 2]
    nx = one_hot(updated.environment.position, problem.observations)
    for i in range(4):
        weights = jax.tree.map(lambda a: a[i], state.weights)
        z = stopping_value(problem.main_weights, nx, problem.hallway_features[i])
        expected, diagnostic = actor_critic_update(
            weights, x, 0, -1.0, nx, z, 0.25, False, PARAMETERS
        )
        for actual, wanted in zip(updated.weights, expected, strict=True):
            assert jnp.allclose(actual[i], wanted, atol=1e-6)
        assert float(info.rho[i]) == pytest.approx(float(diagnostic.rho))
    assert float(info.rho[0]) == pytest.approx(
        float(softmax_policy(state.weights.actor[0], x)[0]) / 0.25
    )
    assert float(info.rho[0]) > float(info.rho[1])


def test_terminal_reward_is_learned_by_all_four_options_before_reset(problem):
    problem = deterministic(problem)
    state = initialize(7000, problem)._replace(
        environment=four_room.State(jnp.array([9, 8]))
    )
    updated, info = experience_step(state, 2, state.key, state.key, problem)
    assert info.delta.tolist() == [1.0] * 4
    assert info.beta.tolist() == [True] * 4
    assert int(updated.episodes) == 1
    assert updated.environment.position.tolist() == [4, 1]
    feature = problem.observations.tolist().index([9, 8])
    assert updated.weights.critic[:, feature].tolist() == pytest.approx([0.05] * 4)
    assert jnp.array_equal(
        one_hot(problem.params.goal, problem.observations), jnp.zeros(103)
    )


def test_key_schedule_eager_compiled_scan_and_reference_outcome_support(problem):
    state = initialize(7001, problem)
    for _ in range(12):
        next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
        action = jax.random.randint(action_key, (), 0, 4)
        raw, obs, reward, done = four_room.step(
            step_key, state.environment, action, problem.params
        )
        expected = four_room_reference.distribution(
            tuple(state.environment.position.tolist()), int(action)
        )
        assert (tuple(obs.tolist()), float(reward), bool(done)) in {
            (o.position, o.reward, o.terminated) for o in expected
        }
        manual, _ = experience_step(state, action, step_key, reset_key, problem)
        state, _ = behavior_step(state, problem)
        for a, b in zip(state.weights, manual.weights, strict=True):
            assert jnp.allclose(a, b, atol=1e-6)
        assert jnp.array_equal(
            jax.random.key_data(state.key), jax.random.key_data(next_key)
        )
    compiled = train_block(initialize(7001, problem), problem, steps=12)
    for a, b in zip(compiled.weights, state.weights, strict=True):
        assert jnp.allclose(a, b, atol=1e-6)
    assert jnp.array_equal(compiled.environment.position, state.environment.position)


def test_short_runs_are_finite_reproducible_batched_and_independent(problem):
    def run(seed):
        return train_block(initialize(seed, problem), problem, steps=256)

    first, second = run(7000), run(7000)
    batched = jax.vmap(run)(jnp.array([7000, 7001]))
    for a, b, c in zip(first.weights, second.weights, batched.weights, strict=True):
        assert jnp.array_equal(a, b)
        assert jnp.allclose(a, c[0], atol=1e-6)
        assert jnp.isfinite(c).all()
    assert not jnp.array_equal(batched.weights.critic[0], batched.weights.critic[1])
    # A short uniform walk may never encounter a reward or a hallway feature.
    assert bool(jnp.any(batched.weights.critic != 0))
    assert jnp.array_equal(first.weights.critic_trace, jnp.zeros((4, 103)))
