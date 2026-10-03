import jax
import jax.numpy as jnp
import pytest

from rrs.envs import two_room
from rrs.envs.reference import step as reference_step
from rrs.experiments.option_learning import (
    behavior_step,
    experience_step,
    initialize,
    train,
    training_problem,
)


@pytest.fixture(scope="module")
def problem():
    return training_problem()


def test_option_stopping_does_not_reset_behavior(problem):
    state = initialize(0, problem)
    # All zero values => beta=1. Moving down must still leave behavior at (4,1).
    state, info = experience_step(state, 1, state.key, state.key, problem)
    assert bool(info.beta)
    assert state.environment.position.tolist() == [4, 1]
    assert int(state.episodes) == 0


def test_goal_update_uses_terminal_features_before_reset(problem):
    state = initialize(0, problem)._replace(
        environment=two_room.State(jnp.array([6, 9]))
    )
    state, info = experience_step(state, 3, state.key, state.key, problem)
    feature = problem.observations.tolist().index([6, 9])
    assert float(info.delta) == 1
    assert bool(info.beta)
    assert state.environment.position.tolist() == [3, 1]
    assert int(state.episodes) == 1
    assert float(state.weights.critic[feature]) == pytest.approx(0.1, abs=1e-6)
    assert state.weights.actor[:, feature].tolist() == pytest.approx(
        [-0.025, -0.025, -0.025, 0.075], abs=1e-6
    )
    assert int(jnp.count_nonzero(state.weights.critic)) == 1


def test_behavior_key_schedule_and_eager_compiled_step_agree(problem):
    state = initialize(7, problem)
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    expected, _ = experience_step(state, action, step_key, reset_key, problem)
    expected = expected._replace(key=next_key)
    actual, _ = behavior_step(state, problem)
    compiled, _ = jax.jit(behavior_step)(state, problem)
    for result in (actual, compiled):
        assert jnp.array_equal(
            jax.random.key_data(result.key), jax.random.key_data(expected.key)
        )
        assert jnp.array_equal(
            result.environment.position, expected.environment.position
        )
        for a, b in zip(result.weights, expected.weights, strict=True):
            assert jnp.allclose(a, b, atol=1e-6)


def test_scan_matches_eager_sequence_and_independent_environment(problem):
    state = initialize(3, problem)
    position = (3, 1)
    expected = [state.weights.critic]
    for step in range(12):
        _, action_key, _, _ = jax.random.split(state.key, 4)
        action = int(jax.random.randint(action_key, (), 0, 4))
        position, _, done = reference_step(position, action)
        position = (3, 1) if done else position
        state, _ = behavior_step(state, problem)
        assert tuple(state.environment.position.tolist()) == position
        if (step + 1) % 4 == 0:
            expected.append(state.weights.critic)
    result = train(3, problem, steps=12, checkpoint=4)
    assert jnp.allclose(result.critics, jnp.stack(expected), atol=1e-6)
    for a, b in zip(result.final.weights, state.weights, strict=True):
        assert jnp.allclose(a, b, atol=1e-6)
    assert jnp.array_equal(
        jax.random.key_data(result.final.key), jax.random.key_data(state.key)
    )


def test_short_training_is_reproducible_finite_and_learns(problem):
    first = train(0, problem, steps=512, checkpoint=128)
    second = train(0, problem, steps=512, checkpoint=128)
    batched = jax.vmap(lambda seed: train(seed, problem, steps=512, checkpoint=128))(
        jnp.array([0, 1])
    )
    assert first.critics.shape == (5, 72)
    assert jnp.array_equal(first.critics, second.critics)
    assert jnp.allclose(first.critics, batched.critics[0], atol=1e-6)
    assert not jnp.array_equal(batched.critics[0], batched.critics[1])
    for weights in first.final.weights:
        assert jnp.isfinite(weights).all()
    assert bool(jnp.any(first.final.weights.critic != 0))
    assert bool(jnp.any(first.final.weights.actor != 0))
    assert jnp.array_equal(first.final.weights.critic_trace, jnp.zeros(72))
    assert jnp.array_equal(first.final.weights.actor_trace, jnp.zeros((4, 72)))
