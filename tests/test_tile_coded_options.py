import jax
import jax.numpy as jnp
import pytest

from rrs.envs import two_room
from rrs.experiments import option_learning as legacy, tile_coded_options as experiment
from rrs.rl.learning import LearningParameters, actor_critic_update, initial_weights
from rrs.rl.subtasks import stopping_value


def test_tabular_trajectory_weights_traces_and_compiled_training_match_legacy():
    old_problem = legacy.training_problem()
    problem, encode = experiment.training_problem("one_hot")
    old, new = (
        legacy.initialize(7, old_problem),
        experiment.initialize(7, problem, encode),
    )
    params = LearningParameters(lambda_=0.4, lambda_actor=0.2)
    old_step = jax.jit(legacy.behavior_step)
    new_step = jax.jit(experiment.behavior_step, static_argnames=("encode",))
    for _ in range(80):
        old, old_info = old_step(old, old_problem, params)
        new, new_info = new_step(new, problem, encode, params)
        assert jnp.array_equal(
            jax.random.key_data(old.key), jax.random.key_data(new.key)
        )
        assert jnp.array_equal(old.environment.position, new.environment.position)
        assert old.episodes == new.episodes
        for a, b in zip(
            (*old.weights, *old_info), (*new.weights, *new_info), strict=True
        ):
            assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)
    old_run = legacy.train(7, old_problem, steps=200, checkpoint=20)
    new_run = experiment.train(7, problem, encode, steps=200, checkpoint=20)
    assert jnp.allclose(old_run.critics, new_run.critics, atol=1e-6, rtol=1e-6)
    for a, b in zip(old_run.final.weights, new_run.final.weights, strict=True):
        assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ["one_hot", "tile_coding"])
def test_target_stays_binary_and_terminal_or_illegal_observations_are_zero(name):
    problem, encode = experiment.training_problem(name)
    old = legacy.training_problem()
    for observation in two_room.legal_positions(problem.environment):
        obs = jnp.array(observation)
        expected = stopping_value(
            old.main_weights, legacy.one_hot(obs, old.observations), old.hallway_feature
        )
        assert experiment.hallway_signal(obs) == expected
        x = encode(problem.representation, obs)
        assert float(x @ x) == (
            0.0 if observation == tuple(problem.environment.goal.tolist()) else 1.0
        )
    assert jnp.all(encode(problem.representation, jnp.array([0, 0])) == 0)


@pytest.mark.parametrize("name", ["one_hot", "tile_coding"])
@pytest.mark.parametrize("position, resets", [((3, 6), False), ((6, 9), True)])
def test_hallway_and_terminal_transitions_update_before_reset(name, position, resets):
    problem, encode = experiment.training_problem(name)
    state = experiment.initialize(0, problem, encode)._replace(
        environment=two_room.State(jnp.array(position))
    )
    key = jax.random.key(1)
    updated, info = experiment.experience_step(state, 3, key, key, problem, encode)
    x = encode(problem.representation, jnp.array(position))
    assert bool(info.beta)
    assert float(info.delta) == 1.0
    assert jnp.allclose(updated.weights.critic, 0.1 * x)
    assert jnp.allclose(updated.weights.actor[3], 0.075 * x)
    assert jnp.allclose(
        updated.weights.actor[:3], jnp.broadcast_to(-0.025 * x, (3, len(x)))
    )
    assert int(updated.episodes) == int(resets)
    assert updated.environment.position.tolist() == ([3, 1] if resets else [3, 7])
    assert jnp.all(updated.weights.critic_trace == 0)


def test_overlapping_features_use_shared_continuation_then_terminal_override():
    x, nx = jnp.array([0.5, 0.5, 0.5, 0.5, 0, 0]), jnp.array([0.5, 0.5, 0, 0, 0.5, 0.5])
    weights = initial_weights(6, 4)._replace(critic=jnp.full(6, 2.0))
    updated, info = actor_critic_update(weights, x, 0, 1.0, nx, 0.0, 0.25, False)
    # V=V'=4, delta=1+.99*4-4=.96, pi=mu=.25, rho=1.
    assert float(info.delta) == pytest.approx(0.96, abs=1e-6)
    assert not bool(info.beta)
    assert jnp.allclose(updated.critic, jnp.array([2.048] * 4 + [2.0] * 2), atol=1e-6)
    assert jnp.allclose(updated.actor[0], jnp.array([0.036] * 4 + [0.0] * 2), atol=1e-6)
    terminal, info = actor_critic_update(weights, x, 0, 1.0, nx, 999.0, 0.25, True)
    assert float(info.delta) == -3.0
    assert jnp.allclose(terminal.critic, jnp.array([1.85] * 4 + [2.0] * 2))


def test_short_tile_training_is_reproducible_vmapped_and_pairs_environment():
    problem, encode = experiment.training_problem("tile_coding")

    def run(seed):
        return experiment.train(seed, problem, encode, steps=40, checkpoint=20)

    scalar = run(3)
    with jax.disable_jit():
        eager = experiment.train(3, problem, encode, steps=4, checkpoint=2)
    compiled = experiment.train(3, problem, encode, steps=4, checkpoint=2)
    assert jnp.allclose(eager.critics, compiled.critics, atol=1e-6, rtol=1e-6)
    batch = jax.vmap(run)(jnp.array([3, 4]))
    assert jnp.array_equal(scalar.critics, run(3).critics)
    assert jnp.allclose(batch.critics[0], scalar.critics, atol=1e-6)
    baseline = legacy.train(3, legacy.training_problem(), steps=40, checkpoint=20)
    assert jnp.array_equal(
        scalar.final.environment.position, baseline.final.environment.position
    )
    assert jnp.array_equal(
        jax.random.key_data(scalar.final.key), jax.random.key_data(baseline.final.key)
    )
    assert scalar.final.episodes == baseline.final.episodes
