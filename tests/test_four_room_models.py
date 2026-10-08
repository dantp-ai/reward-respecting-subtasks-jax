import jax
import jax.numpy as jnp
import pytest

from rrs.envs import four_room
from rrs.experiments.four_room_models import (
    behavior_step,
    experience_step,
    initialize,
    train_block,
    training_problem,
)
from rrs.experiments.model_learning import FixedOptions
from rrs.rl.model_learning import ModelLearningParameters


@pytest.fixture(scope="module")
def setup():
    problem = training_problem()
    n = len(problem.observations)
    pi = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, n)),
            jnp.broadcast_to(
                jnp.array(
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.4, 0.3, 0.2, 0.1],
                        [0.25] * 4,
                        [0.5, 0.25, 0.125, 0.125],
                    ]
                )[:, :, None],
                (4, 4, n),
            ),
        )
    )
    beta = jnp.ones((8, n)).at[4:].set(0)
    return problem, FixedOptions(pi, beta)


def step_key_for(state, action, position, params):
    for seed in range(100):
        key = jax.random.key(seed)
        if tuple(four_room.step(key, state, action, params)[1].tolist()) == position:
            return key
    raise AssertionError("No key for desired stochastic outcome")


def test_shared_outcome_uses_requested_action_ratios_and_environment_reward(setup):
    problem, options = setup
    state = initialize(0, problem)._replace(
        environment=four_room.State(jnp.array((6, 2)))
    )
    # Request UP, realize DOWN into a penalty: only the UP primitive updates.
    key = step_key_for(state.environment, 0, (7, 2), problem.params)
    result, errors = experience_step(state, 0, key, key, problem, options)
    source = problem.observations.tolist().index([6, 2])
    dest = problem.observations.tolist().index([7, 2])
    rho = jnp.array([4, 0, 0, 0, 0.4, 1.6, 1, 2])
    assert result.learners.model.reward_weights[:, source].tolist() == pytest.approx(
        (-0.1 * rho).tolist()
    )
    assert result.learners.model.successor_weights[
        :, dest, source
    ].tolist() == pytest.approx([0.396, 0, 0, 0, 0, 0, 0, 0])
    assert errors.reward.tolist() == [-1] * 8
    assert result.environment.position.tolist() == [7, 2]
    assert int(result.episodes) == 0


def test_arrival_stopping_has_no_bonus_and_never_resets_behavior(setup):
    problem, options = setup
    state = initialize(0, problem)._replace(
        environment=four_room.State(jnp.array((3, 5)))
    )
    target = problem.observations.tolist().index([3, 6])
    source = problem.observations.tolist().index([3, 5])
    options = options._replace(
        stopping_weights=options.stopping_weights.at[5, target].set(1)
    )
    key = step_key_for(state.environment, 3, (3, 6), problem.params)
    result, _ = experience_step(state, 3, key, key, problem, options)
    assert jnp.all(result.learners.model.reward_weights == 0)
    assert float(
        result.learners.model.successor_weights[5, target, source]
    ) == pytest.approx(0.0396)
    assert float(result.learners.model.successor_weights[4, target, source]) == 0
    assert result.environment.position.tolist() == [3, 6]
    assert int(result.episodes) == 0


def test_terminal_reward_is_learned_before_reset_and_clears_traces(setup):
    problem, options = setup
    state = initialize(0, problem)._replace(
        environment=four_room.State(jnp.array((9, 8)))
    )
    key = step_key_for(state.environment, 2, (9, 7), problem.params)
    source = problem.observations.tolist().index([9, 8])
    result, _ = experience_step(
        state, 2, key, key, problem, options, ModelLearningParameters(lambda_=0.8)
    )
    assert result.learners.model.reward_weights[:, source].tolist() == pytest.approx(
        [0, 0, 0.4, 0, 0.12, 0.08, 0.1, 0.05]
    )
    assert jnp.all(result.learners.model.successor_weights == 0)
    assert jnp.all(result.learners.reward_trace == 0)
    assert jnp.all(result.learners.successor_trace == 0)
    assert result.environment.position.tolist() == [4, 1]
    assert int(result.episodes) == 1


def assert_trees_equal(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        assert jnp.allclose(
            jax.random.key_data(x)
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
            else x,
            jax.random.key_data(y)
            if jax.dtypes.issubdtype(y.dtype, jax.dtypes.prng_key)
            else y,
            atol=1e-6,
        )


def test_key_schedule_eager_jit_scan_and_vmap_agree(setup):
    problem, options = setup
    initial = initialize(8000, problem)
    next_key, action_key, key, reset = jax.random.split(initial.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    explicit, _ = experience_step(initial, action, key, reset, problem, options)
    actual, _ = behavior_step(initial, problem, options)
    assert_trees_equal(actual, explicit._replace(key=next_key))
    eager = initial
    for _ in range(4):
        eager, _ = behavior_step(eager, problem, options)
    compiled = train_block(initial, problem, options, steps=4)
    assert_trees_equal(eager, compiled)
    batched = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array([8000, 8001]))
    run = jax.jit(jax.vmap(lambda s: train_block(s, problem, options, steps=64)))
    a, b = run(batched), run(batched)
    assert_trees_equal(a, b)
    assert jnp.any(a.learners.model.successor_weights != 0)
    assert jnp.isfinite(a.learners.model.successor_weights).all()
    assert_trees_equal(
        jax.tree.map(lambda x: x[0], a),
        train_block(initial, problem, options, steps=64),
    )


def test_continuing_destination_bootstraps_each_models_own_weights(setup):
    problem, options = setup
    state = initialize(0, problem)
    key = step_key_for(state.environment, 0, (3, 1), problem.params)
    source = problem.observations.tolist().index([4, 1])
    dest = problem.observations.tolist().index([3, 1])
    model = state.learners.model._replace(
        reward_weights=state.learners.model.reward_weights.at[:, dest].set(
            jnp.arange(8)
        )
    )
    state = state._replace(learners=state.learners._replace(model=model))
    result, _ = experience_step(state, 0, key, key, problem, options)
    assert result.learners.model.reward_weights[:4, source].tolist() == [0] * 4
    assert result.learners.model.reward_weights[4:, source].tolist() == pytest.approx(
        [0.1584, 0.792, 0.594, 1.386]
    )
