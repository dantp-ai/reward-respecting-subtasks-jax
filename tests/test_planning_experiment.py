import jax
import jax.numpy as jnp
import pytest

from rrs.experiments.planning import run_plan, sample_states
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.planning import planning_step, tabular_plan


def toy_models(count):
    rewards = jnp.array([[0.0, 1.0], [-1.0, 0.0]] + [[-2.0, -2.0]] * (count - 2))
    successor = jnp.zeros((count, 2, 2)).at[0, 1, 0].set(0.5)
    return LinearExpectationModel(rewards, successor)


def test_sampling_is_explicit_and_paired_prefix_is_independent_of_length():
    small, large = sample_states(2000, 72, 12), sample_states(2000, 72, 24)
    assert jnp.array_equal(small, large[:12])
    key, expected = jax.random.key(2000), []
    for _ in range(12):
        key, sample_key = jax.random.split(key)
        expected.append(int(jax.random.randint(sample_key, (), 0, 72)))
    assert small.tolist() == expected


def test_equal_query_budgets_mean_different_update_counts():
    order = jnp.array([1, 0, 1, 0, 1])
    primitive = run_plan(toy_models(4), order, lookahead_budget=20, checkpoint=20)
    option = run_plan(toy_models(5), order, lookahead_budget=20, checkpoint=20)
    assert int(primitive.updates) == 5
    assert int(option.updates) == 4
    assert int(primitive.lookaheads) == int(option.lookaheads) == 20
    with pytest.raises(ValueError, match="complete"):
        run_plan(toy_models(4), order, lookahead_budget=20, checkpoint=10)


def test_scan_matches_eager_and_independent_tabular_sequence():
    models, order = toy_models(4), jnp.array([0, 1, 0, 1, 0, 1, 1, 0])
    weights, expected = jnp.zeros(2), [jnp.zeros(2)]
    for i, state in enumerate(order.tolist()):
        weights, _ = planning_step(weights, jax.nn.one_hot(state, 2), models)
        if (i + 1) % 2 == 0:
            expected.append(weights)
    compiled = run_plan(models, order, lookahead_budget=32, checkpoint=8)
    assert jnp.allclose(compiled.history, jnp.stack(expected), atol=1e-6)
    rewards = models.reward_weights.T.tolist()
    successors = jnp.transpose(models.successor_weights, (2, 0, 1)).tolist()
    oracle = tabular_plan(rewards, successors, order.tolist())
    assert compiled.weights.tolist() == pytest.approx(oracle.values, abs=1e-6)
    assert int(compiled.lookaheads) == oracle.lookaheads


def test_short_planning_is_finite_reproducible_and_batchable():
    models = toy_models(5)
    order = sample_states(2000, 2, 40)
    first = run_plan(models, order, lookahead_budget=200, checkpoint=100)
    second = run_plan(models, order, lookahead_budget=200, checkpoint=100)
    batched = jax.vmap(
        lambda seed: run_plan(models, sample_states(seed, 2, 40), 200, 100)
    )(jnp.array([2000, 2001]))
    assert jnp.array_equal(first.history, second.history)
    assert jnp.allclose(first.history, batched.history[0], atol=1e-6)
    assert jnp.isfinite(batched.history).all()
    assert first.weights.tolist() == [0.5, 1.0]
