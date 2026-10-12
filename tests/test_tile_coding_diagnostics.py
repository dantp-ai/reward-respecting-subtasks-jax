import jax
import jax.numpy as jnp
import pytest

from rrs.experiments import (
    tile_coded_options as original,
    tile_coding_diagnostics as diagnostics,
)
from rrs.experiments.hallway import build_baselines
from rrs.rl.learning import LearningParameters


@pytest.mark.parametrize("name", ["one_hot", "tile_coding"])
def test_diagnostic_runner_matches_original_updates_and_key_stream(name):
    data = build_baselines()
    tables = diagnostics.feature_tables(data)
    problem = diagnostics.make_problem(data, tables, name, name)
    old_problem, encode = original.training_problem(name)
    state, old = (
        diagnostics.initialize(9, problem),
        original.initialize(9, old_problem, encode),
    )
    params = LearningParameters(lambda_=0.2, lambda_actor=0.4)
    new_step = jax.jit(diagnostics.step)
    old_step = jax.jit(original.behavior_step, static_argnames=("encode",))
    for _ in range(40):
        state = new_step(state, problem, params)
        old, _ = old_step(old, old_problem, encode, params)
        assert jnp.array_equal(state.environment.position, old.environment.position)
        assert jnp.array_equal(
            jax.random.key_data(state.key), jax.random.key_data(old.key)
        )
        for a, b in zip(state.weights, old.weights, strict=True):
            assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)
    compiled = diagnostics.train(9, problem, steps=200, checkpoint=50)
    baseline = original.train(9, old_problem, encode, steps=200, checkpoint=50)
    assert jnp.allclose(compiled.critics, baseline.critics, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(
        compiled.actors[-1], baseline.final.weights.actor, atol=1e-6, rtol=1e-6
    )


def test_binary_tile_scaling_preserves_predictions_and_actor_preferences():
    data = build_baselines()
    tables = diagnostics.feature_tables(data)
    normalized = diagnostics.make_problem(data, tables, "tile_coding", "tile_coding")
    binary = diagnostics.make_problem(data, tables, "binary_tiles", "binary_tiles")
    a = diagnostics.train(0, normalized, steps=200, checkpoint=50)
    b = diagnostics.train(
        0,
        binary,
        LearningParameters(alpha=0.025, alpha_actor=0.025),
        steps=200,
        checkpoint=50,
    )
    assert jnp.allclose(a.critics, 2 * b.critics, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(a.actors, 2 * b.actors, atol=1e-6, rtol=1e-6)


def test_hybrid_dimensions_and_independent_geometry_audit():
    data = build_baselines()
    tables = diagnostics.feature_tables(data)
    audit = diagnostics.geometry_audit(data, tables)
    assert audit["independent_features_equal"]
    assert audit["rank"] == 72
    assert audit["optimal_value_fit_rmse"] < 1e-12
    problem = diagnostics.make_problem(data, tables, "one_hot", "tile_coding")
    result = diagnostics.train(0, problem, steps=20, checkpoint=10)
    assert result.critics.shape == (3, 72)
    assert result.actors.shape == (3, 4, 98)
    penalty = jnp.array(
        [bool(data.params.penalties[p[0], p[1]]) for p in data.positions]
    )
    gram = tables["split_penalty_tiles"] @ tables["split_penalty_tiles"].T
    assert jnp.all(gram[penalty[:, None] != penalty[None, :]] == 0)
    assert jnp.all(
        jnp.sum(
            tables["split_penalty_tiles"][jnp.array(data.nonterminal_indices)] ** 2,
            axis=1,
        )
        == 1
    )


def test_independent_sampler_matches_exact_deterministic_safe_route():
    data = build_baselines()
    option = data.options["reward_respecting"]
    pi = jax.nn.one_hot(jnp.array(option.policy), 4).tolist()
    sampled = diagnostics.sampled_audit(data, pi, option.stopping, samples=10)
    assert sampled["safe_success"] == {"mean": 1.0, "se": 0.0}
    assert sampled["return"]["mean"] == pytest.approx(0.99**11, abs=1e-12)


def test_sample_cap_is_an_error_and_not_silent_truncation():
    data = build_baselines()
    pi = [[0, 0, 1, 0]] * len(data.positions)  # left into wall from S forever
    with pytest.raises(RuntimeError, match="cap"):
        diagnostics.sampled_audit(
            data, pi, [False] * len(data.positions), samples=1, cap=3
        )
