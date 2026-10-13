import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rrs.envs import two_room
from rrs.experiments import (
    model_learning as tabular,
    tile_coded_model_report as reporting,
    tile_coded_models as experiment,
)
from rrs.experiments.hallway import build_baselines
from rrs.experiments.model_audit import sample_model
from rrs.experiments.option_learning import training_problem
from rrs.experiments.tile_coded_model_report import (
    build_references,
    decoder,
    model_errors,
    save_json,
    summarize,
)
from rrs.rl.learning import initial_weights
from rrs.rl.model_learning import ModelLearningParameters
from rrs.rl.option_models import LinearExpectationModel


@pytest.fixture(scope="module")
def data():
    return build_baselines()


@pytest.mark.parametrize("trace", [0.0, 0.3])
def test_one_hot_runner_preserves_tabular_trajectories_parameters_and_keys(data, trace):
    old_problem = training_problem()
    options = tabular.freeze_options(data, initial_weights(72, 4))
    pi, beta = tabular.option_tables(options, data)
    problem = experiment.Problem(
        data.params,
        jnp.array(data.positions),
        experiment.features(data, "one_hot"),
        pi,
        beta,
    )
    old, new = (
        tabular.initialize(2000, old_problem),
        experiment.initialize(2000, problem),
    )
    params = ModelLearningParameters(lambda_=trace)
    old_step = jax.jit(tabular.behavior_step)
    new_step = jax.jit(experiment.behavior_step)
    for _ in range(20):
        old, _ = old_step(old, old_problem, options, params)
        new, _ = new_step(new, problem, params)
        assert jnp.array_equal(old.environment.position, new.environment.position)
        assert jnp.array_equal(
            jax.random.key_data(old.key), jax.random.key_data(new.key)
        )
        for a, b in zip(
            jax.tree.leaves(old.learners), jax.tree.leaves(new.learners), strict=True
        ):
            assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)
    old = tabular.train_block(old, old_problem, options, steps=100, parameters=params)
    new = experiment.train_block(new, problem, steps=100, parameters=params)
    for a, b in zip(
        jax.tree.leaves(old.learners), jax.tree.leaves(new.learners), strict=True
    ):
        assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "position,destination,reward,done",
    [
        ((3, 1), (3, 2), -1.0, False),
        ((3, 6), (3, 7), 0.0, False),
        ((6, 9), (6, 10), 1.0, True),
    ],
)
def test_mixed_feature_updates_use_observation_policy_and_actual_reward(
    data, position, destination, reward, done
):
    x = experiment.features(data, "tile_coding")
    s, ns = data.positions.index(position), data.positions.index(destination)
    pi = (
        jnp.full((1, len(data.positions), 4), 0.25)
        .at[0, s]
        .set(jnp.array([0.1, 0.1, 0.1, 0.7]))
    )
    beta = jnp.ones((1, len(data.positions))).at[0, ns].set(0.3)
    problem = experiment.Problem(data.params, jnp.array(data.positions), x, pi, beta)
    state = experiment.initialize(0, problem)._replace(
        environment=two_room.State(jnp.array(position))
    )
    updated, _ = experiment.experience_step(state, 3, state.key, state.key, problem)
    # rho=.7/.25=2.8, alpha=.1; no subtask bonus on hallway arrival.
    assert jnp.allclose(
        updated.learners.model.reward_weights[0], 0.28 * reward * x[s], atol=1e-7
    )
    expected = (
        jnp.zeros((98, 98)) if done else 0.28 * 0.99 * 0.3 * jnp.outer(x[ns], x[s])
    )
    assert jnp.allclose(
        updated.learners.model.successor_weights[0], expected, atol=1e-7
    )
    assert int(updated.episodes) == int(done)
    assert tuple(updated.environment.position.tolist()) == (
        (3, 1) if done else destination
    )


def test_tile_scan_is_reproducible_and_vmap_preserves_seed_axis(data):
    pi = jnp.full((1, 73, 4), 0.25)
    problem = experiment.Problem(
        data.params,
        jnp.array(data.positions),
        experiment.features(data, "tile_coding"),
        pi,
        jnp.ones((1, 73)),
    )

    def run(seed):
        return experiment.train_block(
            experiment.initialize(seed, problem), problem, steps=128
        )

    a, b = run(2000), run(2000)
    batched = jax.vmap(run)(jnp.array([2000, 2001]))
    for first, second, many in zip(
        a.learners.model, b.learners.model, batched.learners.model, strict=True
    ):
        assert jnp.array_equal(first, second)
        assert jnp.allclose(first, many[0], atol=1e-6)
        assert jnp.isfinite(many).all()
        assert not jnp.array_equal(many[0], many[1])


def test_frozen_options_do_not_depend_on_model_representation(data):
    pi, beta, weights = experiment.freeze_options(data, seed=100, steps=100)
    assert pi.shape == (7, 73, 4)
    assert beta.shape == (7, 73)
    assert weights["one_hot"]["seed"] == weights["tile_coding"]["seed"] == 100
    for name in ("one_hot", "tile_coding"):
        problem = experiment.Problem(
            data.params,
            jnp.array(data.positions),
            experiment.features(data, name),
            pi,
            beta,
        )
        assert problem.probabilities is pi
        assert problem.stopping is beta
        assert jnp.all(problem.features[jnp.array(data.terminal)] == 0)
    assert jnp.allclose(pi.sum(axis=-1), 1)
    assert jnp.all(beta[:, jnp.array(data.terminal)] == 1)


def test_decoder_and_independent_feature_space_references(data):
    tables = {
        name: experiment.features(data, name) for name in ("one_hot", "tile_coding")
    }
    option = data.options["shortest_path"]
    pi = jax.nn.one_hot(jnp.array(option.policy), 4)[None]
    beta = jnp.array(option.stopping)[None]
    refs, agreement = build_references(data, pi, beta, tables)
    assert agreement["successor_max_error"] < 1e-8
    for name, x in tables.items():
        d, metadata = decoder(x, data.nonterminal_indices)
        assert metadata["rank"] == 72
        assert np.allclose(
            np.asarray(refs[name]["successor"]) @ d,
            refs["one_hot"]["successor"],
            atol=1e-10,
        )
    with pytest.raises(ValueError, match="full row rank"):
        decoder(jnp.ones((3, 1)), (0, 1, 2))


def test_model_errors_measure_predictions_with_correct_matrix_orientation():
    # Two source states, three features. Unused feature weights must not affect error.
    x = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    r = jnp.array([2.0, 6.0, 999.0])
    n = jnp.array([[2.0, 8.0, 999.0], [1.0, 6.0, 777.0], [0.0, 0.0, 555.0]])
    d = jnp.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    rt = jnp.array([[2.0, 3.0]])
    nt = jnp.array([[[2.0, 1.0, 0.0], [4.0, 3.0, 0.0]]])
    ct = jnp.array([[[2.0, 2.0], [4.0, 6.0]]])
    model = LinearExpectationModel(r[None, None], n[None, None])
    errors = model_errors(model, x, rt, nt, d, ct)
    assert errors.shape == (1, 1, 3)
    assert jnp.array_equal(errors, jnp.zeros_like(errors))
    perturbed = model._replace(reward_weights=model.reward_weights.at[0, 0, 0].add(2))
    assert float(model_errors(perturbed, x, rt, nt, d, ct)[0, 0, 0]) == pytest.approx(
        2**0.5
    )


def test_rollout_audit_accumulates_both_feature_maps_and_second_moments(data):
    option = data.options["shortest_path"]
    pi = jax.nn.one_hot(jnp.array(option.policy), 4).tolist()
    x = np.concatenate(
        [experiment.features(data, name) for name in ("one_hot", "tile_coding")], axis=1
    )
    result = sample_model(
        data.positions,
        pi,
        option.stopping,
        (3, 1),
        1,
        samples=16,
        feature_rows=x.tolist(),
    )
    target = 0.99**6 * x[data.positions.index((3, 7))]
    assert result["reward_mean"] == pytest.approx(-sum(0.99**t for t in range(4)))
    assert np.allclose(result["successor_mean"], target)
    assert max(result["successor_se"]) < 1e-8
    terminal = sample_model(
        data.positions,
        pi,
        option.stopping,
        (6, 10),
        1,
        samples=16,
        feature_rows=x.tolist(),
    )
    assert terminal["successor_mean"] == [0.0] * 170


def test_paired_summary_and_deterministic_serialization(tmp_path):
    history = np.zeros((2, 3, 7, 3))
    history[1, :, 0, 0] = [1, 2, 3]
    stats = summarize(history)["up"]["reward_rmse"]
    assert stats["mean"] == [0, 2]
    assert stats["se"][1] == pytest.approx(1 / np.sqrt(3))
    first = save_json(
        tmp_path / "a.json.gz", {"history": history.tolist()}, compressed=True
    )
    second = save_json(
        tmp_path / "b.json.gz", {"history": history.tolist()}, compressed=True
    )
    assert first["sha256"] == second["sha256"]


@pytest.mark.parametrize("name", ["one_hot", "tile_coding"])
def test_short_model_comparison_preserves_all_seeds_and_checkpoints(
    data, name, tmp_path, monkeypatch
):
    tables = {n: experiment.features(data, n) for n in ("one_hot", "tile_coding")}
    pi = jnp.full((7, 73, 4), 0.25)
    beta = jnp.ones((7, 73))
    refs, _ = build_references(data, pi, beta, tables)
    monkeypatch.setattr(reporting, "SEEDS", (2000, 2001))
    monkeypatch.setattr(reporting, "STEPS", 20)
    monkeypatch.setattr(reporting, "CHECKPOINT", 10)
    case, history = reporting.train_case(
        data, name, pi, beta, tables, refs, tmp_path, {}
    )
    assert history.shape == (3, 2, 7, 3)
    assert case["finite"] and case["terminal_predictions_zero"]
    assert len(case["summary"]["tile_option"]["reward_rmse"]["mean"]) == 3
    assert len(case["checkpoints"]) == 1
    assert np.isfinite(history).all()
    assert not np.array_equal(history[-1, 0], history[-1, 1])
