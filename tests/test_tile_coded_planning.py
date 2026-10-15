import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rrs.experiments import tile_coded_planning_inputs as inputs
from rrs.experiments.hallway import build_baselines
from rrs.experiments.learning_report import reference_model
from rrs.experiments.planning import run_plan, sample_states
from rrs.experiments.tile_coded_model_report import build_references, save_json
from rrs.experiments.tile_coded_models import NAMES, PARAMETERS, features
from rrs.experiments.tile_coded_planning import (
    explicit_nonfinite,
    hit_summary,
    plan_batch,
    policies_with_validity,
    report_case,
    summary_columns,
)
from rrs.experiments.tile_coded_planning_audit import state_space_replay
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.plan_evaluation import induced_policy
from rrs.rl.planning import planning_step


def toy_models(n=4):
    return LinearExpectationModel(
        jnp.array([[0.0, 1.0], [-1.0, 0.0]] + [[-2.0, -2.0]] * (n - 2)),
        jnp.zeros((n, 2, 2)).at[0, 1, 0].set(0.5),
    )


def test_explicit_one_hot_features_preserve_original_runner():
    models = toy_models()
    order = sample_states(12000, 2, 50)
    old = run_plan(models, order, 200, 100)
    new = run_plan(models, order, 200, 100, state_features=jnp.eye(2))
    assert jnp.array_equal(old.history, new.history)
    assert int(new.lookaheads) == 200 and int(new.updates) == 50
    with pytest.raises(ValueError, match="input dimension"):
        run_plan(models, order, 200, 100, state_features=jnp.ones((2, 3)))


def test_overlapping_feature_backup_and_no_extra_discount():
    x = jnp.array([[0.5, 0.5], [0.5, -0.5]])
    models = LinearExpectationModel(
        jnp.array([[1.0, 2.0], [-1.0, 0.0]]),
        jnp.array([[[0.2, 0.4], [0.6, 0.8]], [[0.0, 0.0], [0.0, 0.0]]]),
    )
    weights = jnp.array([2.0, 3.0])
    # q0 = 1.5 + [2,3] dot [.3,.7] = 4.2; V=2.5; delta=1.7
    updated, info = planning_step(weights, x[0], models)
    assert updated.tolist() == pytest.approx([2.85, 3.85], abs=1e-6)
    assert info.action_values.tolist() == pytest.approx([4.2, -0.5], abs=1e-6)
    order = jnp.array([0, 1, 1, 0])
    expected = jnp.zeros(2)
    for state in order:
        expected, _ = planning_step(expected, x[state], models)
    result = run_plan(models, order, 8, 4, state_features=x)
    assert jnp.allclose(result.weights, expected, atol=1e-6)
    zero, _ = planning_step(weights, jnp.zeros(2), models)
    assert jnp.array_equal(zero, weights)


def test_batched_plans_pair_state_prefix_and_spend_equal_queries():
    order = jax.vmap(lambda seed: sample_states(seed, 2, 50))(jnp.array([12000, 12001]))
    results = []
    for n in (4, 5):
        models = jax.tree.map(lambda x: jnp.stack([x, x]), toy_models(n))
        result = plan_batch(models, order, jnp.eye(2), budget=200, checkpoint=100)
        again = plan_batch(models, order, jnp.eye(2), budget=200, checkpoint=100)
        assert jnp.array_equal(result.history, again.history)
        assert result.lookaheads.tolist() == [200, 200]
        results.append(result)
    assert results[0].updates.tolist() == [50, 50]
    assert results[1].updates.tolist() == [40, 40]


def test_policy_reselection_matches_existing_helper_and_rejects_nonfinite():
    models = toy_models()
    x = jnp.eye(2)
    pi = jnp.broadcast_to(jnp.eye(4)[:, None, :], (4, 2, 4))
    w = jnp.array([0.5, 1.0])
    expected, choice = induced_policy(w, models, pi, x)
    many = jax.tree.map(lambda a: jnp.stack([a, a]), models)
    policies, choices, valid = policies_with_validity(
        jnp.stack([w, w * jnp.nan]), many, pi, x
    )
    assert jnp.array_equal(policies[0], expected)
    assert jnp.array_equal(choices[0], choice)
    assert valid.tolist() == [True, False]


def test_missing_metrics_and_nonfinite_parameters_are_explicit():
    assert summary_columns([[1.0, 2.0], [3.0, None]]) == {
        "mean": [2.0, None],
        "se": [1.0, None],
    }
    hit = hit_summary([[0.0, 1.0], [0.0, None]], [0, 100], 0.9)
    assert hit == {"per_seed": [100, None], "reached": 1, "total": 2, "mean": None}
    assert explicit_nonfinite([float("inf"), float("-inf"), float("nan"), 1.0]) == [
        "inf",
        "-inf",
        "nan",
        1.0,
    ]


def test_independent_state_replay_propagates_updates_through_feature_overlap():
    rewards = np.array([[1.0, 0.0], [-1.0, -1.0]])
    successor = np.zeros((2, 2, 2))
    successor[0, 1, 0] = 0.5
    gram = np.array([[1.0, 0.5], [0.5, 1.0]])
    replay = state_space_replay(rewards, successor, gram, [0, 1, 0], (0, 2, 4, 6))
    # First backup sets V0=1 and changes V1 by .5; next targets .5, so no change.
    assert np.array_equal(replay, [[0.0, 0.0], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]])
    with pytest.raises(ValueError, match="cover"):
        state_space_replay(rewards, successor, gram, [0], (0, 2, 4))


@pytest.fixture(scope="module")
def source_data():
    data = build_baselines()
    tables = {n: features(data, n) for n in ("one_hot", "tile_coding")}
    pi = np.full((7, 73, 4), 0.25)
    beta = np.ones((7, 73))
    for i, name in enumerate(NAMES[:5]):
        option = data.options[name]
        pi[i] = np.eye(4)[list(option.policy)]
        pi[i, np.array(data.terminal)] = 0.25
        beta[i] = np.where(data.terminal, 1, option.stopping)
    options = {
        "positions": [list(p) for p in data.positions],
        "model_names": list(NAMES),
        "option_parameters": {
            "alpha": 0.1,
            "alpha_actor": 0.1,
            "gamma": 0.99,
            "lambda_": 0.0,
            "lambda_actor": 0.0,
        },
        "probabilities": pi.tolist(),
        "stopping": beta.tolist(),
        "trained_options": {
            n: {
                "seed": 100,
                "steps": 200000,
                "critic_weights": [0.0] * x.shape[1],
                "actor_weights": np.zeros((4, x.shape[1])).tolist(),
            }
            for n, x in tables.items()
        },
    }
    refs, _ = build_references(data, jnp.array(pi), jnp.array(beta), tables)
    return data, tables, options, refs


def make_source(directory, fixture):
    data, tables, options, refs = fixture
    source = save_json(directory / "frozen_options.json", options)
    report = {
        "milestone": "11-tile-coded-models",
        "seeds": [2000, 2001],
        "steps": 50000,
        "parameters": PARAMETERS._asdict(),
        "model_names": list(NAMES),
        "implementation_checks": dict.fromkeys(
            ("finite", "terminal_predictions_zero", "oracle_bounds", "rollout_audits"),
            True,
        ),
        "frozen_options": source,
        "provenance": {},
        "references": refs,
        "cases": {},
    }
    for name, x in tables.items():
        entries = []
        for step in inputs.MODEL_STEPS:
            payload = {
                "schema_version": 1,
                "step": step,
                "seeds": [2000, 2001],
                "model_names": list(NAMES),
                "parameters": PARAMETERS._asdict(),
                "frozen_options": source,
                "positions": [list(p) for p in data.positions],
                "features": x.tolist(),
                "reward_axes": ["run", "option", "input_feature"],
                "successor_axes": ["run", "option", "output_feature", "input_feature"],
                "reward_weights": np.zeros((2, 7, x.shape[1])).tolist(),
                "successor_weights": np.zeros((2, 7, x.shape[1], x.shape[1])).tolist(),
            }
            entries.append(
                save_json(
                    directory / f"{name}_step_{step:05d}.json.gz",
                    payload,
                    compressed=True,
                )
            )
        report["cases"][name] = {"features": x.tolist(), "checkpoints": entries}
    save_json(directory / "learning.json", report)
    return report


def test_source_loading_and_exact_model_reconstruction(
    source_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(inputs, "SEEDS", (2000, 2001))
    make_source(tmp_path, source_data)
    data = source_data[0]
    loaded = inputs.load_inputs(tmp_path, data)
    models, checks = inputs.exact_models(data, loaded)
    assert set(loaded["checkpoints"]["tile_coding"]) == set(inputs.MODEL_STEPS)
    assert checks["reconstruction"]["tile_coding"]["double_error"] <= 1e-10
    assert checks["reconstruction"]["tile_coding"]["float32_error"] <= 1e-5
    assert models["tile_coding"].successor_weights.shape == (7, 98, 98)


def test_corrupt_existing_input_takes_priority_over_missing_input(
    source_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(inputs, "SEEDS", (2000, 2001))
    make_source(tmp_path, source_data)
    (tmp_path / "one_hot_step_00000.json.gz").unlink()
    corrupt = tmp_path / "tile_coding_step_50000.json.gz"
    original = corrupt.read_bytes()
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        inputs.load_inputs(tmp_path, source_data[0])
    corrupt.write_bytes(original)
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        inputs.load_inputs(tmp_path, source_data[0])


def test_checkpoint_geometry_and_negative_coefficients(source_data, tmp_path):
    import gzip

    make_source(tmp_path, source_data)
    data, tables, _, _ = source_data
    p = json.loads(
        gzip.decompress((tmp_path / "tile_coding_step_00000.json.gz").read_bytes())
    )
    p["successor_weights"][0][0][0][0] = -2.0
    model = inputs.validate_checkpoint(
        p, data, tables["tile_coding"], 0, p["frozen_options"], seeds=(2000, 2001)
    )
    assert model.successor_weights[0, 0, 0, 0] == -2.0
    invalid = copy.deepcopy(p)
    invalid["features"][0][0] = 999.0
    with pytest.raises(ValueError, match="features"):
        inputs.validate_checkpoint(
            invalid,
            data,
            tables["tile_coding"],
            0,
            p["frozen_options"],
            seeds=(2000, 2001),
        )
    p["reward_weights"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        inputs.validate_checkpoint(
            p, data, tables["tile_coding"], 0, p["frozen_options"], seeds=(2000, 2001)
        )


def test_report_retains_invalid_run_and_does_not_evaluate_nan_policy(source_data):
    data, tables, options, _ = source_data
    x = tables["one_hot"]
    models = LinearExpectationModel(jnp.zeros((2, 4, 72)), jnp.zeros((2, 4, 72, 72)))
    orders = jnp.zeros((2, 5), dtype=int)
    result = plan_batch(
        models, orders, x[jnp.array(data.nonterminal_indices)], budget=20, checkpoint=20
    )
    result = result._replace(history=result.history.at[1, 1].set(jnp.nan))
    case = report_case(
        data,
        models,
        result,
        x,
        jnp.array(options["probabilities"])[:4],
        (0, 1, 2, 3),
        reference_model(data.positions),
        [0.0] * 73,
        seeds=(12000, 12001),
        checkpoints=(0, 20),
        budgets=(0, 20),
    )
    assert len(case["runs"]) == 2 and case["evaluation_failures"] == 1
    assert case["runs"][1]["actual_return"][-1] is None
    assert case["summary"]["actual_return"]["mean"][-1] is None
    assert "Nonfinite" in case["runs"][1]["evaluation_errors"][-1]
