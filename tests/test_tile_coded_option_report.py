import gzip
import json

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments import tile_coded_option_report as reporting
from rrs.experiments.hallway import build_baselines
from rrs.experiments.option_learning import TrainingResult
from rrs.experiments.tile_coded_options import initialize, training_problem


def initial_case(data, name):
    problem, encode = training_problem(name)
    states = jax.vmap(lambda seed: initialize(seed, problem, encode))(jnp.array([0, 1]))
    result = TrainingResult(
        states, jnp.repeat(states.weights.critic[:, None, :], 2, axis=1)
    )
    return reporting.case_report(data, name, problem, encode, result, (0, 1)), result


def test_state_predictions_report_pairing_and_artifact_roundtrip(tmp_path):
    data = build_baselines()
    cases, results, artifacts = {}, {}, {}
    for name in ("one_hot", "tile_coding"):
        cases[name], results[name] = initial_case(data, name)
        assert cases[name]["summary"]["estimated_start_value"]["mean"] == [0.0, 0.0]
        assert cases[name]["summary"]["stochastic_start_value"]["mean"] == -0.25
        assert cases[name]["summary"]["safe_hallway_probability"]["mean"] == 0.0
        path = tmp_path / f"{name}.json.gz"
        artifacts[name] = reporting.save_training(path, results[name], (0, 1))
        encoded = path.read_bytes()
        assert reporting.save_training(path, results[name], (0, 1)) == artifacts[name]
        assert path.read_bytes() == encoded
        payload = json.loads(gzip.decompress(encoded))
        assert (
            payload["final_weights_and_traces"]["critic"]
            == results[name].final.weights.critic.tolist()
        )
    assert (
        cases["tile_coding"]["summary"]["critic_rmse"]
        == cases["one_hot"]["summary"]["critic_rmse"]
    )
    assert cases["one_hot"]["representation"]["overlapping_state_pairs"] == 0
    assert cases["tile_coding"]["representation"]["overlapping_state_pairs"] > 0
    report = reporting.build_report(
        data, cases, results, artifacts, seeds=(0, 1), steps=1, checkpoint=1
    )
    assert not report["canonical_protocol"]
    assert report["acceptance"]["paired_final_environment_keys_and_episodes"]
    assert not all(
        report["acceptance"].values()
    )  # untrained control fails M3's thresholds
    assert all(v["mean"] == 0 for v in report["paired_differences"].values())
    json.dumps(report, allow_nan=False)


def test_failed_evaluation_is_preserved_and_not_dropped_from_means(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unresolved probability interval")

    monkeypatch.setattr(reporting, "evaluate_policy", fail)
    case, _ = initial_case(build_baselines(), "tile_coding")
    assert len(case["runs"]) == 2
    assert case["summary"]["stochastic_start_value"]["mean"] is None
    assert all(
        r["evaluation_failure"] == "unresolved probability interval"
        for r in case["runs"]
    )
    assert not case["acceptance"]["all_evaluations_bounded"]


def test_tile_metrics_project_weights_to_states_instead_of_indexing_them():
    data = build_baselines()
    problem, encode = training_problem("tile_coding")
    states = jax.vmap(lambda seed: initialize(seed, problem, encode))(jnp.array([0]))
    critic = jnp.full_like(states.weights.critic, 0.2)
    states = states._replace(weights=states.weights._replace(critic=critic))
    result = TrainingResult(states, jnp.stack((jnp.zeros_like(critic), critic), axis=1))
    case = reporting.case_report(data, "tile_coding", problem, encode, result, (0,))
    # Four active .5 features: each nonterminal prediction is .4, not .2.
    assert case["summary"]["estimated_start_value"]["mean"] == pytest.approx([0, 0.4])
    oracle = data.solutions["reward_respecting"].values
    expected = (
        sum((0.4 - oracle[s]) ** 2 for s in data.nonterminal_indices) / 72
    ) ** 0.5
    assert case["summary"]["critic_rmse"]["mean"][-1] == pytest.approx(
        expected, abs=1e-6
    )
    terminal = data.positions.index((6, 10))
    assert case["runs"][0]["predicted_values"][terminal] == 0.0


def test_paired_standard_error_uses_differences_and_rejects_different_seeds():
    def runs(values):
        return [
            {
                "seed": i,
                "critic_rmse": [v],
                "estimated_start_value": [v],
                "stochastic_start_value": v,
                "safe_hallway_probability": v,
            }
            for i, v in enumerate(values)
        ]

    cases = {
        "one_hot": {"runs": runs([1.0, 3.0])},
        "tile_coding": {"runs": runs([2.0, 4.0])},
    }
    for stats in reporting.paired_differences(cases).values():
        assert stats == {"mean": 1.0, "se": 0.0, "per_seed": [1.0, 1.0]}
    cases["tile_coding"]["runs"].reverse()
    with pytest.raises(ValueError, match="same seed"):
        reporting.paired_differences(cases)
