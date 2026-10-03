import json
import math

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments.hallway import build_baselines
from rrs.experiments.learning_report import build_report, mean_and_se
from rrs.experiments.option_learning import TrainingResult, initialize, training_problem
from rrs.rl.learning import LearningParameters


def test_initial_policy_report_separates_estimates_from_actual_returns():
    data = build_baselines()
    states = jax.vmap(lambda seed: initialize(seed, training_problem()))(
        jnp.array([0, 1])
    )
    result = TrainingResult(states, jnp.zeros((2, 2, 72)))
    report = build_report(data, result, (0, 1), 1, 1, LearningParameters())
    expected_rmse = math.sqrt(
        sum(
            data.solutions["reward_respecting"].values[i] ** 2
            for i in data.nonterminal_indices
        )
        / 72
    )
    assert report["summary"]["critic_rmse"]["mean"] == pytest.approx(
        [expected_rmse] * 2, abs=1e-6
    )
    assert report["summary"]["estimated_start_value"]["mean"] == [0.0, 0.0]
    assert report["summary"]["stochastic_start_value"]["mean"] == -0.25
    assert report["summary"]["safe_hallway_probability"]["mean"] == 0.0
    assert not report["canonical_protocol"]
    assert not report["acceptance"]["mean_start_estimate_within_0.15_of_optimal"]
    assert not report["acceptance"]["mean_safe_hallway_probability_at_least_0.80"]
    assert json.loads(json.dumps(report, allow_nan=False))["seeds"] == [0, 1]


def test_standard_error_uses_independent_runs():
    assert mean_and_se([1.0, 3.0]) == {"mean": 2.0, "se": 1.0}
    assert mean_and_se([2.0]) == {"mean": 2.0, "se": 0.0}
