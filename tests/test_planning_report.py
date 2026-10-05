"""Protect query accounting and the distinction between predicted/actual return."""

import copy

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments.hallway import build_baselines
from rrs.experiments.learning_report import reference_model
from rrs.experiments.planning import PlanningCase, run_plan
from rrs.experiments.planning_report import acceptance_checks, case_report, first_hit
from rrs.rl.exact import value_iteration
from rrs.rl.option_models import LinearExpectationModel


def test_first_hit_includes_initial_state_and_preserves_missing_hits():
    assert first_hit([0, 0.8, 0.9], [0, 100, 200], 0.8) == 100
    assert first_hit([0, 0.8, 0.9], [0, 100, 200], 0.95) is None
    assert first_hit([1, 1], [0, 100], 0.95) == 0


def test_report_separates_model_optimism_from_actual_closed_loop_return():
    data = build_baselines()
    n = len(data.nonterminal_indices)
    # A fabricated UP model promises reward 1; the true UP policy stays in a loop.
    models = LinearExpectationModel(jnp.ones((1, 1, n)), jnp.zeros((1, 1, n, n)))
    pi = jax.nn.one_hot(jnp.zeros((1, len(data.positions)), dtype=int), 4)
    case = PlanningCase("learned_00000", "up", models, pi)
    start = data.nonterminal_indices.index(data.positions.index((3, 1)))
    result = jax.vmap(lambda m, states: run_plan(m, states, 2, 1))(
        models, jnp.array([[start, start]])
    )
    environment = reference_model(data.positions)
    report = case_report(
        data,
        case,
        result,
        [2000],
        [1000],
        environment,
        value_iteration(environment).values,
        [0, 1, 2],
        [0, 2],
    )
    assert report["planning_lookaheads_per_run"] == 2
    assert report["planning_updates_per_run"] == 2
    assert report["diagnostic_lookaheads_per_run"] == 2 * len(data.positions)
    assert report["summary"]["planned_start"]["mean"] == [0, 1, 1]
    assert report["summary"]["actual_return"]["mean"] == [0, 0]
    assert report["summary"]["first_95_percent"]["mean"] == 1
    assert report["finite"] and report["terminal_zero"]


@pytest.fixture
def passing_cases():
    cases = {}
    for source in ("exact_optimal", "exact_frozen", "learned_10000", "learned_50000"):
        for name in ("primitives", "shortest_path", "reward_respecting"):
            if source == "exact_frozen" and name != "reward_respecting":
                continue
            cases[f"{source}/{name}"] = {
                "source": source,
                "finite": True,
                "terminal_zero": True,
                "summary": {
                    "planned_start": {"mean": [0, 0.84]},
                    "actual_return": {"mean": [0, 0.84]},
                    "final_max_value_error": 1e-7,
                    "final_actual_error": 1e-7,
                    "evaluation_error_bound": 1e-12,
                    "first_95_percent": {
                        "mean": 1000 if name == "reward_respecting" else 2000,
                        "reached": 100,
                        "total": 100,
                    },
                    "final_start_absolute_error": {
                        "mean": 0.1 if source == "learned_10000" else 0.01
                    },
                },
            }
    return cases


def test_acceptance_applies_frozen_accuracy_and_efficiency_limits(passing_cases):
    assert all(acceptance_checks(passing_cases, 0.99**17).values())
    cases = copy.deepcopy(passing_cases)
    cases["exact_optimal/reward_respecting"]["summary"]["first_95_percent"]["mean"] = (
        1700
    )
    checks = acceptance_checks(cases, 0.99**17)
    assert not checks["exact_reward_respecting_at_most_80_percent_of_primitives"]
    cases["learned_50000/primitives"]["summary"]["actual_return"]["mean"][-1] = 0
    assert not acceptance_checks(cases, 0.99**17)[
        "learned_50000/primitives/actual_return_within_0.05"
    ]


def test_missing_hit_fails_without_dropping_unsuccessful_runs(passing_cases):
    case = passing_cases["exact_optimal/reward_respecting"]
    case["summary"]["first_95_percent"] = {"mean": None, "reached": 99, "total": 100}
    checks = acceptance_checks(passing_cases, 0.99**17)
    assert not checks["exact_optimal/reward_respecting/all_reach_95_percent"]
    assert not checks["exact_reward_respecting_at_most_80_percent_of_primitives"]
