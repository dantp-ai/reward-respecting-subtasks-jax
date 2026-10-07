import gzip
import hashlib
import json

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments.four_room_option_report import (
    acceptance_checks,
    build_references,
    policy_tables,
    save_snapshot,
)
from rrs.experiments.four_room_options import (
    OPTION_NAMES,
    STEPS,
    initialize,
    training_problem,
)


@pytest.fixture(scope="module")
def references():
    return build_references()


def test_stochastic_subtask_references_and_equation_nine_agree(references):
    data = references
    for solution, evaluation, z in zip(
        data.solutions, data.evaluations, data.stopping_values, strict=True
    ):
        assert solution.bellman_residual <= 1e-12
        assert evaluation.values == pytest.approx(solution.values, abs=2e-8)
        assert solution.stopping == tuple(
            done or stop >= value
            for done, stop, value in zip(data.terminal, z, solution.values, strict=True)
        )
        assert evaluation.value_error_bound <= 1e-8
        assert evaluation.probability_gap <= 1e-8


def test_snapshot_exports_all_options_with_explicit_axes_and_hash(tmp_path, references):
    problem = training_problem()
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array([7000, 7001]))
    path = tmp_path / "options.json.gz"
    first = save_snapshot(path, state, 0, [7000, 7001], references)
    encoded = path.read_bytes()
    payload = json.loads(gzip.decompress(encoded))
    assert first["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert payload["option_names"] == list(OPTION_NAMES)
    assert payload["actor_axes"] == ["run", "option", "action", "feature"]
    assert jnp.array(payload["actor_weights"]).shape == (2, 4, 4, 103)
    assert jnp.array(payload["policy_probabilities"]).shape == (2, 4, 104, 4)
    assert all(all(row) for row in payload["stopping"][0])
    save_snapshot(path, state, 0, [7000, 7001], references)
    assert path.read_bytes() == encoded
    weights = jax.tree.map(lambda x: x[0], state.weights)
    pi, beta = policy_tables(
        weights,
        jnp.array(references.features),
        jnp.array(references.stopping_values),
        jnp.array(references.terminal),
    )
    assert jnp.all(pi == 0.25) and jnp.all(beta)


def test_each_option_must_pass_no_cross_option_average():
    summary = {
        name: {
            "critic_rmse": {"mean": [0.8, 0.2]},
            "estimated_start": {"mean": [0.0, 0.8]},
            "evaluations": {
                str(STEPS): {
                    "actual_start": {"mean": 0.8},
                    "actual_value_rmse": {"mean": 0.2},
                }
            },
        }
        for name in OPTION_NAMES
    }
    assert all(acceptance_checks(summary, [0.85] * 4).values())
    summary["H3"]["evaluations"][str(STEPS)]["actual_start"]["mean"] = 0.5
    checks = acceptance_checks(summary, [0.85] * 4)
    assert not checks["H3/actual_start_within_0.15"]
    assert checks["H1/actual_start_within_0.15"]
