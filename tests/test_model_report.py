import gzip
import json
import math

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments.hallway import build_baselines
from rrs.experiments.model_learning import freeze_options, initialize
from rrs.experiments.model_report import build_references, model_rmse, save_checkpoint
from rrs.experiments.option_learning import training_problem
from rrs.rl.learning import initial_weights
from rrs.rl.option_models import LinearExpectationModel


def test_successor_metric_does_not_dilute_vector_error_by_feature_count():
    model = LinearExpectationModel(
        jnp.array([2.0, 4.0]), jnp.array([[1.0, 2.0], [3.0, 4.0]])
    )
    zero = LinearExpectationModel(jnp.zeros(2), jnp.zeros((2, 2)))
    errors = model_rmse(model, zero)
    assert float(errors.reward) == pytest.approx(math.sqrt(10), abs=1e-6)
    assert float(errors.successor) == pytest.approx(math.sqrt(15), abs=1e-6)


def test_references_for_uniform_immediately_stopping_option_are_one_step_models():
    data = build_baselines()
    options = freeze_options(data, initial_weights(72, 4))
    oracles, targets = build_references(data, options)
    assert max(oracle.error_bound for oracle in oracles) <= 1e-10
    for j, s in enumerate(data.nonterminal_indices):
        assert float(targets.reward_weights[4, j]) == pytest.approx(
            sum(data.model.rewards[s]) / 4, abs=1e-6
        )
        expected = [
            sum(
                0.99 * data.features[ns][feature] / 4
                for ns in data.model.next_states[s]
            )
            for feature in range(72)
        ]
        assert targets.successor_weights[4, :, j].tolist() == pytest.approx(
            expected, abs=1e-6
        )


def test_checkpoint_preserves_axes_features_seeds_and_is_byte_reproducible(tmp_path):
    data, problem = build_baselines(), training_problem()
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array([1000]))
    model = state.learners.model._replace(
        successor_weights=state.learners.model.successor_weights.at[0, 4, 7, 8].set(0.5)
    )
    path = tmp_path / "models.json.gz"
    first = save_checkpoint(path, model, 0, (1000,), data)
    second = save_checkpoint(path, model, 0, (1000,), data)
    assert first["sha256"] == second["sha256"]
    saved = json.loads(gzip.decompress(path.read_bytes()))
    assert saved["successor_weights"][0][4][7][8] == 0.5
    assert saved["successor_axes"] == [
        "run",
        "option",
        "output_feature",
        "input_feature",
    ]
    assert saved["seeds"] == [1000]
    assert saved["feature_positions"] == [
        list(data.positions[i]) for i in data.nonterminal_indices
    ]
