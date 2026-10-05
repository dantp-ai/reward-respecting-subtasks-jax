import gzip
import hashlib
import json

import jax.numpy as jnp
import pytest

from rrs.experiments.model_learning import MODEL_NAMES
from rrs.experiments.planning_inputs import MODEL_STEPS, load_inputs


@pytest.fixture
def artifacts(tmp_path):
    features, seeds = [[1, 1], [1, 2]], [1000, 1001]
    pi = jnp.concatenate(
        (jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, 2)), jnp.full((2, 4, 2), 0.25))
    )
    report = {
        "feature_positions": features,
        "model_names": list(MODEL_NAMES),
        "seeds": seeds,
        "steps": 50_000,
        "canonical_protocol": True,
        "learning_parameters": {"gamma": 0.99},
        "code_revision": "test-fixture",
        "tracked_worktree_dirty": False,
        "frozen_option": {
            "all_option_probability_weights": pi.tolist(),
            "all_option_stopping_weights": jnp.ones((6, 2)).tolist(),
        },
        "model_checkpoints": [],
    }
    for step in MODEL_STEPS:
        payload = {
            "schema_version": 1,
            "step": step,
            "gamma": 0.99,
            "seeds": seeds,
            "model_names": list(MODEL_NAMES),
            "feature_positions": features,
            "reward_axes": ["run", "option", "input_feature"],
            "successor_axes": ["run", "option", "output_feature", "input_feature"],
            "reward_weights": jnp.zeros((2, 6, 2)).tolist(),
            "successor_weights": jnp.zeros((2, 6, 2, 2)).tolist(),
        }
        path = tmp_path / f"models_step_{step:05d}.json.gz"
        encoded = gzip.compress(json.dumps(payload).encode(), mtime=0)
        path.write_bytes(encoded)
        report["model_checkpoints"].append(
            {
                "step": step,
                "path": str(path),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    (tmp_path / "learning.json").write_text(json.dumps(report))
    return tmp_path, features, seeds


def test_loader_preserves_model_axes_and_source_provenance(artifacts):
    directory, features, seeds = artifacts
    result = load_inputs(directory, features, seeds)
    assert tuple(result.checkpoints) == MODEL_STEPS
    assert result.checkpoints[50_000].successor_weights.shape == (2, 6, 2, 2)
    assert result.provenance["model_code_revision"] == "test-fixture"
    assert result.provenance["model_seeds"] == seeds


def test_corrupted_checkpoint_is_rejected_before_parsing(artifacts):
    directory, features, seeds = artifacts
    path = directory / "models_step_10000.json.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(directory, features, seeds)


@pytest.mark.parametrize(
    "change, message", [("axes", "successor_axes"), ("mass", "successor mass")]
)
def test_incompatible_axes_and_undiscounted_successors_are_rejected(
    artifacts, change, message
):
    directory, features, seeds = artifacts
    path = directory / "models_step_00000.json.gz"
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if change == "axes":
        payload["successor_axes"][-2:] = ["input_feature", "output_feature"]
    else:
        payload["successor_weights"][0][0][0][0] = 1.0
    encoded = gzip.compress(json.dumps(payload).encode(), mtime=0)
    path.write_bytes(encoded)
    report_path = directory / "learning.json"
    report = json.loads(report_path.read_text())
    report["model_checkpoints"][0]["sha256"] = hashlib.sha256(encoded).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        load_inputs(directory, features, seeds)


def test_feature_order_must_match_the_current_representation(artifacts):
    directory, features, seeds = artifacts
    with pytest.raises(ValueError, match="order"):
        load_inputs(directory, features[::-1], seeds)
