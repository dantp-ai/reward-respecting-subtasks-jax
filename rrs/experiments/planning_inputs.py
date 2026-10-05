"""Validate and load the Milestone 4 models used in planning comparisons."""

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from rrs.experiments.model_learning import MODEL_NAMES, FixedOptions
from rrs.rl.option_models import LinearExpectationModel

MODEL_STEPS = (0, 10_000, 20_000, 50_000)


@dataclass(frozen=True)
class PlanningInputs:
    options: FixedOptions
    checkpoints: dict[int, LinearExpectationModel]
    provenance: dict


def _checkpoint(payload, features, seeds):
    expected = {
        "schema_version": 1,
        "gamma": 0.99,
        "seeds": list(seeds),
        "model_names": list(MODEL_NAMES),
        "feature_positions": features,
        "reward_axes": ["run", "option", "input_feature"],
        "successor_axes": ["run", "option", "output_feature", "input_feature"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the planning contract")
    model = LinearExpectationModel(
        jnp.array(payload["reward_weights"], dtype=jnp.float32),
        jnp.array(payload["successor_weights"], dtype=jnp.float32),
    )
    shape = (len(seeds), len(MODEL_NAMES), len(features))
    if model.reward_weights.shape != shape or model.successor_weights.shape != shape + (
        len(features),
    ):
        raise ValueError("Checkpoint array dimensions do not match its metadata")
    if not all(bool(jnp.isfinite(x).all()) for x in model):
        raise ValueError("Checkpoint model coefficients must be finite")
    if bool(jnp.any(model.successor_weights < 0)) or bool(
        jnp.any(jnp.sum(model.successor_weights, axis=-2) > 0.99 + 1e-6)
    ):
        raise ValueError(
            "Checkpoint successor mass must be nonnegative and at most gamma"
        )
    return model


def load_inputs(
    directory: Path, feature_positions, seeds=tuple(range(1000, 1100))
) -> PlanningInputs:
    """Require matching feature/seed order and content hashes before using a model."""
    features = [list(p) for p in feature_positions]
    report_path = directory / "learning.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (
        report.get("feature_positions") != features
        or report.get("model_names") != list(MODEL_NAMES)
        or report.get("seeds") != list(seeds)
    ):
        raise ValueError("Source report feature, model or seed order does not match")
    if (
        report.get("steps") != 50_000
        or report.get("learning_parameters", {}).get("gamma") != 0.99
        or not report.get("canonical_protocol")
    ):
        raise ValueError(
            "Source report does not describe the canonical Milestone 4 protocol"
        )
    frozen = report["frozen_option"]
    options = FixedOptions(
        jnp.array(frozen["all_option_probability_weights"], dtype=jnp.float32),
        jnp.array(frozen["all_option_stopping_weights"], dtype=jnp.float32),
    )
    if options.probability_weights.shape != (
        6,
        4,
        len(features),
    ) or options.stopping_weights.shape != (6, len(features)):
        raise ValueError("Frozen policy dimensions do not match the feature order")
    if (
        not all(bool(jnp.isfinite(x).all()) for x in options)
        or bool(jnp.any(options.probability_weights < 0))
        or not bool(
            jnp.allclose(
                options.probability_weights.sum(axis=1), 1.0, rtol=0, atol=1e-6
            )
        )
        or bool(
            jnp.any((options.stopping_weights < 0) | (options.stopping_weights > 1))
        )
    ):
        raise ValueError("Frozen policy and stopping probabilities are invalid")
    primitive_policies = jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, len(features)))
    if not bool(
        jnp.array_equal(options.probability_weights[:4], primitive_policies)
    ) or not bool(jnp.all(options.stopping_weights[:4] == 1)):
        raise ValueError("Primitive policy definitions are inconsistent")
    entries = report["model_checkpoints"]
    if len(entries) != len(MODEL_STEPS) or {entry["step"] for entry in entries} != set(
        MODEL_STEPS
    ):
        raise ValueError("Source report must contain all four model checkpoints")
    checkpoints, sources, missing = {}, [], []
    for entry in entries:
        step = entry["step"]
        name = f"models_step_{step:05d}.json.gz"
        if Path(entry["path"]).name != name:
            raise ValueError("Checkpoint filename does not match its step")
        try:
            encoded = (directory / name).read_bytes()
        except FileNotFoundError:
            missing.append(name)
            continue  # Validate every existing file before allowing regeneration.
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
        payload = json.loads(gzip.decompress(encoded))
        if payload.get("step") != step:
            raise ValueError("Checkpoint step does not match the report")
        checkpoints[step] = _checkpoint(payload, features, seeds)
        sources.append({"step": step, "path": str(directory / name), "sha256": digest})
    if missing:
        raise FileNotFoundError(f"Missing model checkpoints: {', '.join(missing)}")
    return PlanningInputs(
        options,
        checkpoints,
        {
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "model_code_revision": report["code_revision"],
            "model_tracked_worktree_dirty": report["tracked_worktree_dirty"],
            "model_seeds": list(seeds),
            "checkpoints": sources,
            "frozen_option": frozen,
        },
    )
