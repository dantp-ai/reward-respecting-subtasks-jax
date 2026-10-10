"""Validate frozen Milestone 7 policies and all Milestone 8 models."""

import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp

from rrs.experiments.four_room_model_inputs import SNAPSHOT_SHA256
from rrs.experiments.four_room_models import MODEL_NAMES
from rrs.rl.option_models import LinearExpectationModel

MODEL_STEPS = (0, 10_000, 20_000, 30_000, 40_000, 200_000)
MODEL_SEEDS = tuple(range(8000, 8030))
MODEL_REVISION = "d245abe258dd1471e1baf8ceb458ec583c581f54"
OPTION_REVISION = "e24dc66e5f9a490afd1fc3fe681c087478cbb827"


@dataclass(frozen=True)
class PlanningInputs:
    option_policies: jnp.ndarray  # [hallway option, state, action]
    option_stopping: jnp.ndarray  # [hallway option, state]
    checkpoints: dict[int, LinearExpectationModel]  # [run, model, ...]
    provenance: dict


def _checkpoint(payload, features, seeds, step):
    expected = {
        "schema_version": 1,
        "step": step,
        "seeds": list(seeds),
        "model_names": list(MODEL_NAMES),
        "gamma": 0.99,
        "learning_parameters": {"gamma": 0.99, "alpha_r": 0.1, "alpha_p": 0.1, "lambda_": 0.0},
        "feature_positions": features,
        "reward_axes": ["run", "option", "input_feature"],
        "successor_axes": ["run", "option", "output_feature", "input_feature"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Milestone 8 checkpoint {key} differs from the frozen protocol")
    rewards = jnp.asarray(payload["reward_weights"], dtype=jnp.float32)
    successors = jnp.asarray(payload["successor_weights"], dtype=jnp.float32)
    shape = (len(seeds), len(MODEL_NAMES), len(features))
    if rewards.shape != shape or successors.shape != shape + (len(features),):
        raise ValueError("Milestone 8 checkpoint dimensions do not match its axes")
    if not bool(jnp.isfinite(rewards).all() & jnp.isfinite(successors).all()):
        raise ValueError("Milestone 8 model coefficients must be finite")
    if bool((successors < -1e-7).any()) or bool((successors.sum(axis=-2) > 0.99 + 1e-6).any()):
        raise ValueError("Milestone 8 successor mass is outside its frozen bounds")
    return LinearExpectationModel(rewards, successors)


def load_inputs(directory, positions, nonterminal_indices, seeds=MODEL_SEEDS):
    directory = Path(directory)
    features = [list(positions[i]) for i in nonterminal_indices]
    report_path = directory / "learning.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (
        report.get("milestone") != "08-four-room-models"
        or report.get("code_revision") != MODEL_REVISION
        or not report.get("canonical_protocol")
        or report.get("steps") != 200_000
        or report.get("checkpoint") != 1_000
        or report.get("seeds") != list(seeds)
        or report.get("model_names") != list(MODEL_NAMES)
        or report.get("feature_positions") != features
        or report.get("learning_parameters") != {"gamma": 0.99, "alpha_r": 0.1, "alpha_p": 0.1, "lambda_": 0.0}
    ):
        raise ValueError("Milestone 8 report differs from the canonical planning inputs")
    source = report["frozen_policy_source"]
    if source.get("sha256") != SNAPSHOT_SHA256 or source.get("code_revision") != OPTION_REVISION or source.get("seed") != 7000:
        raise ValueError("Milestone 8 report used a different frozen Milestone 7 policy")
    frozen_entry = report["frozen_options"]
    frozen_path = directory / "frozen_options.json.gz"
    if Path(frozen_entry["path"]).name != frozen_path.name:
        raise ValueError("Frozen option export name differs from its report")
    frozen_bytes = frozen_path.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen_hash != frozen_entry["sha256"]:
        raise ValueError("Frozen option export hash does not match its report")
    frozen = json.loads(gzip.decompress(frozen_bytes))
    if (
        frozen.get("schema_version") != 1
        or frozen.get("source", {}).get("sha256") != SNAPSHOT_SHA256
        or frozen.get("model_names") != list(MODEL_NAMES)
        or frozen.get("feature_positions") != features
        or frozen.get("policy_positions") != [list(p) for p in positions]
        or frozen.get("policy_axes") != ["option", "state", "action"]
        or frozen.get("stopping_axes") != ["option", "state"]
    ):
        raise ValueError("Frozen policy export has incompatible axes or source metadata")
    policies = jnp.asarray(frozen["policy_probabilities"], dtype=jnp.float32)
    stopping = jnp.asarray(frozen["stopping"], dtype=jnp.float32)
    expected_policy_shape = (len(MODEL_NAMES), len(positions), 4)
    if policies.shape != expected_policy_shape or stopping.shape != expected_policy_shape[:2]:
        raise ValueError("Frozen policy dimensions do not match the four-room states")
    if not bool(
        jnp.isfinite(policies).all()
        & (policies >= 0).all()
        & (jnp.abs(policies.sum(axis=-1) - 1) <= 1e-6).all()
        & ((stopping == 0) | (stopping == 1)).all()
    ):
        raise ValueError("Frozen policy probabilities or stopping rules are invalid")
    for i in range(4):
        if not bool(
            jnp.all(policies[i, jnp.asarray(nonterminal_indices), i] == 1)
            & jnp.all(stopping[i] == 1)
        ):
            raise ValueError("Primitive action policies in the frozen export are invalid")
    entries = report["model_checkpoints"]
    if {entry["step"] for entry in entries} != set(MODEL_STEPS) or len(entries) != len(MODEL_STEPS):
        raise ValueError("Milestone 8 report must contain all six frozen model checkpoints")
    checkpoints, sources = {}, []
    for entry in entries:
        step = entry["step"]
        name = f"models_step_{step:06d}.json.gz"
        if Path(entry["path"]).name != name:
            raise ValueError("Milestone 8 checkpoint filename does not match its step")
        encoded = (directory / name).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Milestone 8 checkpoint hash mismatch: {name}")
        payload = json.loads(gzip.decompress(encoded))
        frozen_link = payload.get("frozen_options", {})
        if frozen_link.get("sha256") != frozen_hash or payload.get("code_revision") != MODEL_REVISION:
            raise ValueError("Milestone 8 checkpoint uses a different policy or code revision")
        checkpoints[step] = _checkpoint(payload, features, seeds, step)
        sources.append({"step": step, "path": str(directory / name), "sha256": digest})
    return PlanningInputs(
        policies[4:],
        stopping[4:],
        checkpoints,
        {
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "model_code_revision": report["code_revision"],
            "model_seeds": list(seeds),
            "checkpoints": sources,
            "frozen_option_source": source,
            "frozen_option_export": {"path": str(frozen_path), "sha256": frozen_hash},
        },
    )


def ensure_inputs(directory=None, figures_dir=Path("figures")):
    """Regenerate the default absent source; preserve explicit/malformed sources."""
    directory = Path("artifacts/four_room_models") if directory is None else Path(directory)
    report = directory / "learning.json"
    if report.is_file():
        return directory
    # A user-supplied directory is an explicit artifact source and must be complete.
    if directory != Path("artifacts/four_room_models"):
        raise FileNotFoundError(report)
    print("Milestone 8 artifacts absent; regenerating its frozen experiment", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "rrs.experiments.four_room_models",
            "--output-dir", str(directory), "--figures-dir", str(figures_dir),
        ],
        check=True,
    )
    if not report.is_file():
        raise FileNotFoundError(report)
    return directory
