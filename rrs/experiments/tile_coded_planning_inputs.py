"""Strict Milestone 11 input validation and exact feature-model controls."""

import gzip
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rrs.experiments.tile_coded_model_report import build_references, decoder
from rrs.experiments.tile_coded_models import NAMES, PARAMETERS, SEEDS, features
from rrs.experiments.tile_coded_options import hallway_signal
from rrs.rl.learning import stopping_rule
from rrs.rl.option_models import LinearExpectationModel

MODEL_STEPS = (0, 10_000, 20_000, 50_000)


def require_fields(payload, expected):
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Milestone 11 {name} does not match its contract")


def checked_bytes(directory, entry, name):
    if Path(entry.get("path", "")).name != name:
        raise ValueError("Artifact filename does not match its declaration")
    raw = (directory / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
        raise ValueError(f"Artifact hash mismatch: {name}")
    return raw


def validate_options(payload, data, tables):
    require_fields(
        payload,
        {
            "positions": [list(p) for p in data.positions],
            "model_names": list(NAMES),
            "option_parameters": {
                "alpha": 0.1,
                "alpha_actor": 0.1,
                "gamma": 0.99,
                "lambda_": 0.0,
                "lambda_actor": 0.0,
            },
        },
    )
    pi, beta = np.asarray(payload["probabilities"]), np.asarray(payload["stopping"])
    if pi.shape != (7, 73, 4) or beta.shape != (7, 73):
        raise ValueError("Frozen option dimensions are invalid")
    if (
        not np.isfinite(pi).all()
        or not np.isfinite(beta).all()
        or np.any(pi < 0)
        or not np.allclose(pi.sum(axis=-1), 1, atol=1e-6, rtol=0)
        or np.any((beta < 0) | (beta > 1))
    ):
        raise ValueError("Frozen option probabilities are invalid")
    terminal = np.array(data.terminal)
    for i, name in enumerate(NAMES[:5]):
        option = data.options[name]
        expected_pi = np.eye(4)[list(option.policy)]
        expected_pi[terminal] = 0.25
        expected_beta = np.where(terminal, 1, option.stopping)
        if not np.array_equal(pi[i], expected_pi) or not np.array_equal(
            beta[i], expected_beta
        ):
            raise ValueError("Primitive or shortest-path option changed")
    for i, name in enumerate(("one_hot", "tile_coding"), 5):
        frozen = payload["trained_options"][name]
        require_fields(frozen, {"seed": 100, "steps": 200_000})
        x = tables[name]
        actor, critic = (
            jnp.array(frozen["actor_weights"]),
            jnp.array(frozen["critic_weights"]),
        )
        if (
            actor.shape != (4, x.shape[1])
            or critic.shape != (x.shape[1],)
            or not (
                bool(jnp.isfinite(actor).all()) and bool(jnp.isfinite(critic).all())
            )
        ):
            raise ValueError("Frozen option parameters are invalid")
        expected_pi = jax.nn.softmax(x @ actor.T, axis=-1)
        expected_beta = stopping_rule(
            jax.vmap(hallway_signal)(jnp.array(data.positions)),
            x @ critic,
            jnp.array(terminal),
        )
        if not np.allclose(pi[i], expected_pi, atol=1e-6, rtol=0) or not np.array_equal(
            beta[i], expected_beta
        ):
            raise ValueError("Frozen option tables disagree with their weights")
    return jnp.array(pi), jnp.array(beta)


def validate_checkpoint(payload, data, x, step, source, seeds=SEEDS):
    require_fields(
        payload,
        {
            "schema_version": 1,
            "step": step,
            "seeds": list(seeds),
            "model_names": list(NAMES),
            "parameters": PARAMETERS._asdict(),
            "frozen_options": source,
            "positions": [list(p) for p in data.positions],
            "features": x.tolist(),
            "reward_axes": ["run", "option", "input_feature"],
            "successor_axes": ["run", "option", "output_feature", "input_feature"],
        },
    )
    model = LinearExpectationModel(
        jnp.array(payload["reward_weights"]), jnp.array(payload["successor_weights"])
    )
    shape = (len(seeds), len(NAMES), x.shape[1])
    if model.reward_weights.shape != shape or model.successor_weights.shape != shape + (
        x.shape[1],
    ):
        raise ValueError("Checkpoint model dimensions are invalid")
    if not all(bool(jnp.isfinite(v).all()) for v in model):
        raise ValueError("Checkpoint coefficients must be finite")
    if not bool(jnp.all(x[jnp.array(data.terminal)] == 0)):
        raise ValueError("Terminal features must be zero")
    # Coefficients in an overlapping feature basis are NOT transition probabilities.
    return model


def load_inputs(directory, data):
    directory = Path(directory)
    raw = (directory / "learning.json").read_bytes()
    report = json.loads(raw)
    require_fields(
        report,
        {
            "milestone": "11-tile-coded-models",
            "seeds": list(SEEDS),
            "steps": 50_000,
            "parameters": PARAMETERS._asdict(),
            "model_names": list(NAMES),
        },
    )
    if report.get("implementation_checks") != dict.fromkeys(
        ("finite", "terminal_predictions_zero", "oracle_bounds", "rollout_audits"), True
    ):
        raise ValueError("Source experiment did not pass its implementation checks")
    tables = {name: features(data, name) for name in ("one_hot", "tile_coding")}
    missing, checkpoints, hashes = [], {}, []
    source = report["frozen_options"]
    try:
        options = json.loads(checked_bytes(directory, source, "frozen_options.json"))
        pi, beta = validate_options(options, data, tables)
    except FileNotFoundError:
        missing.append("frozen_options.json")
    for name, x in tables.items():
        case = report["cases"][name]
        if case["features"] != x.tolist():
            raise ValueError("Report feature ordering changed")
        entries = {Path(e["path"]).name: e for e in case["checkpoints"]}
        expected = {f"{name}_step_{step:05d}.json.gz" for step in MODEL_STEPS}
        if set(entries) != expected or len(case["checkpoints"]) != 4:
            raise ValueError("Expected four unique model checkpoints")
        checkpoints[name] = {}
        for step in MODEL_STEPS:
            filename = f"{name}_step_{step:05d}.json.gz"
            try:
                raw_checkpoint = checked_bytes(directory, entries[filename], filename)
            except FileNotFoundError:
                missing.append(filename)
                continue
            payload = json.loads(gzip.decompress(raw_checkpoint))
            checkpoints[name][step] = validate_checkpoint(
                payload, data, x, step, source, seeds=SEEDS
            )
            hashes.append(
                {
                    "representation": name,
                    "step": step,
                    "sha256": entries[filename]["sha256"],
                }
            )
    if missing:
        raise FileNotFoundError(f"Incomplete Milestone 11 inputs: {', '.join(missing)}")
    return {
        "features": tables,
        "policies": pi,
        "stopping": beta,
        "checkpoints": checkpoints,
        "saved_references": report["references"],
        "provenance": {
            "report_path": str(directory / "learning.json"),
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "source_provenance": report["provenance"],
            "frozen_options": source,
            "checkpoints": hashes,
        },
    }


def exact_models(data, inputs):
    refs, agreement = build_references(
        data, inputs["policies"], inputs["stopping"], inputs["features"]
    )
    models, reconstructions = {}, {}
    ids = list(data.nonterminal_indices)
    for name, x in inputs["features"].items():
        for key in ("reward", "successor"):
            if not np.allclose(
                refs[name][key],
                inputs["saved_references"][name][key],
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError("Recomputed exact references disagree with source")
        d, geometry = decoder(x, ids)
        r, n = (
            np.array(refs[name]["reward"])[:, ids],
            np.array(refs[name]["successor"])[:, ids],
        )
        rw, nw = r @ d.T, n.transpose(0, 2, 1) @ d.T
        xn = np.array(x)[ids]
        double_error = max(
            float(np.max(np.abs(rw @ xn.T - r))),
            float(np.max(np.abs((nw @ xn.T).transpose(0, 2, 1) - n))),
        )
        model = LinearExpectationModel(jnp.array(rw), jnp.array(nw))
        float_error = max(
            float(jnp.max(jnp.abs(model.reward_weights @ x[jnp.array(ids)].T - r))),
            float(
                jnp.max(
                    jnp.abs(
                        jnp.swapaxes(
                            model.successor_weights @ x[jnp.array(ids)].T, -1, -2
                        )
                        - n
                    )
                )
            ),
        )
        if double_error > 1e-10 or float_error > 1e-5:
            raise ValueError("Exact feature-model reconstruction exceeds tolerance")
        models[name] = model
        reconstructions[name] = {
            "double_error": double_error,
            "float32_error": float_error,
            "geometry": geometry,
        }
    return models, {"reconstruction": reconstructions, "reference_agreement": agreement}
