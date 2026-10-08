"""Independent references, per-model checks and artifacts for Milestone 8."""

import gzip
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from importlib.metadata import version

import jax
import jax.numpy as jnp

from rrs.envs import four_room
from rrs.experiments.four_room import build_model
from rrs.experiments.four_room_models import (
    CHECKPOINT,
    MODEL_NAMES,
    PARAMETERS,
    SAVE_STEPS,
    SEEDS,
    STEPS,
)
from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.model_learning import option_tables
from rrs.experiments.two_room import _revision
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.stochastic_models import stochastic_option_model
from rrs.rl.stochastic_subtasks import evaluate_option


@dataclass(frozen=True)
class ReferenceData:
    positions: tuple
    model: object
    terminal: tuple
    nonterminal_indices: tuple
    features: tuple


def build_data():
    positions, model = build_model()
    terminal = tuple(p == (9, 7) for p in positions)
    indices = tuple(i for i, done in enumerate(terminal) if not done)
    features = tuple(
        tuple(float(s == i) for i in indices) for s in range(len(positions))
    )
    return ReferenceData(positions, model, terminal, indices, features)


def build_references(data, options, progress=False):
    pi, beta = option_tables(options, data)
    pis, betas = pi.tolist(), beta.tolist()
    oracles, identities = [], {}
    for i, name in enumerate(MODEL_NAMES):
        oracle = stochastic_option_model(
            data.model, pis[i], betas[i], data.features, data.terminal
        )
        oracles.append(oracle)
        if i < 4:
            errors = []
            for s, row in enumerate(data.model.transitions):
                if data.terminal[s]:
                    expected_r, expected_n = 0.0, [0.0] * len(data.nonterminal_indices)
                else:
                    expected_r = math.fsum(o.probability * o.reward for o in row[i])
                    expected_n = [
                        PARAMETERS.gamma
                        * math.fsum(
                            o.probability * data.features[o.next_state][j]
                            for o in row[i]
                            if not o.terminated and not data.terminal[o.next_state]
                        )
                        for j in range(len(data.nonterminal_indices))
                    ]
                errors.append(abs(oracle.model.rewards[s] - expected_r))
                errors.extend(
                    abs(a - b)
                    for a, b in zip(oracle.model.successors[s], expected_n, strict=True)
                )
            identities[name] = {"maximum_error": max(errors), "tolerance": 1e-10}
        else:
            target = data.positions.index(four_room.HALLWAYS[i - 4])
            feature = data.nonterminal_indices.index(target)
            z = tuple(float(s == target) for s in range(len(data.positions)))
            evaluated = evaluate_option(
                data.model, pis[i], betas[i], z, data.terminal, target
            )
            difference = max(
                abs(r + n[feature] / PARAMETERS.gamma - v)
                for r, n, v in zip(
                    oracle.model.rewards,
                    oracle.model.successors,
                    evaluated.values,
                    strict=True,
                )
            )
            identities[name] = {"maximum_error": difference, "tolerance": 2e-8}
        if progress:
            print(
                f"Reference {name}: {oracle.iterations} sweeps, error bound {oracle.error_bound:.3g}",
                flush=True,
            )
    linear = [o.model.as_linear_model(data.nonterminal_indices) for o in oracles]
    targets = LinearExpectationModel(
        jnp.stack([m.reward_weights for m in linear]),
        jnp.stack([m.successor_weights for m in linear]),
    )
    return oracles, targets, identities


def _write_gzip(path, payload):
    encoded = gzip.compress(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def save_options(path, options, source, data):
    pi, beta = option_tables(options, data)
    return _write_gzip(
        path,
        {
            "schema_version": 1,
            "source": source,
            "model_names": MODEL_NAMES,
            "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
            "policy_positions": data.positions,
            "policy_axes": ["option", "state", "action"],
            "stopping_axes": ["option", "state"],
            "policy_probabilities": pi.tolist(),
            "stopping": beta.tolist(),
            "probability_weight_axes": ["option", "action", "input_feature"],
            "stopping_weight_axes": ["option", "input_feature"],
            "probability_weights": options.probability_weights.tolist(),
            "stopping_weights": options.stopping_weights.tolist(),
            "gamma": PARAMETERS.gamma,
            "minimum_actions": 1,
        },
    )


def save_checkpoint(path, models, step, seeds, data, frozen):
    revision, dirty = _revision()
    artifact = _write_gzip(
        path,
        {
            "schema_version": 1,
            "step": step,
            "seeds": list(seeds),
            "model_names": MODEL_NAMES,
            "gamma": PARAMETERS.gamma,
            "learning_parameters": PARAMETERS._asdict(),
            "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
            "reward_axes": ["run", "option", "input_feature"],
            "successor_axes": ["run", "option", "output_feature", "input_feature"],
            "reward_weights": models.reward_weights.tolist(),
            "successor_weights": models.successor_weights.tolist(),
            "frozen_options": frozen,
            "code_revision": revision,
            "tracked_worktree_dirty": dirty,
        },
    )
    return {"step": step, **artifact}


def summarize(history):
    result = {}
    for i, name in enumerate(MODEL_NAMES):
        result[name] = {}
        for metric, values in zip(
            ("reward_rmse", "successor_rmse"), history, strict=True
        ):
            columns = [
                mean_and_se(values[:, t, i].tolist()) for t in range(values.shape[1])
            ]
            result[name][metric] = {k: [c[k] for c in columns] for k in ("mean", "se")}
    return result


def acceptance_checks(summary):
    checks = {}
    for i, name in enumerate(MODEL_NAMES):
        reward_limit, successor_limit, fraction = (
            (0.15, 0.40, 0.65) if i < 4 else (0.25, 0.25, 0.50)
        )
        for metric, limit in (
            ("reward_rmse", reward_limit),
            ("successor_rmse", successor_limit),
        ):
            means = summary[name][metric]["mean"]
            checks[f"{name}/{metric}_at_most_{limit:.2f}"] = means[-1] <= limit
            checks[f"{name}/{metric}_at_most_{fraction:.2f}_initial"] = (
                means[-1] <= fraction * means[0]
            )
    return checks


def build_report(
    data,
    source,
    frozen,
    oracles,
    identities,
    state,
    history,
    audits,
    artifacts,
    finite,
    mass_ok,
):
    history = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *history)
    summary = summarize(history)
    zeros = jnp.zeros(len(data.nonterminal_indices))
    terminal_zero = bool(
        jnp.all(state.learners.model.reward_weights @ zeros == 0)
        & jnp.all(state.learners.model.successor_weights @ zeros == 0)
    )
    revision, dirty = _revision()
    return {
        "milestone": "08-four-room-models",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/15",
        "canonical_protocol": history.reward.shape
        == (len(SEEDS), STEPS // CHECKPOINT + 1, len(MODEL_NAMES))
        and tuple(a["step"] for a in artifacts) == SAVE_STEPS,
        "seeds": list(SEEDS),
        "steps": STEPS,
        "checkpoint": CHECKPOINT,
        "checkpoints": list(range(0, STEPS + 1, CHECKPOINT)),
        "model_names": MODEL_NAMES,
        "learning_parameters": PARAMETERS._asdict(),
        "behavior_probabilities": [0.25] * 4,
        "key_schedule": "key(seed); split(next, action, environment, reset); randint(action, (), 0, 4)",
        "reset": "S, after learning goal arrival; option stopping never resets",
        "initial_weights_and_traces": 0.0,
        "frozen_policy_source": source,
        "frozen_options": frozen,
        "shared_experience_across_models": True,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "policy_positions": data.positions,
        "precision": "float32 learning/metric targets; Python double reference and audit",
        "model_reward": "discounted environment reward, no artificial stopping bonus",
        "successor_discount": "gamma**K (Equation 15)",
        "successor_error_definition": "sqrt(mean_source(sum_output((prediction-target)**2)))",
        "oracle_tolerance": 1e-10,
        "oracle_max_iterations": 20_000,
        "references": {
            name: {
                "iterations": oracle.iterations,
                "bellman_residual": oracle.bellman_residual,
                "error_bound": oracle.error_bound,
                "rewards": oracle.model.rewards,
                "successors": oracle.model.successors,
            }
            for name, oracle in zip(MODEL_NAMES, oracles, strict=True)
        },
        "reference_identity_errors": identities,
        "audit_tolerance": "0.01 + 6 * sample_standard_error",
        "rollout_audits": audits,
        "summary": summary,
        "runs": [
            {
                "seed": seed,
                "episodes": int(state.episodes[i]),
                "reward_rmse": history.reward[i].tolist(),
                "successor_rmse": history.successor[i].tolist(),
            }
            for i, seed in enumerate(SEEDS)
        ],
        "final_mean_predictions": {
            "reward_weights": state.learners.model.reward_weights.mean(axis=0).tolist(),
            "successor_weights": state.learners.model.successor_weights.mean(
                axis=0
            ).tolist(),
        },
        "model_checkpoints": artifacts,
        "acceptance": {
            "finite": finite,
            "terminal_predictions_zero": terminal_zero,
            "successor_mass_in_bounds": mass_ok,
            "oracle_error_bounds": all(o.error_bound <= 1e-10 for o in oracles),
            "primitive_and_subtask_identities": all(
                v["maximum_error"] <= v["tolerance"] for v in identities.values()
            ),
            "independent_rollout_audits": len(audits) == 24
            and all(a["reward_passed"] and a["successor_passed"] for a in audits),
            **acceptance_checks(summary),
        },
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }
