"""State-space errors and independent option evaluation for Milestone 10."""

import gzip
import hashlib
import json
import math
import platform
from importlib.metadata import version

import jax
import jax.numpy as jnp

from rrs.experiments.learning_report import (
    acceptance_checks,
    mean_and_se,
    reference_model,
)
from rrs.experiments.tile_coded_options import CHECKPOINT, PARAMETERS, SEEDS, STEPS
from rrs.experiments.two_room import _revision
from rrs.rl.learning import softmax_policy, stopping_rule
from rrs.rl.policy_evaluation import evaluate_policy

CURVES = ("critic_rmse", "estimated_start_value")
OUTCOMES = ("stochastic_start_value", "safe_hallway_probability")


def _nonfinite_labels(value):
    """Preserve failed weights explicitly in JSON without invalid NaN tokens."""
    if isinstance(value, list):
        return [_nonfinite_labels(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_training(path, result, seeds):
    revision, dirty = _revision()
    payload = {
        "schema_version": 1,
        "seeds": list(seeds),
        "critic_history_axes": ["run", "checkpoint", "feature"],
        "actor_axes": ["run", "action", "feature"],
        "critic_history": _nonfinite_labels(result.critics.tolist()),
        "final_weights_and_traces": {
            name: _nonfinite_labels(value.tolist())
            for name, value in result.final.weights._asdict().items()
        },
        "final_positions": result.final.environment.position.tolist(),
        "final_keys": jax.random.key_data(result.final.key).tolist(),
        "episodes": result.final.episodes.tolist(),
        "nonfinite_encoding": "strings nan, inf, -inf identify failed parameters",
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
    }
    encoded = gzip.compress(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def _summary(values):
    # Never silently exclude a failed evaluation from the seed population.
    return (
        mean_and_se(values)
        if all(v is not None for v in values)
        else {"mean": None, "se": None}
    )


def case_report(data, name, problem, encode, result, seeds, progress=False):
    if not all(
        bool(jnp.isfinite(x).all()) for x in (*result.final.weights, result.critics)
    ):
        raise ValueError(f"Nonfinite {name} training parameters; inspect saved weights")
    features = jax.vmap(lambda obs: encode(problem.representation, obs))(
        jnp.asarray(data.positions)
    )
    predictions = result.critics @ features.T
    nonterminal = jnp.asarray(data.nonterminal_indices)
    target = jnp.asarray(data.solutions["reward_respecting"].values)[nonterminal]
    rmse = jnp.sqrt(
        jnp.mean((predictions[:, :, nonterminal] - target) ** 2, axis=-1)
    ).tolist()
    start = data.positions.index((3, 1))
    starts = predictions[:, :, start].tolist()
    model = reference_model(data.positions)
    z = data.subtasks["reward_respecting"].stopping_values
    runs = []
    for i, seed in enumerate(seeds):
        critic, actor = result.final.weights.critic[i], result.final.weights.actor[i]
        pi = jax.vmap(softmax_policy, in_axes=(None, 0))(actor, features).tolist()
        beta = stopping_rule(
            jnp.asarray(z), features @ critic, jnp.asarray(data.terminal)
        ).tolist()
        run = {
            "seed": seed,
            "episodes": int(result.final.episodes[i]),
            "critic_rmse": rmse[i],
            "estimated_start_value": starts[i],
            "critic_weights": critic.tolist(),
            "actor_weights": actor.tolist(),
            "predicted_values": predictions[i, -1].tolist(),
            "policy_probabilities": pi,
            "stopping": beta,
        }
        try:
            evaluation = evaluate_policy(
                model,
                pi,
                beta,
                z,
                data.terminal,
                data.positions.index((3, 7)),
                gamma=PARAMETERS.gamma,
            )
            run.update(
                {
                    "stochastic_start_value": evaluation.values[start],
                    "safe_hallway_probability": evaluation.safe_success[start],
                    "actual_values": evaluation.values,
                    "safe_success": evaluation.safe_success,
                    "evaluation_value_error_bound": evaluation.value_error_bound,
                    "evaluation_probability_gap": evaluation.probability_gap,
                    "evaluation_iterations": evaluation.iterations,
                    "evaluation_failure": None,
                }
            )
        except RuntimeError as error:
            run.update(
                {
                    key: None
                    for key in (
                        *OUTCOMES,
                        "evaluation_value_error_bound",
                        "evaluation_probability_gap",
                    )
                }
            )
            run["evaluation_failure"] = str(error)
        runs.append(run)
        if progress and (i + 1) % 25 == 0:
            print(f"Evaluated {name}: {i + 1}/{len(seeds)} options", flush=True)
    summary = {key: mean_curve([r[key] for r in runs]) for key in CURVES}
    summary.update({key: _summary([r[key] for r in runs]) for key in OUTCOMES})
    rows = features[nonterminal].tolist()
    overlaps = features[nonterminal] @ features[nonterminal].T
    metadata = {
        "name": name,
        "n_features": int(features.shape[1]),
        "feature_axes": ["state", "feature"],
        "state_features": features.tolist(),
        "distinct_nonterminal_vectors": len({tuple(row) for row in rows}),
        "nonterminal_squared_norms": jnp.sum(
            features[nonterminal] ** 2, axis=-1
        ).tolist(),
        "overlapping_state_pairs": int(jnp.sum(jnp.triu(overlaps > 0, k=1))),
    }
    if name == "tile_coding":
        metadata["parameters"] = {
            key: value.tolist()
            for key, value in problem.representation._asdict().items()
        }
        metadata["active_value"] = 0.5
    else:
        metadata["feature_positions"] = problem.representation.tolist()
    evaluated = all(r["evaluation_failure"] is None for r in runs)
    return {
        "representation": metadata,
        "summary": summary,
        "runs": runs,
        "acceptance": {
            "finite": bool(jnp.isfinite(predictions).all())
            and evaluated
            and all(math.isfinite(r[k]) for r in runs for k in OUTCOMES),
            "terminal_predictions_zero": bool(
                jnp.all(predictions[:, :, jnp.asarray(data.terminal)] == 0)
            ),
            "all_evaluations_bounded": evaluated
            and all(
                r["evaluation_value_error_bound"] <= 1e-8
                and r["evaluation_probability_gap"] <= 1e-8
                for r in runs
            ),
        },
    }


def mean_curve(rows):
    columns = [mean_and_se(values) for values in zip(*rows, strict=True)]
    return {key: [c[key] for c in columns] for key in ("mean", "se")}


def paired_differences(cases):
    baseline, tiles = cases["one_hot"]["runs"], cases["tile_coding"]["runs"]
    pairs = list(zip(baseline, tiles, strict=True))
    if any(a["seed"] != b["seed"] for a, b in pairs):
        raise ValueError("Representation runs must pair by the same seed")
    final = {}
    for metric in (*CURVES, *OUTCOMES):
        differences = []
        for a, b in pairs:
            av, bv = (
                (a[metric][-1], b[metric][-1])
                if metric in CURVES
                else (a[metric], b[metric])
            )
            differences.append(bv - av if av is not None and bv is not None else None)
        final[metric] = {**_summary(differences), "per_seed": differences}
    return final


def build_report(
    data, cases, results, artifacts, seeds=SEEDS, steps=STEPS, checkpoint=CHECKPOINT
):
    revision, dirty = _revision()
    baseline, tiles = results["one_hot"].final, results["tile_coding"].final
    checks = {
        f"{name}/{key}": value
        for name, case in cases.items()
        for key, value in case["acceptance"].items()
    }
    checks["paired_final_environment_keys_and_episodes"] = bool(
        jnp.array_equal(baseline.environment.position, tiles.environment.position)
        & jnp.array_equal(
            jax.random.key_data(baseline.key), jax.random.key_data(tiles.key)
        )
        & jnp.array_equal(baseline.episodes, tiles.episodes)
    )
    if cases["one_hot"]["acceptance"]["all_evaluations_bounded"]:
        checks.update(
            {
                f"milestone_3/{k}": v
                for k, v in acceptance_checks(
                    cases["one_hot"]["summary"],
                    cases["one_hot"]["acceptance"]["finite"],
                ).items()
            }
        )
    else:
        checks["milestone_3/baseline_evaluation"] = False
    return {
        "schema_version": 1,
        "milestone": "10-tile-coded-options",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/19",
        "canonical_protocol": tuple(seeds) == SEEDS
        and steps == STEPS
        and checkpoint == CHECKPOINT,
        "seeds": list(seeds),
        "steps": steps,
        "checkpoint": checkpoint,
        "checkpoints": list(range(0, steps + 1, checkpoint)),
        "learning_parameters": PARAMETERS._asdict(),
        "behavior_probabilities": [0.25] * 4,
        "key_schedule": "key(seed); split(next, action, environment, reset) per transition",
        "reset": "S after terminal learning update; option stopping does not reset",
        "target": {
            "hallway": [3, 7],
            "bonus": 1.0,
            "main_task_value": 0.0,
            "signal_independent_of_representation": True,
        },
        "initial_weights_and_traces": 0.0,
        "state_positions": data.positions,
        "optimal_start_value": data.solutions["reward_respecting"].values[
            data.positions.index((3, 1))
        ],
        "precision": "float32 learning; Python-double independent policy evaluation",
        "evaluation_tolerance": 1e-8,
        "evaluation_max_iterations": 20_000,
        "cases": cases,
        "paired_difference_direction": "tile_coding minus one_hot, same seed",
        "paired_differences": paired_differences(cases),
        "artifacts": artifacts,
        "acceptance": checks,
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }
