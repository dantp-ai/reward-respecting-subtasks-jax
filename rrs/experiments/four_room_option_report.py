"""Independent references, policy evaluation and artifacts for Milestone 7."""

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
from rrs.experiments.four_room_options import (
    CHECKPOINT,
    EVALUATION_STEPS,
    OPTION_NAMES,
    PARAMETERS,
    SEEDS,
    STEPS,
)
from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.two_room import _revision
from rrs.rl.learning import softmax_policy, stopping_rule
from rrs.rl.stochastic_subtasks import evaluate_option, solve_subtask


@dataclass(frozen=True)
class References:
    positions: tuple
    model: object
    terminal: tuple
    nonterminal: tuple
    features: tuple
    stopping_values: tuple
    solutions: tuple
    evaluations: tuple


def build_references():
    positions, model = build_model()
    terminal = tuple(p == (9, 7) for p in positions)
    nonterminal = tuple(i for i, done in enumerate(terminal) if not done)
    features = tuple(
        tuple(float(i == j) for j in nonterminal) for i in range(len(positions))
    )
    z = tuple(
        tuple(float(p == target) for p in positions) for target in four_room.HALLWAYS
    )
    solutions = tuple(solve_subtask(model, values, terminal) for values in z)
    evaluations = []
    for i, solution in enumerate(solutions):
        pi = tuple(
            tuple(float(a == (actions[0] if actions else 0)) for a in range(4))
            for actions in solution.greedy_actions
        )
        evaluations.append(
            evaluate_option(
                model,
                pi,
                solution.stopping,
                z[i],
                terminal,
                positions.index(four_room.HALLWAYS[i]),
            )
        )
    return References(
        positions,
        model,
        terminal,
        nonterminal,
        features,
        z,
        solutions,
        tuple(evaluations),
    )


def policy_tables(weights, features, z, terminal):
    probabilities = jax.vmap(
        lambda actor: jax.vmap(softmax_policy, in_axes=(None, 0))(actor, features)
    )(weights.actor)
    stopping = stopping_rule(z, weights.critic @ features.T, terminal)
    return probabilities, stopping


_batch_tables = jax.jit(jax.vmap(policy_tables, in_axes=(0, None, None, None)))


def save_snapshot(path, state, step, seeds, data):
    pi, beta = _batch_tables(
        state.weights,
        jnp.array(data.features),
        jnp.array(data.stopping_values),
        jnp.array(data.terminal),
    )
    payload = {
        "schema_version": 1,
        "step": step,
        "seeds": list(seeds),
        "option_names": OPTION_NAMES,
        "feature_positions": [data.positions[i] for i in data.nonterminal],
        "policy_positions": data.positions,
        "critic_axes": ["run", "option", "feature"],
        "actor_axes": ["run", "option", "action", "feature"],
        "policy_axes": ["run", "option", "state", "action"],
        "stopping_axes": ["run", "option", "state"],
        "learning_parameters": PARAMETERS._asdict(),
        "main_task_weights": [0.0] * len(data.nonterminal),
        "hallway_bonus": 1.0,
        "critic_weights": state.weights.critic.tolist(),
        "actor_weights": state.weights.actor.tolist(),
        "policy_probabilities": pi.tolist(),
        "stopping": beta.tolist(),
    }
    encoded = gzip.compress(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {
        "step": step,
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _evaluation_metrics(evaluation, solution, data):
    start = data.positions.index((4, 1))
    result = {
        "actual_start": evaluation.values[start],
        "actual_value_rmse": math.sqrt(
            math.fsum(
                (evaluation.values[s] - solution.values[s]) ** 2
                for s in data.nonterminal
            )
            / len(data.nonterminal)
        ),
        "max_value_excess": max(
            v - opt for v, opt in zip(evaluation.values, solution.values, strict=True)
        ),
    }
    for name, values in (
        ("target", evaluation.target_probability),
        ("safe_target", evaluation.safe_target_probability),
        ("penalty", evaluation.penalty_probability),
    ):
        result[f"{name}_start"] = values[start]
        result[f"{name}_mean"] = math.fsum(values[s] for s in data.nonterminal) / len(
            data.nonterminal
        )
    return result


def _evaluate_snapshots(data, snapshots, seeds, progress):
    evaluations = {str(step): [] for step in snapshots}
    max_error, max_gap, max_excess, stopping_ok = 0.0, 0.0, 0.0, True
    for step, state in snapshots.items():
        pi, beta = _batch_tables(
            state.weights,
            jnp.array(data.features),
            jnp.array(data.stopping_values),
            jnp.array(data.terminal),
        )
        pis, betas = pi.tolist(), beta.tolist()
        for run, seed in enumerate(seeds):
            options = {}
            for i, name in enumerate(OPTION_NAMES):
                target = data.positions.index(four_room.HALLWAYS[i])
                evaluation = evaluate_option(
                    data.model,
                    pis[run][i],
                    betas[run][i],
                    data.stopping_values[i],
                    data.terminal,
                    target,
                )
                metrics = _evaluation_metrics(evaluation, data.solutions[i], data)
                options[name] = {
                    **metrics,
                    "values": evaluation.values,
                    "target_probability": evaluation.target_probability,
                    "safe_target_probability": evaluation.safe_target_probability,
                    "penalty_probability": evaluation.penalty_probability,
                    "stopping": betas[run][i],
                    "value_error_bound": evaluation.value_error_bound,
                    "probability_gap": evaluation.probability_gap,
                    "iterations": evaluation.iterations,
                }
                max_error = max(max_error, evaluation.value_error_bound)
                max_gap = max(max_gap, evaluation.probability_gap)
                max_excess = max(max_excess, metrics["max_value_excess"])
                stopping_ok = (
                    stopping_ok
                    and betas[run][i][target]
                    and all(
                        stop
                        for stop, terminal in zip(
                            betas[run][i], data.terminal, strict=True
                        )
                        if terminal
                    )
                )
            evaluations[str(step)].append({"seed": seed, "options": options})
            if progress and (run + 1) % 10 == 0:
                print(
                    f"Evaluated {run + 1}/{len(seeds)} seeds at {step:,} transitions",
                    flush=True,
                )
    checks = {
        "evaluation_value_bounds": max_error <= 1e-8,
        "evaluation_probability_bounds": max_gap <= 1e-8,
        "actual_values_do_not_exceed_optimum": max_excess <= 2e-8,
        "targets_and_goal_are_stopping_states": stopping_ok,
    }
    return evaluations, checks


def acceptance_checks(summary, optimal_starts):
    checks = {}
    for name, optimum in zip(OPTION_NAMES, optimal_starts, strict=True):
        option = summary[name]
        rmse = option["critic_rmse"]["mean"]
        actual = option["evaluations"][str(STEPS)]
        checks[f"{name}/critic_rmse_at_most_0.40"] = rmse[-1] <= 0.40
        checks[f"{name}/critic_rmse_at_most_60_percent_peak"] = rmse[-1] <= 0.60 * max(
            rmse
        )
        checks[f"{name}/estimated_start_within_0.15"] = (
            abs(option["estimated_start"]["mean"][-1] - optimum) <= 0.15
        )
        checks[f"{name}/actual_start_within_0.15"] = (
            abs(actual["actual_start"]["mean"] - optimum) <= 0.15
        )
        checks[f"{name}/actual_value_rmse_at_most_0.40"] = (
            actual["actual_value_rmse"]["mean"] <= 0.40
        )
    return checks


def build_report(data, history, snapshots, seeds, artifacts, progress=False):
    start = data.positions.index((4, 1))
    start_feature = data.nonterminal.index(start)
    targets = jnp.array(
        [[s.values[i] for i in data.nonterminal] for s in data.solutions]
    )
    errors = jnp.sqrt(jnp.mean((history - targets) ** 2, axis=-1)).tolist()
    starts = history[:, :, :, start_feature].tolist()
    evaluations, evaluation_checks = _evaluate_snapshots(
        data, snapshots, seeds, progress
    )
    summary = {}
    for i, name in enumerate(OPTION_NAMES):
        summary[name] = {}
        for metric, values in (("critic_rmse", errors), ("estimated_start", starts)):
            columns = [
                mean_and_se([run[t][i] for run in values])
                for t in range(history.shape[1])
            ]
            summary[name][metric] = {k: [c[k] for c in columns] for k in ("mean", "se")}
        summary[name]["evaluations"] = {
            step: {
                metric: mean_and_se([run["options"][name][metric] for run in runs])
                for metric in _evaluation_metrics(
                    data.evaluations[i], data.solutions[i], data
                )
            }
            for step, runs in evaluations.items()
        }
    optimal_starts = [s.values[start] for s in data.solutions]
    reference_difference = max(
        abs(v - opt)
        for e, s in zip(data.evaluations, data.solutions, strict=True)
        for v, opt in zip(e.values, s.values, strict=True)
    )
    eq9 = all(
        result.stopping
        == tuple(
            done or z >= value
            for done, z, value in zip(
                data.terminal, stopping, result.values, strict=True
            )
        )
        for result, stopping in zip(data.solutions, data.stopping_values, strict=True)
    )
    finite = bool(jnp.isfinite(history).all()) and all(
        bool(jnp.isfinite(w).all())
        for state in snapshots.values()
        for w in state.weights
    )
    finite = finite and all(
        math.isfinite(v)
        for runs in evaluations.values()
        for run in runs
        for option in run["options"].values()
        for field in (
            "values",
            "target_probability",
            "safe_target_probability",
            "penalty_probability",
        )
        for v in option[field]
    )
    terminal_zero = bool(
        jnp.all(history @ jnp.array(data.features)[jnp.array(data.terminal)].T == 0)
    )
    revision, dirty = _revision()
    return {
        "milestone": "07-four-room-options",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/13",
        "canonical_protocol": tuple(seeds) == SEEDS
        and history.shape[1] == STEPS // CHECKPOINT + 1
        and tuple(snapshots) == EVALUATION_STEPS,
        "seeds": list(seeds),
        "steps": STEPS,
        "checkpoint": CHECKPOINT,
        "checkpoints": list(range(0, STEPS + 1, CHECKPOINT)),
        "evaluation_steps": list(snapshots),
        "option_names": OPTION_NAMES,
        "hallways": four_room.HALLWAYS,
        "learning_parameters": PARAMETERS._asdict(),
        "main_task_weights": [0.0] * len(data.nonterminal),
        "hallway_bonus": 1.0,
        "behavior_probabilities": [0.25] * 4,
        "shared_experience_across_options": True,
        "key_schedule": "key(seed); split(next, action, environment, reset); randint(action, (), 0, 4)",
        "reset": "S, only after terminal transition learning; option stopping never resets behavior",
        "option_minimum_actions": 1,
        "stopping_rule": "Equation 9, z >= pre-update critic value",
        "precision": "float32 learning; Python double independent references/evaluation",
        "evaluation_tolerance": 1e-8,
        "evaluation_max_iterations": 20_000,
        "initial_weights_and_traces": 0.0,
        "feature_positions": [data.positions[i] for i in data.nonterminal],
        "policy_positions": data.positions,
        "references": {
            name: {
                "values": s.values,
                "stopping": s.stopping,
                "greedy_actions": s.greedy_actions,
                "bellman_residual": s.bellman_residual,
                "iterations": s.iterations,
                "evaluation": e.__dict__,
                "metrics": _evaluation_metrics(e, s, data),
            }
            for name, s, e in zip(
                OPTION_NAMES, data.solutions, data.evaluations, strict=True
            )
        },
        "runs": [
            {
                "seed": seed,
                "episodes": int(snapshots[STEPS].episodes[i]),
                "options": {
                    name: {
                        "critic_rmse": [row[j] for row in errors[i]],
                        "estimated_start": [row[j] for row in starts[i]],
                    }
                    for j, name in enumerate(OPTION_NAMES)
                },
            }
            for i, seed in enumerate(seeds)
        ],
        "evaluations": evaluations,
        "summary": summary,
        "policy_snapshots": artifacts,
        "acceptance": {
            "finite": finite,
            "terminal_zero": terminal_zero,
            "reference_values_match_optimized_subtasks": reference_difference <= 2e-8,
            "reference_and_equation_nine_stopping_agree": eq9,
            **evaluation_checks,
            **acceptance_checks(summary, optimal_starts),
        },
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }
