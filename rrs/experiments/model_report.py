"""References, checkpoint artifacts and scientific reporting for Milestone 4."""

import gzip
import hashlib
import json
import platform
from importlib.metadata import version

import jax
import jax.numpy as jnp

from rrs.experiments.learning_report import mean_and_se, reference_model
from rrs.experiments.model_learning import MODEL_NAMES, option_tables
from rrs.experiments.two_room import _revision
from rrs.rl.learning import LearningParameters
from rrs.rl.model_learning import ModelErrors, ModelLearningParameters
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.stochastic_models import stochastic_option_model


def build_references(data, options):
    """Use independent environment transitions and frozen stochastic options."""
    model = reference_model(data.positions)
    probabilities, stopping = option_tables(options, data)
    oracles = [
        stochastic_option_model(model, pi, beta, data.features, data.terminal)
        for pi, beta in zip(probabilities.tolist(), stopping.tolist(), strict=True)
    ]
    linear = [
        oracle.model.as_linear_model(data.nonterminal_indices) for oracle in oracles
    ]
    targets = LinearExpectationModel(
        jnp.stack([m.reward_weights for m in linear]),
        jnp.stack([m.successor_weights for m in linear]),
    )
    return oracles, targets


def model_rmse(
    models: LinearExpectationModel, targets: LinearExpectationModel
) -> ModelErrors:
    """Equal weight per source state; successor error uses squared vector norm."""
    reward = jnp.sqrt(
        jnp.mean((models.reward_weights - targets.reward_weights) ** 2, axis=-1)
    )
    difference = models.successor_weights - targets.successor_weights
    successor = jnp.sqrt(jnp.mean(jnp.sum(difference**2, axis=-2), axis=-1))
    return ModelErrors(reward, successor)


def save_checkpoint(path, models, step, seeds, data):
    """Self-contained compressed JSON, with deterministic gzip metadata."""
    payload = {
        "schema_version": 1,
        "step": step,
        "seeds": list(seeds),
        "model_names": MODEL_NAMES,
        "gamma": 0.99,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "reward_axes": ["run", "option", "input_feature"],
        "successor_axes": ["run", "option", "output_feature", "input_feature"],
        "reward_weights": models.reward_weights.tolist(),
        "successor_weights": models.successor_weights.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(encoded, mtime=0)
    path.write_bytes(compressed)
    return {
        "step": step,
        "path": str(path),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def _summarize(history):
    summaries = {}
    for option, name in enumerate(MODEL_NAMES):
        summaries[name] = {}
        for key, values in zip(("reward_rmse", "successor_rmse"), history, strict=True):
            columns = [
                mean_and_se(values[:, step, option].tolist())
                for step in range(values.shape[1])
            ]
            summaries[name][key] = {
                k: [column[k] for column in columns] for k in ("mean", "se")
            }
    return summaries


def acceptance_checks(summary, finite, terminal_zero, audits):
    checks = {
        "finite": bool(finite),
        "terminal_predictions_zero": bool(terminal_zero),
        "monte_carlo_audit": len(audits) == 4
        and all(a["reward_passed"] and a["successor_passed"] for a in audits),
    }
    limits = {
        **dict.fromkeys(MODEL_NAMES[:4], 0.01),
        "reward_respecting": 0.20,
        "shortest_path": 0.10,
    }
    for name, metrics in summary.items():
        for metric, values in metrics.items():
            means = values["mean"]
            checks[f"{name}_{metric}_within_{limits[name]:.2f}"] = (
                means[-1] <= limits[name]
            )
            checks[f"{name}_{metric}_at_most_half_initial"] = (
                means[-1] <= 0.5 * means[0]
            )
    return checks


def build_report(
    data,
    frozen_weights,
    options,
    oracles,
    targets,
    state,
    history,
    seeds,
    steps,
    checkpoint,
    parameters,
    audits,
    artifacts,
    finite,
):
    history = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *history)
    summary = _summarize(history)
    zeros = jnp.zeros(len(data.nonterminal_indices))
    terminal_zero = bool(
        jnp.all(state.learners.model.reward_weights @ zeros == 0)
        & jnp.all(state.learners.model.successor_weights @ zeros == 0)
    )
    revision, dirty = _revision()
    return {
        "milestone": "04-option-model-learning",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/7",
        "canonical_protocol": tuple(seeds) == tuple(range(1000, 1100))
        and steps == 50_000
        and checkpoint == 500
        and parameters == ModelLearningParameters(),
        "seeds": list(seeds),
        "steps": steps,
        "checkpoint": checkpoint,
        "learning_parameters": parameters._asdict(),
        "model_names": MODEL_NAMES,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "behavior_probabilities": [0.25] * 4,
        "key_schedule": "key(seed); split(next, action, environment, reset) each transition",
        "environment_reset": "S, only after learning the terminal reward at G",
        "initial_model_weights_and_traces": 0.0,
        "frozen_option": {
            "source": "Milestone 3 learner, seed 0, 50000 steps, unchanged default parameters",
            "seed": 0,
            "steps": 50_000,
            "checkpoint": 500,
            "learning_parameters": LearningParameters()._asdict(),
            "main_task_weights": [0.0] * 72,
            "hallway_bonus": 1.0,
            "critic_weights": frozen_weights.critic.tolist(),
            "actor_weights": frozen_weights.actor.tolist(),
            "all_option_probability_weights": options.probability_weights.tolist(),
            "all_option_stopping_weights": options.stopping_weights.tolist(),
        },
        "precision": "float32 learning; Python double exact oracle and Monte Carlo audit",
        "successor_error_definition": "sqrt(mean_source(sum_output((prediction-target)**2)))",
        "model_reward": "discounted environment rewards; no artificial stopping bonus",
        "successor_discount": "gamma**K, Equation 15",
        "oracle_tolerance": 1e-10,
        "oracle_max_iterations": 20_000,
        "references": {
            name: {
                "iterations": oracle.iterations,
                "bellman_residual": oracle.bellman_residual,
                "error_bound": oracle.error_bound,
                "reward_weights": targets.reward_weights[i].tolist(),
                "successor_weights": targets.successor_weights[i].tolist(),
            }
            for i, (name, oracle) in enumerate(zip(MODEL_NAMES, oracles, strict=True))
        },
        "audit_tolerance": "0.01 + 6 * sample_standard_error",
        "rollout_audits": audits,
        "checkpoints": list(range(0, steps + 1, checkpoint)),
        "summary": summary,
        "runs": [
            {
                "seed": seed,
                "episodes": int(state.episodes[i]),
                "reward_rmse": history.reward[i].tolist(),
                "successor_rmse": history.successor[i].tolist(),
            }
            for i, seed in enumerate(seeds)
        ],
        "model_checkpoints": artifacts,
        "acceptance": acceptance_checks(summary, finite, terminal_zero, audits),
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }


def plot_results(data, report, models, targets, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    curves = directory / "milestone_04_model_learning.png"
    predictions = directory / "milestone_04_hallway_model.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, key, label in zip(
        axes,
        ("reward_rmse", "successor_rmse"),
        ("Reward RMSE", "Successor RMSE (vector norm per state)"),
        strict=True,
    ):
        for name in MODEL_NAMES:
            values = report["summary"][name][key]
            mean, se = values["mean"], values["se"]
            ax.plot(
                report["checkpoints"],
                mean,
                label=name.replace("_", " "),
                linewidth=2 if name in MODEL_NAMES[4:] else 1,
            )
            ax.fill_between(
                report["checkpoints"],
                [max(1e-12, m - e) for m, e in zip(mean, se, strict=True)],
                [m + e for m, e in zip(mean, se, strict=True)],
                alpha=0.12,
            )
        ax.set(xlabel="Environment transitions", ylabel=label, yscale="log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Milestone 4 — Frozen options, {len(report['seeds'])} model-training seeds, mean ± SE"
    )
    fig.savefig(curves, dpi=160)
    plt.close(fig)

    hallway = data.nonterminal_indices.index(data.positions.index((3, 7)))
    pairs = (
        (
            targets.reward_weights[4],
            models.reward_weights[:, 4],
            "Discounted environment reward",
        ),
        (
            targets.successor_weights[4, hallway],
            models.successor_weights[:, 4, hallway],
            "Discounted final hallway feature",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, (exact, samples, label) in zip(axes, pairs, strict=True):
        mean = jnp.mean(samples, axis=0)
        se = jnp.std(samples, axis=0, ddof=1) / jnp.sqrt(samples.shape[0])
        ax.errorbar(
            exact.tolist(),
            mean.tolist(),
            yerr=se.tolist(),
            fmt="o",
            markersize=3,
            capsize=2,
            alpha=0.75,
        )
        lower, upper = (
            float(jnp.minimum(exact.min(), mean.min())),
            float(jnp.maximum(exact.max(), mean.max())),
        )
        ax.plot(
            [lower, upper], [lower, upper], "k--", linewidth=1, label="Exact agreement"
        )
        ax.set(xlabel="Exact model", ylabel="Learned model (mean ± SE)", title=label)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        "Milestone 4 — Frozen stochastic hallway option, all 72 nonterminal starts"
    )
    fig.savefig(predictions, dpi=160)
    plt.close(fig)
    return curves, predictions
