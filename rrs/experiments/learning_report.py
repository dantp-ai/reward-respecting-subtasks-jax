"""Independent stochastic-policy evaluation and scientific Milestone 3 output."""

import math
import platform
import statistics
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp

from rrs.envs import reference
from rrs.experiments.hallway import _draw_grid
from rrs.experiments.two_room import _revision
from rrs.rl.exact import DeterministicModel
from rrs.rl.learning import LearningParameters, softmax_policy, stopping_rule
from rrs.rl.policy_evaluation import evaluate_policy


def mean_and_se(samples):
    return {
        "mean": statistics.fmean(samples),
        "se": statistics.stdev(samples) / math.sqrt(len(samples))
        if len(samples) > 1
        else 0.0,
    }


def reference_model(positions):
    """Enumerate the independent Python environment, not the training transition."""
    indices = {position: i for i, position in enumerate(positions)}
    rows = [
        [reference.step(position, action) for action in range(4)]
        for position in positions
    ]
    return DeterministicModel(
        tuple(tuple(indices[p] for p, _, _ in row) for row in rows),
        tuple(tuple(r for _, r, _ in row) for row in rows),
        tuple(tuple(done for _, _, done in row) for row in rows),
    )


def policy_tables(critic, actor, data):
    features = jnp.array(data.features)
    probabilities = jax.vmap(softmax_policy, in_axes=(None, 0))(actor, features)
    beta = stopping_rule(
        jnp.array(data.subtasks["reward_respecting"].stopping_values),
        features @ critic,
        jnp.array(data.terminal),
    )
    return probabilities.tolist(), beta.tolist()


def acceptance_checks(summary, finite):
    """Thresholds frozen in option_learning_contract.md before training."""
    rmse = summary["critic_rmse"]["mean"]
    estimated = summary["estimated_start_value"]["mean"]
    return {
        "finite": bool(finite),
        "final_mean_rmse_at_most_0.40": rmse[-1] <= 0.40,
        "final_mean_rmse_at_most_75_percent_of_peak": rmse[-1] <= 0.75 * max(rmse),
        "mean_start_estimate_within_0.15_of_optimal": abs(estimated[-1] - 0.99**11)
        <= 0.15,
        "mean_safe_hallway_probability_at_least_0.80": summary[
            "safe_hallway_probability"
        ]["mean"]
        >= 0.80,
    }


def build_report(data, result, seeds, steps, checkpoint, parameters, progress=False):
    """Keep per-seed metrics and final parameters so all conclusions are auditable."""
    model = reference_model(data.positions)
    start = data.positions.index((3, 1))
    start_feature = data.nonterminal_indices.index(start)
    exact = jnp.array(
        [
            data.solutions["reward_respecting"].values[i]
            for i in data.nonterminal_indices
        ]
    )
    errors = jnp.sqrt(jnp.mean((result.critics - exact) ** 2, axis=-1)).tolist()
    starts = result.critics[:, :, start_feature].tolist()
    runs = []
    for i, seed in enumerate(seeds):
        critic, actor = result.final.weights.critic[i], result.final.weights.actor[i]
        probabilities, stopping = policy_tables(critic, actor, data)
        evaluation = evaluate_policy(
            model,
            probabilities,
            stopping,
            data.subtasks["reward_respecting"].stopping_values,
            data.terminal,
            data.positions.index((3, 7)),
            gamma=parameters.gamma,
        )
        runs.append(
            {
                "seed": seed,
                "episodes": int(result.final.episodes[i]),
                "critic_rmse": errors[i],
                "estimated_start_value": starts[i],
                "stochastic_start_value": evaluation.values[start],
                "safe_hallway_probability": evaluation.safe_success[start],
                "evaluation_iterations": evaluation.iterations,
                "evaluation_value_error_bound": evaluation.value_error_bound,
                "evaluation_probability_gap": evaluation.probability_gap,
                "critic_weights": critic.tolist(),
                "actor_weights": actor.tolist(),
                "policy_probabilities": probabilities,
                "stopping_positions": [
                    p for p, stop in zip(data.positions, stopping, strict=True) if stop
                ],
            }
        )
        if progress and (i + 1) % 25 == 0:
            print(f"Evaluated {i + 1}/{len(seeds)} stochastic policies", flush=True)
    summary = {}
    for name in ("critic_rmse", "estimated_start_value"):
        columns = [
            mean_and_se(column)
            for column in zip(*(run[name] for run in runs), strict=True)
        ]
        summary[name] = {
            key: [column[key] for column in columns] for key in ("mean", "se")
        }
    for name in ("stochastic_start_value", "safe_hallway_probability"):
        summary[name] = mean_and_se([run[name] for run in runs])
    finite = bool(jnp.isfinite(result.critics).all()) and all(
        bool(jnp.isfinite(w).all()) for w in result.final.weights
    )
    finite = finite and all(
        math.isfinite(run[key])
        for run in runs
        for key in ("stochastic_start_value", "safe_hallway_probability")
    )
    revision, dirty = _revision()
    return {
        "milestone": "03-hallway-option-learning",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/5",
        "canonical_protocol": tuple(seeds) == tuple(range(100))
        and steps == 50_000
        and checkpoint == 500
        and parameters == LearningParameters(),
        "seeds": list(seeds),
        "steps": steps,
        "checkpoint": checkpoint,
        "learning_parameters": parameters._asdict(),
        "behavior_probabilities": [0.25] * 4,
        "main_task_weights": [0.0] * 72,
        "hallway_bonus": 1.0,
        "initial_weights_and_traces": 0.0,
        "precision": "float32 learning; Python double policy evaluation",
        "key_schedule": "key(seed); split(next, action, environment, reset) each transition",
        "environment_reset": "start at S; reset only after terminal update at G",
        "evaluation_tolerance": 1e-8,
        "evaluation_max_iterations": 20_000,
        "policy_probability_roundoff": "normalize float32 rows in double precision",
        "optimal_start_value": data.solutions["reward_respecting"].values[start],
        "initial_stochastic_start_value": -0.25,
        "initial_safe_hallway_probability": 0.0,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "policy_positions": data.positions,
        "checkpoints": list(range(0, steps + 1, checkpoint)),
        "summary": summary,
        "runs": runs,
        "acceptance": acceptance_checks(summary, finite),
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }


def plot_results(data, report, directory: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    directory.mkdir(parents=True, exist_ok=True)
    curves = directory / "milestone_03_option_learning.png"
    policy = directory / "milestone_03_stochastic_policy.png"
    steps, summary = report["checkpoints"], report["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, key, title in zip(
        axes,
        ("critic_rmse", "estimated_start_value"),
        ("Critic RMSE over 72 states", "Subtask return from S"),
        strict=True,
    ):
        mean, se = summary[key]["mean"], summary[key]["se"]
        ax.plot(steps, mean, label="Critic mean")
        ax.fill_between(
            steps,
            [m - e for m, e in zip(mean, se)],
            [m + e for m, e in zip(mean, se)],
            alpha=0.25,
            label="±1 standard error",
        )
        ax.set(xlabel="Environment transitions", ylabel=title)
        ax.grid(alpha=0.25)
    axes[0].axhline(0.40, color="gray", linestyle=":", label="Acceptance: 0.40")
    axes[0].legend()
    axes[1].axhline(
        report["optimal_start_value"],
        color="black",
        linestyle="--",
        label="Exact optimum",
    )
    actual = summary["stochastic_start_value"]
    axes[1].errorbar(
        [steps[-1]],
        [actual["mean"]],
        yerr=[actual["se"]],
        fmt="o",
        color="#b34e16",
        capsize=4,
        label="Actual stochastic policy",
    )
    axes[1].legend()
    fig.suptitle(
        f"Milestone 3 — Uniform behavior, {len(report['seeds'])} seeds, α = α_actor = 0.1"
    )
    fig.savefig(curves, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    run = report["runs"][0]  # Fixed seed 0, never selected by performance.
    stopping_positions = {tuple(position) for position in run["stopping_positions"]}
    _draw_grid(axes[0], data.params)
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for (row, col), probabilities in zip(
        data.positions, run["policy_probabilities"], strict=True
    ):
        if (row, col) in stopping_positions:
            axes[0].add_patch(
                Rectangle(
                    (col - 0.44, row - 0.44),
                    0.88,
                    0.88,
                    fill=False,
                    edgecolor="#c63632",
                    linewidth=1.5,
                )
            )
        if (row, col) not in ((3, 1), (3, 7), (6, 10)):
            for (dr, dc), probability in zip(moves, probabilities, strict=True):
                axes[0].arrow(
                    col,
                    row,
                    dc * 0.42 * probability,
                    dr * 0.42 * probability,
                    head_width=0.12 * probability,
                    head_length=0.1 * probability,
                    length_includes_head=True,
                    color="#1764ab",
                )
    axes[0].set_title(
        f"Seed {run['seed']}: arrow length ∝ action probability\nRed outline: stop on arrival"
    )
    probabilities = [r["safe_hallway_probability"] for r in report["runs"]]
    axes[1].scatter(report["seeds"], probabilities, s=12, alpha=0.7)
    mean = summary["safe_hallway_probability"]["mean"]
    axes[1].axhline(mean, color="#1764ab", label=f"Mean: {mean:.3f}")
    axes[1].axhline(0.80, color="gray", linestyle="--", label="Mean acceptance: 0.80")
    axes[1].set(
        xlabel="Seed",
        ylabel="P(reach H without penalties)",
        ylim=(-0.03, 1.03),
        title="Actual stochastic policies after training",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        "Milestone 3 — Frozen policy and Equation 9 stopping, independent evaluation"
    )
    fig.savefig(policy, dpi=160)
    plt.close(fig)
    return curves, policy
