"""Four-room primitive baseline: uv run python -m rrs.experiments.four_room."""

import argparse
import json
import math
import platform
import random
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

from rrs.envs import four_room as env, four_room_reference as reference
from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.two_room import _revision
from rrs.rl.stochastic_exact import (
    StochasticModel,
    Transition,
    evaluate_policy,
    value_iteration,
)

AUDIT_STARTS = ((4, 1), (6, 2), (7, 9), (9, 3))
AUDIT_SEEDS = (6000, 6001, 6002, 6003)


def build_model():
    """Exact double-precision probabilities from the independent Python model."""
    positions = reference.legal_positions()
    index = {p: i for i, p in enumerate(positions)}
    transitions = tuple(
        tuple(
            tuple(
                Transition(o.probability, index[o.position], o.reward, o.terminated)
                for o in reference.distribution(p, a)
            )
            for a in range(4)
        )
        for p in positions
    )
    return positions, StochasticModel(transitions)


class Episode(NamedTuple):
    discounted_return: float
    steps: int
    truncated: bool


def rollout(start, policy, rng, gamma=0.99, max_steps=3000) -> Episode:
    """Sample true directions with the reference's independent nine-ticket RNG."""
    if not 0 <= gamma < 1 or max_steps < 1:
        raise ValueError("Require gamma in [0,1) and a positive step limit")
    if start == (9, 7):
        return Episode(0.0, 0, False)
    position, total, discount = start, 0.0, 1.0
    for t in range(max_steps):
        position, reward, done = reference.step(position, policy[position], rng)
        total += discount * reward
        discount *= gamma
        if done:
            return Episode(total, t + 1, False)
    return Episode(total, max_steps, True)


def audit_policy(
    positions,
    actions,
    values,
    episodes=20_000,
    max_steps=3000,
    gamma=0.99,
    progress=False,
):
    """Fixed starts and independent streams; preserve every signed return."""
    if episodes < 2:
        raise ValueError("An audit requires at least two episodes per start")
    if not 0 <= gamma < 1 or max_steps < 1:
        raise ValueError("Require gamma in [0,1) and a positive step limit")
    policy = dict(zip(positions, actions, strict=True))
    index = {p: i for i, p in enumerate(positions)}
    tail_bound = gamma**max_steps / (1 - gamma)
    audits = []
    for start, seed in zip(AUDIT_STARTS, AUDIT_SEEDS, strict=True):
        rng = random.Random(seed)
        samples = [
            rollout(start, policy, rng, gamma, max_steps) for _ in range(episodes)
        ]
        returns = [s.discounted_return for s in samples]
        moments = mean_and_se(returns)
        exact = values[index[start]]
        tolerance = 0.005 + 6 * moments["se"] + tail_bound
        audits.append(
            {
                "start": start,
                "seed": seed,
                "episodes": episodes,
                "gamma": gamma,
                "max_steps": max_steps,
                "returns": returns,
                "episode_lengths": [s.steps for s in samples],
                "truncations": sum(s.truncated for s in samples),
                "mean": moments["mean"],
                "se": moments["se"],
                "oracle_value": exact,
                "absolute_error": abs(moments["mean"] - exact),
                "tail_bound": tail_bound,
                "tolerance": tolerance,
                "passed": abs(moments["mean"] - exact) <= tolerance,
            }
        )
        if progress:
            print(
                f"Audited {episodes:,} episodes from {start}: mean={moments['mean']:.6f}, oracle={exact:.6f}, SE={moments['se']:.6f}",
                flush=True,
            )
    return audits


def build_report(positions, model, solution, evaluation, audits):
    revision, dirty = _revision()
    value_difference = max(
        abs(a - b) for a, b in zip(solution.values, evaluation.values, strict=True)
    )
    bound = solution.bellman_residual / (1 - 0.99)
    goal = positions.index((9, 7))
    checks = {
        "finite_values_and_returns": all(
            math.isfinite(v) for v in (*solution.values, *evaluation.values)
        )
        and all(math.isfinite(v) for audit in audits for v in audit["returns"]),
        "goal_value_zero": solution.values[goal] == evaluation.values[goal] == 0.0,
        "optimal_value_error_bound_at_most_1e-10": bound <= 1e-10,
        "policy_error_bound_at_most_1e-10": evaluation.error_bound <= 1e-10,
        "policy_values_match_optimum_within_1e-9": value_difference <= 1e-9,
        "all_four_rollout_audits_pass": len(audits) == 4
        and all(a["passed"] for a in audits),
        "no_truncated_episodes": all(a["truncations"] == 0 for a in audits),
    }
    return {
        "milestone": "06-four-room-foundation",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/11",
        "paper_target": "Section 7 / Figure 6: stochastic four-room primitive baseline",
        "canonical_protocol": len(audits) == 4
        and all(
            a["episodes"] == 20_000
            and a["gamma"] == 0.99
            and a["max_steps"] == 3000
            and tuple(a["start"]) == start
            and a["seed"] == seed
            for a, start, seed in zip(audits, AUDIT_STARTS, AUDIT_SEEDS, strict=True)
        ),
        "layout": env.LAYOUT,
        "start": [4, 1],
        "goal": [9, 7],
        "hallways": env.HALLWAYS,
        "nonterminal_states": len(positions) - 1,
        "feature_positions": [p for p in positions if p != (9, 7)],
        "reward_timing": "arrival: goal +1, penalty -1, otherwise 0; absorbing goal reward 0",
        "motion_probabilities": [
            [2 / 3 if a == direction else 1 / 9 for direction in range(4)]
            for a in range(4)
        ],
        "gamma": 0.99,
        "residual_tolerance": 1e-12,
        "max_iterations": 20_000,
        "initial_values": 0.0,
        "precision": "Python double oracles and rollouts; float32 JAX environment",
        "tie_rule": "first action within 1e-12 of maximum; UP, DOWN, LEFT, RIGHT",
        "iterations": solution.iterations,
        "bellman_residual": solution.bellman_residual,
        "value_error_bound": bound,
        "policy_evaluation_error_bound": evaluation.error_bound,
        "policy_evaluation_residual": evaluation.bellman_residual,
        "max_policy_value_difference": value_difference,
        "optimal_start_value": solution.values[positions.index((4, 1))],
        "rollout_max_steps": audits[0]["max_steps"],
        "audit_seeds": AUDIT_SEEDS,
        "rollout_sampling": "random.Random(seed); randrange(9); 0..5 intended, 6..8 other directions in action order",
        "audit_tolerance": "0.005 + 6*sample_standard_error + gamma**max_steps/(1-gamma)",
        "states": [
            {
                "position": p,
                "value": v,
                "greedy_actions": actions,
                "policy_value": evaluated,
            }
            for p, v, actions, evaluated in zip(
                positions,
                solution.values,
                solution.greedy_actions,
                evaluation.values,
                strict=True,
            )
        ],
        "transition_axes": ["state", "action", "outcome"],
        "transitions": [
            [[o._asdict() for o in outcomes] for outcomes in row]
            for row in model.transitions
        ],
        "audits": audits,
        "acceptance": checks,
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
    }


def plot_results(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    directory.mkdir(parents=True, exist_ok=True)
    baseline_path = directory / "milestone_06_four_room_baseline.png"
    audit_path = directory / "milestone_06_four_room_return_audit.png"
    grid = [
        [0 if c == "#" else 2 if c == "P" else 1 for c in row]
        for row in report["layout"]
    ]
    values = [[math.nan] * 13 for _ in range(13)]
    labels = {
        tuple(report["start"]): "S",
        tuple(report["goal"]): "G",
        **{tuple(p): f"H{i + 1}" for i, p in enumerate(report["hallways"])},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    axes[0].imshow(
        grid, cmap=ListedColormap(["#222222", "white", "#bbbbbb"]), vmin=0, vmax=2
    )
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for state in report["states"]:
        row, col = state["position"]
        values[row][col] = state["value"]
        if (row, col) not in labels:
            dr, dc = moves[state["greedy_actions"][0]]
            axes[0].arrow(
                col - 0.22 * dc,
                row - 0.22 * dr,
                0.44 * dc,
                0.44 * dr,
                head_width=0.15,
                head_length=0.12,
                length_includes_head=True,
                color="#1764ab",
            )
    for (row, col), label in labels.items():
        axes[0].text(
            col,
            row,
            label,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="#176422" if label == "S" else "black",
        )
    palette = plt.get_cmap("viridis").with_extremes(bad="#222222")
    heatmap = axes[1].imshow(values, cmap=palette)
    fig.colorbar(heatmap, ax=axes[1], label="Optimal discounted value")
    for ax in axes:
        ax.set(xticks=range(13), yticks=range(13), xlabel="Column", ylabel="Row")
        ax.set_xticks([i - 0.5 for i in range(14)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(14)], minor=True)
        ax.grid(which="minor", color="#888888", linewidth=0.35)
        ax.tick_params(which="minor", length=0)
    axes[0].set_title("First greedy action; gray cells reward −1")
    axes[1].set_title(
        f"Exact stochastic values; V(S) = {report['optimal_start_value']:.6f}"
    )
    fig.suptitle(
        "Milestone 6 — Four-room primitive baseline\nIntended direction 2/3; each other direction 1/9; γ = 0.99"
    )
    fig.savefig(baseline_path, dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = [
        f"{label}\n{tuple(audit['start'])}"
        for label, audit in zip(
            ("S", "H1", "H3", "Penalty"), report["audits"], strict=True
        )
    ]
    x = list(range(len(labels)))
    errors = [a["mean"] - a["oracle_value"] for a in report["audits"]]
    for ax, metric in zip(axes, ("returns", "errors"), strict=True):
        ys = [a["mean"] for a in report["audits"]] if metric == "returns" else errors
        ax.errorbar(
            x,
            ys,
            yerr=[6 * a["se"] for a in report["audits"]],
            fmt="o",
            capsize=5,
            label="Rollout mean ±6 SE",
        )
        if metric == "returns":
            ax.scatter(
                x,
                [a["oracle_value"] for a in report["audits"]],
                marker="x",
                s=70,
                color="black",
                label="Exact oracle",
            )
        else:
            ax.axhline(0, color="black", linewidth=1)
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Signed discounted return"
            if metric == "returns"
            else "Rollout mean − oracle value",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    count = report["audits"][0]["episodes"]
    truncated = sum(a["truncations"] for a in report["audits"])
    fig.suptitle(
        f"Milestone 6 — Independent greedy-policy return audit\n{count:,} episodes per start; {truncated} truncated episodes"
    )
    fig.savefig(audit_path, dpi=170)
    plt.close(fig)
    return baseline_path, audit_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/four_room"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    positions, model = build_model()
    solution = value_iteration(model)
    actions = [row[0] for row in solution.greedy_actions]
    pi = tuple(tuple(float(a == chosen) for a in range(4)) for chosen in actions)
    evaluation = evaluate_policy(model, pi)
    audits = audit_policy(positions, actions, solution.values, progress=True)
    report = build_report(positions, model, solution, evaluation, audits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "baseline.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = plot_results(report, args.figures_dir)
    print(
        json.dumps(
            {
                "report": str(path),
                "figures": [str(p) for p in paths],
                "optimal_start_value": report["optimal_start_value"],
                "acceptance": report["acceptance"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not all(report["acceptance"].values()):
        raise RuntimeError(
            "Frozen acceptance criteria failed; results preserved for diagnosis"
        )


if __name__ == "__main__":
    main()
