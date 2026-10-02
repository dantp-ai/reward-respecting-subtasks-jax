"""Exact hallway baselines: uv run python -m rrs.experiments.hallway."""

import argparse
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.experiments.two_room import _revision, build_model
from rrs.representations.tabular import one_hot
from rrs.rl.exact import DeterministicModel
from rrs.rl.option_models import (
    DeterministicOption,
    ExactOptionModel,
    exact_option_model,
)
from rrs.rl.subtasks import Subtask, SubtaskResult, solve_subtask, stopping_value


@dataclass(frozen=True)
class Baselines:
    params: two_room.Params
    positions: tuple[tuple[int, int], ...]
    model: DeterministicModel
    terminal: tuple[bool, ...]
    nonterminal_indices: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    subtasks: dict[str, Subtask]
    solutions: dict[str, SubtaskResult]
    options: dict[str, DeterministicOption]
    models: dict[str, ExactOptionModel]


class Rollout(NamedTuple):
    positions: tuple[tuple[int, int], ...]
    actions: tuple[int, ...]
    rewards: tuple[float, ...]


def build_baselines() -> Baselines:
    """Freeze the illustrative experiment: main-task weights zero, bonus one."""
    params = two_room.default_params()
    positions, model = build_model(params)
    terminal = tuple(p == tuple(params.goal.tolist()) for p in positions)
    nonterminal = tuple(i for i, done in enumerate(terminal) if not done)
    observations = jnp.array([positions[i] for i in nonterminal])
    feature_array = jax.vmap(one_hot, in_axes=(0, None))(
        jnp.array(positions), observations
    )
    features = tuple(tuple(row) for row in feature_array.tolist())
    hallway_feature = nonterminal.index(positions.index((3, 7)))
    main_weights = jnp.zeros(len(nonterminal))
    stopping_values = tuple(
        float(stopping_value(main_weights, x, hallway_feature, bonus=1.0))
        for x in feature_array
    )
    subtasks = {
        "reward_respecting": Subtask(
            model.rewards, stopping_values, (True,) * len(positions)
        ),
        "shortest_path": Subtask(
            tuple((0.0,) * 4 if done else (-1.0,) * 4 for done in terminal),
            (0.0,) * len(positions),
            tuple(
                p == (3, 7) or done for p, done in zip(positions, terminal, strict=True)
            ),
        ),
    }
    solutions = {
        name: solve_subtask(model, task, terminal) for name, task in subtasks.items()
    }
    options = {
        name: DeterministicOption(
            tuple(actions[0] if actions else 0 for actions in result.greedy_actions),
            result.stopping,
        )
        for name, result in solutions.items()
    }
    for action in two_room.Action:
        options[action.name.lower()] = DeterministicOption(
            (int(action),) * len(positions), (True,) * len(positions)
        )
    models = {
        name: exact_option_model(model, option, features, terminal)
        for name, option in options.items()
    }
    return Baselines(
        params,
        positions,
        model,
        terminal,
        nonterminal,
        features,
        subtasks,
        solutions,
        options,
        models,
    )


def option_rollout(
    data: Baselines,
    option: DeterministicOption,
    start: tuple[int, int] = (3, 1),
) -> Rollout:
    """Execute through the JAX environment, checking stopping only after arrival."""
    index = {p: i for i, p in enumerate(data.positions)}
    if data.terminal[index[start]]:
        return Rollout((start,), (), ())
    state = two_room.State(jnp.array(start, dtype=jnp.int32))
    key = jax.random.key(0)
    positions, actions, rewards = [start], [], []
    for _ in data.positions:
        action = option.policy[index[positions[-1]]]
        key, step_key = jax.random.split(key)
        state, _, reward, done = two_room.step(step_key, state, action, data.params)
        positions.append(tuple(state.position.tolist()))
        actions.append(action)
        rewards.append(float(reward))
        if bool(done) or option.stopping[index[positions[-1]]]:
            return Rollout(tuple(positions), tuple(actions), tuple(rewards))
    raise RuntimeError("Option did not terminate within the number of states")


def _draw_grid(ax, params):
    from matplotlib.colors import ListedColormap

    walls, penalties = params.walls.tolist(), params.penalties.tolist()
    grid = [
        [0 if wall else 2 if penalties[r][c] else 1 for c, wall in enumerate(row)]
        for r, row in enumerate(walls)
    ]
    ax.imshow(
        grid, cmap=ListedColormap(["#222222", "white", "#bdbdbd"]), vmin=0, vmax=2
    )
    ax.set_xticks(range(15))
    ax.set_yticks(range(8))
    ax.set_xticks([c - 0.5 for c in range(16)], minor=True)
    ax.set_yticks([r - 0.5 for r in range(9)], minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set(xlabel="Column", ylabel="Row")
    for label, (row, col) in (("S", (3, 1)), ("H", (3, 7)), ("G", (6, 10))):
        ax.text(
            col,
            row,
            label,
            ha="center",
            va="center",
            weight="bold",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )


def plot_comparisons(data: Baselines, directory: Path) -> tuple[Path, Path]:
    """Show routes and full policies, preserving all formal stopping states."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    directory.mkdir(parents=True, exist_ok=True)
    names = ("reward_respecting", "shortest_path")
    titles = ("Reward-respecting", "Shortest path")
    colors = ("#1764ab", "#ae4b22")
    routes_path = directory / "milestone_02_hallway_routes.png"
    policies_path = directory / "milestone_02_hallway_policies.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.7), layout="constrained")
    for ax, name, title, color in zip(axes, names, titles, colors, strict=True):
        _draw_grid(ax, data.params)
        route = option_rollout(data, data.options[name])
        ax.plot(
            [c for r, c in route.positions],
            [r for r, c in route.positions],
            "o-",
            color=color,
            markersize=4,
        )
        cost = sum(r < 0 for r in route.rewards)
        ax.set_title(f"{title}: {len(route.actions)} actions, {cost} penalty arrivals")
    fig.suptitle("Milestone 2 — Exact hallway routes (gray cells cost −1 on arrival)")
    fig.savefig(routes_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.7), layout="constrained")
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for ax, name, title in zip(axes, names, titles, strict=True):
        _draw_grid(ax, data.params)
        option = data.options[name]
        for i, (row, col) in enumerate(data.positions):
            if option.stopping[i]:
                ax.add_patch(
                    Rectangle(
                        (col - 0.44, row - 0.44),
                        0.88,
                        0.88,
                        fill=False,
                        edgecolor="#c63632",
                        linewidth=1.8,
                    )
                )
            if (row, col) not in ((3, 1), (3, 7), (6, 10)):
                dr, dc = moves[option.policy[i]]
                ax.arrow(
                    col - dc * 0.18,
                    row - dr * 0.18,
                    dc * 0.36,
                    dr * 0.36,
                    head_width=0.13,
                    head_length=0.11,
                    length_includes_head=True,
                    color="#333333",
                )
        ax.set_title(f"{title}: {sum(option.stopping)} stopping states")
    fig.suptitle(
        "Milestone 2 — Arrows: action on initiation; red outline: stop on arrival"
    )
    fig.savefig(policies_path, dpi=160)
    plt.close(fig)
    return routes_path, policies_path


def baseline_report(data: Baselines, figures: tuple[Path, Path]) -> dict:
    """Keep state/feature ordering and full exact targets available for later RMSE."""
    revision, dirty = _revision()
    start = data.positions.index((3, 1))
    comparisons = {}
    for name, solution in data.solutions.items():
        route = option_rollout(data, data.options[name])
        comparisons[name] = {
            "start_value": solution.values[start],
            "iterations": solution.iterations,
            "bellman_residual": solution.bellman_residual,
            "route": route.positions,
            "actions": [two_room.Action(a).name for a in route.actions],
            "environment_rewards": route.rewards,
            "environment_return": data.models[name].rewards[start],
            "discounted_successor_mass": sum(data.models[name].successors[start]),
            "values": solution.values,
            "greedy_actions": solution.greedy_actions,
            "stopping_states": [
                p
                for p, stop in zip(data.positions, solution.stopping, strict=True)
                if stop
            ],
        }
    return {
        "milestone": "02-exact-hallway-options",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/3",
        "figures": [str(p) for p in figures],
        "seed": 0,
        "independent_runs": 1,
        "deterministic": True,
        "gamma": 0.99,
        "bonus_weight": 1.0,
        "main_task_weights": [0.0] * 72,
        "bellman_tolerance": 1e-12,
        "max_iterations": 10_000,
        "action_tie_tolerance": 1e-12,
        "initial_subtask_values": 0.0,
        "solver_precision": "Python double precision",
        "option_minimum_duration": 1,
        "stopping_bonus_discount": "gamma**(K-1), Equation 2",
        "successor_feature_discount": "gamma**K, Equation 15",
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
        "positions": data.positions,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "terminal_states": data.terminal,
        "comparisons": comparisons,
        "models": {
            name: {
                "policy": data.options[name].policy,
                "stopping": data.options[name].stopping,
                "reward": model.rewards,
                "successor_features": model.successors,
            }
            for name, model in data.models.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hallway"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data = build_baselines()
    start = data.positions.index((3, 1))
    if abs(data.solutions["reward_respecting"].values[start] - 0.99**11) > 1e-12:
        raise RuntimeError("Hallway subtask does not match the paper target")
    for name, steps, penalties in (
        ("reward_respecting", 12, 0),
        ("shortest_path", 6, 4),
    ):
        route = option_rollout(data, data.options[name])
        if (
            len(route.actions) != steps
            or sum(r < 0 for r in route.rewards) != penalties
            or route.positions[-1] != (3, 7)
        ):
            raise RuntimeError(f"{name} route failed acceptance checks")
    figures = plot_comparisons(data, args.figures_dir)
    report = baseline_report(data, figures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["comparisons"].items():
        print(
            f"{name}: value={result['start_value']:.15f}; environment return={result['environment_return']:.6f}; residual={result['bellman_residual']:.3g}"
        )
    print(f"Report: {args.output_dir / 'baseline.json'}")
    for figure in figures:
        print(f"Figure: {figure}")


if __name__ == "__main__":
    main()
