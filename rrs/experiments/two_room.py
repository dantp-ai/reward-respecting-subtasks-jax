"""Run with: uv run python -m rrs.experiments.two_room."""

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.rl.exact import DeterministicModel, ValueIterationResult, value_iteration


def build_model(params: two_room.Params):
    """Enumerate the eager JAX environment to build an exact action model."""
    positions = two_room.legal_positions(params)
    index = {position: i for i, position in enumerate(positions)}
    next_states, rewards, terminated = [], [], []
    key = jax.random.key(0)
    for position in positions:
        next_row, reward_row, done_row = [], [], []
        state = two_room.State(jnp.array(position, dtype=jnp.int32))
        for action in two_room.Action:
            next_state, _, reward, done = two_room.step(key, state, int(action), params)
            next_row.append(index[tuple(next_state.position.tolist())])
            reward_row.append(float(reward))
            done_row.append(bool(done))
        next_states.append(tuple(next_row))
        rewards.append(tuple(reward_row))
        terminated.append(tuple(done_row))
    return positions, DeterministicModel(
        tuple(next_states), tuple(rewards), tuple(terminated)
    )


def greedy_route(
    params: two_room.Params,
    positions: tuple[tuple[int, int], ...],
    result: ValueIterationResult,
    seed: int = 0,
):
    """Execute the first maximizing action (UP, DOWN, LEFT, RIGHT tie order)."""
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    state = two_room.reset(reset_key, params)
    route = [tuple(state.position.tolist())]
    index = {position: i for i, position in enumerate(positions)}
    actions, rewards = [], []
    for _ in positions:
        action = result.greedy_actions[index[route[-1]]][0]
        key, step_key = jax.random.split(key)
        state, _, reward, done = two_room.step(step_key, state, action, params)
        route.append(tuple(state.position.tolist()))
        actions.append(action)
        rewards.append(float(reward))
        if bool(done):
            return route, actions, rewards
    raise RuntimeError("Greedy route did not reach the goal within the state count")


def plot_route(params, route, path: Path):
    """Save a standalone scientific plot; no interactive display is required."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    walls = params.walls.tolist()
    penalties = params.penalties.tolist()
    grid = [
        [0 if wall else 2 if penalties[r][c] else 1 for c, wall in enumerate(row)]
        for r, row in enumerate(walls)
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        grid, cmap=ListedColormap(["#222222", "#ffffff", "#bdbdbd"]), vmin=0, vmax=2
    )
    ax.plot(
        [c for r, c in route],
        [r for r, c in route],
        "o-",
        color="#1764ab",
        markersize=4,
    )
    for label, (row, column) in (
        ("S", params.start.tolist()),
        ("H", (3, 7)),
        ("G", params.goal.tolist()),
    ):
        ax.text(
            column,
            row,
            label,
            ha="center",
            va="center",
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
    ax.set_xticks(range(len(walls[0])))
    ax.set_yticks(range(len(walls)))
    ax.set_xticks([c - 0.5 for c in range(len(walls[0]) + 1)], minor=True)
    ax.set_yticks([r - 0.5 for r in range(len(walls) + 1)], minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set(
        xlabel="Column",
        ylabel="Row",
        title=f"Exact optimal route: {len(route) - 1} actions, no penalty",
    )
    ax.legend(
        handles=[Patch(facecolor="#bdbdbd", label="Penalty on arrival: −1")],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _revision():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
        )
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/two_room"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    params = two_room.default_params()
    positions, model = build_model(params)
    gamma, tolerance = 0.99, 1e-12
    result = value_iteration(model, gamma=gamma, tolerance=tolerance)
    route, actions, rewards = greedy_route(params, positions, result, seed=args.seed)
    start_value = result.values[positions.index(tuple(params.start.tolist()))]
    actual_return = sum(gamma**t * reward for t, reward in enumerate(rewards))
    if (
        abs(start_value - gamma**17) > tolerance
        or abs(actual_return - start_value) > tolerance
        or len(actions) != 18
        or any(reward < 0 for reward in rewards)
    ):
        raise RuntimeError("Two-room baseline failed milestone acceptance checks")
    revision, dirty = _revision()
    report = {
        "paper_target": "Section 1, Figures 1–2: start value gamma**17",
        "seed": args.seed,
        "independent_runs": 1,
        "deterministic": True,
        "gamma": gamma,
        "tolerance": tolerance,
        "max_iterations": 10_000,
        "initial_values": 0.0,
        "solver_precision": "Python double precision",
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("jax", "jaxlib", "matplotlib")
        },
        "nonterminal_states": len(positions) - 1,
        "iterations": result.iterations,
        "bellman_residual": result.bellman_residual,
        "start_value": start_value,
        "expected_start_value": gamma**17,
        "route_return": actual_return,
        "route": route,
        "actions": [two_room.Action(action).name for action in actions],
        "rewards": rewards,
        "states": [
            {
                "position": p,
                "value": v,
                "greedy_actions": [two_room.Action(a).name for a in actions],
            }
            for p, v, actions in zip(
                positions, result.values, result.greedy_actions, strict=True
            )
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_route(params, route, args.output_dir / "optimal_route.png")
    print(f"Start value: {start_value:.15f} (target {gamma**17:.15f})")
    print(
        f"Bellman residual: {result.bellman_residual:.3g}; iterations: {result.iterations}"
    )
    print(f"Route: {len(actions)} actions; output: {args.output_dir}")


if __name__ == "__main__":
    main()
