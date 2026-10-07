"""Scientific learning curves and four stochastic option maps for Milestone 7."""

import jax
import jax.numpy as jnp

from rrs.envs import four_room
from rrs.experiments.four_room_option_report import policy_tables
from rrs.experiments.four_room_options import OPTION_NAMES, STEPS


def plot_results(data, report, snapshots, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric, actual_metric, suffix, ylabel in (
        (
            "critic_rmse",
            "actual_value_rmse",
            "learning",
            "RMSE across 103 nonterminal states",
        ),
        ("estimated_start", "actual_start", "start_values", "Subtask return from S"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for ax, name in zip(axes.flat, OPTION_NAMES, strict=True):
            summary = report["summary"][name]
            mean, se = summary[metric]["mean"], summary[metric]["se"]
            x = report["checkpoints"]
            ax.plot(x, mean, label="Critic estimate")
            ax.fill_between(
                x,
                [m - e for m, e in zip(mean, se)],
                [m + e for m, e in zip(mean, se)],
                alpha=0.2,
                label="±1 SE",
            )
            steps = report["evaluation_steps"]
            values = [
                summary["evaluations"][str(step)][actual_metric] for step in steps
            ]
            ax.errorbar(
                steps,
                [v["mean"] for v in values],
                yerr=[v["se"] for v in values],
                fmt="o--",
                capsize=4,
                label="Actual stochastic option",
            )
            ax.axvline(
                200_000, color="gray", linestyle=":", label="Paper-duration checkpoint"
            )
            target = (
                0.40
                if metric == "critic_rmse"
                else report["references"][name]["metrics"]["actual_start"]
            )
            ax.axhline(
                target,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Final RMSE limit"
                if metric == "critic_rmse"
                else "Exact subtask optimum",
            )
            ax.set(title=name, xlabel="Environment transitions", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(
            f"Milestone 7 — Four reward-respecting options; {len(report['seeds'])} seeds\nα = α_actor = 0.05; fixed 1,000,000-step convergence budget"
        )
        path = directory / f"milestone_07_{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    weights = jax.tree.map(lambda x: x[0], snapshots[STEPS].weights)
    pi, beta = policy_tables(
        weights,
        jnp.array(data.features),
        jnp.array(data.stopping_values),
        jnp.array(data.terminal),
    )
    grid = [
        [0 if cell == "#" else 2 if cell == "P" else 1 for cell in row]
        for row in four_room.LAYOUT
    ]
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    labels = {
        (4, 1): "S",
        (9, 7): "G",
        **{p: name for p, name in zip(four_room.HALLWAYS, OPTION_NAMES, strict=True)},
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), layout="constrained")
    for i, (ax, name) in enumerate(zip(axes.flat, OPTION_NAMES, strict=True)):
        ax.imshow(
            grid, cmap=ListedColormap(["#222222", "white", "#bbbbbb"]), vmin=0, vmax=2
        )
        for (r, c), probabilities, stop in zip(
            data.positions, pi[i].tolist(), beta[i].tolist(), strict=True
        ):
            if stop:
                ax.add_patch(
                    Rectangle(
                        (c - 0.44, r - 0.44),
                        0.88,
                        0.88,
                        fill=False,
                        edgecolor="#b33434",
                        linewidth=1.2,
                    )
                )
            if (r, c) not in labels:
                for (dr, dc), p in zip(moves, probabilities, strict=True):
                    ax.arrow(
                        c,
                        r,
                        dc * 0.43 * p,
                        dr * 0.43 * p,
                        head_width=0.13 * p,
                        head_length=0.1 * p,
                        length_includes_head=True,
                        color="#1764ab",
                    )
        for (r, c), label in labels.items():
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="#b33434" if label == name else "black",
            )
        ax.set(
            title=f"{name}: seed {report['seeds'][0]}",
            xlabel="Column",
            ylabel="Row",
            xticks=range(13),
            yticks=range(13),
        )
        ax.set_xticks([j - 0.5 for j in range(14)], minor=True)
        ax.set_yticks([j - 0.5 for j in range(14)], minor=True)
        ax.grid(which="minor", color="#888888", linewidth=0.3)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(
        "Milestone 7 — Final stochastic policies and Equation 9 stopping\nArrow length ∝ action probability; red outline: stop on arrival"
    )
    path = directory / "milestone_07_options.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
