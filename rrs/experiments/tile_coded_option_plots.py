"""Learning, stochastic policies and safe attainment for Milestone 10."""


def plot_results(data, report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from rrs.experiments.hallway import _draw_grid

    directory.mkdir(parents=True, exist_ok=True)
    labels = {"one_hot": "One-hot", "tile_coding": "Tile coding"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, label in labels.items():
        summary = report["cases"][name]["summary"]
        for ax, metric in zip(
            axes, ("critic_rmse", "estimated_start_value"), strict=True
        ):
            mean, se = summary[metric]["mean"], summary[metric]["se"]
            (line,) = ax.plot(report["checkpoints"], mean, label=label)
            ax.fill_between(
                report["checkpoints"],
                [m - e for m, e in zip(mean, se, strict=True)],
                [m + e for m, e in zip(mean, se, strict=True)],
                color=line.get_color(),
                alpha=0.18,
            )
        actual = summary["stochastic_start_value"]
        if actual["mean"] is not None:
            axes[1].errorbar(
                [report["steps"]],
                [actual["mean"]],
                yerr=[actual["se"]],
                fmt="o",
                capsize=4,
                color=line.get_color(),
                label=f"{label}: actual option return",
            )
    for ax, ylabel in zip(
        axes,
        ("Critic RMSE over nonterminal states", "Subtask value from S"),
        strict=True,
    ):
        ax.set(xlabel="Environment transitions", ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[1].axhline(
        report["optimal_start_value"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Exact subtask optimum",
    )
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle(
        f"Milestone 10 — Same hallway target, different prediction features\n{len(report['seeds'])} paired seeds; shading ±1 standard error"
    )
    learning = directory / "milestone_10_option_learning.png"
    fig.savefig(learning, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for ax, (name, label) in zip(axes[:2], labels.items(), strict=True):
        run = report["cases"][name]["runs"][
            0
        ]  # Predetermined seed, not selected by performance.
        _draw_grid(ax, data.params)
        for (row, col), pi, stop in zip(
            data.positions, run["policy_probabilities"], run["stopping"], strict=True
        ):
            if stop:
                ax.add_patch(
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
                for (dr, dc), p in zip(moves, pi, strict=True):
                    ax.arrow(
                        col,
                        row,
                        dc * 0.42 * p,
                        dr * 0.42 * p,
                        head_width=0.12 * p,
                        head_length=0.1 * p,
                        length_includes_head=True,
                        color="#1764ab",
                    )
        ax.set_title(f"{label}, seed {run['seed']}\nRed outline: stop on arrival")
    pairs = list(
        zip(
            report["cases"]["one_hot"]["runs"],
            report["cases"]["tile_coding"]["runs"],
            strict=True,
        )
    )
    evaluated = [
        (a["safe_hallway_probability"], b["safe_hallway_probability"])
        for a, b in pairs
        if a["safe_hallway_probability"] is not None
        and b["safe_hallway_probability"] is not None
    ]
    axes[2].scatter(
        [a for a, _ in evaluated], [b for _, b in evaluated], s=14, alpha=0.6
    )
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1, label="Equal safe attainment")
    axes[2].set(
        xlabel="One-hot safe hallway probability",
        ylabel="Tile-coded safe hallway probability",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        title=f"Independent stochastic option evaluation\n{len(evaluated)}/{len(pairs)} pairs evaluated",
    )
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Milestone 10 — Arrow length ∝ action probability; safe attainment excludes penalties"
    )
    policies = directory / "milestone_10_option_policies.png"
    fig.savefig(policies, dpi=160)
    plt.close(fig)
    return learning, policies
