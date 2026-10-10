"""Milestone 9 planning and option comparison figures."""


def plot_results(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = report["checkpoints"]
    evaluations = report["evaluation_budgets"]
    optimal = report["optimal_start_value"]
    cases = report["cases"]
    sets = [
        (
            "exact_references",
            "Exact stochastic reference models",
            ["exact_primitive/actions", "exact_optimal_options/actions_and_options", "exact_frozen_options/actions_and_options"],
        ),
        (
            "learned_models",
            "Learned models at the final Milestone 8 checkpoint",
            ["learned_200000/actions", "learned_200000/actions_and_options"],
        ),
        (
            "model_maturity",
            "Model checkpoint maturity with and without options",
            [
                f"learned_{step:06d}/{config}"
                for step in (0, 10_000, 20_000, 30_000, 40_000, 200_000)
                for config in ("actions", "actions_and_options")
            ],
        ),
    ]
    paths = []
    for suffix, title, keys in sets:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for key in keys:
            case = cases[key]
            x = evaluations if key.endswith("actions") and key.startswith("learned_") else None
            for ax, metric, axis in zip(
                axes,
                ("planned_start", "actual_return"),
                (checkpoints, evaluations),
                strict=True,
            ):
                stats = case["summary"][metric]
                label = key.replace("_", " ").replace("/", " — ")
                (line,) = ax.plot(axis, stats["mean"], label=label)
                ax.fill_between(
                    axis,
                    [m - s for m, s in zip(stats["mean"], stats["se"], strict=True)],
                    [m + s for m, s in zip(stats["mean"], stats["se"], strict=True)],
                    color=line.get_color(),
                    alpha=0.16,
                )
        for ax, ylabel in zip(axes, ("Planned value from S", "Actual return from S"), strict=True):
            ax.axhline(optimal, color="black", linestyle="--", linewidth=1, label="Main-task optimum")
            ax.set(xlabel="Planning look-aheads", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="best")
        axes[0].set_title("Estimated value")
        axes[1].set_title("Reselect model at each primitive step")
        fig.suptitle(f"Milestone 9 — {title}\n30 seeds; shading ±1 standard error")
        path = directory / f"milestone_09_{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
