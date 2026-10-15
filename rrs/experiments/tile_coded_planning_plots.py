"""Signed value and actual-return figures; missing evaluations remain visible."""

import numpy as np

from rrs.experiments.tile_coded_planning import CONFIGURATIONS


def plot_results(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for source, suffix in (
        ("exact", "exact_models"),
        ("learned_50000", "learned_models"),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
        for col, option in enumerate(CONFIGURATIONS):
            for row, metric in enumerate(("planned_start", "actual_return")):
                ax = axes[row, col]
                for rep, label in (("one_hot", "One-hot"), ("tile_coding", "Tiles")):
                    stats = report["cases"][f"{rep}/{source}/{option}"]["summary"][
                        metric
                    ]
                    x = (
                        report["checkpoints"]
                        if row == 0
                        else report["evaluation_budgets"]
                    )
                    mean, se = (
                        np.array(stats["mean"], dtype=float),
                        np.array(stats["se"], dtype=float),
                    )
                    ax.plot(x, mean, label=label)
                    ax.fill_between(x, mean - se, mean + se, alpha=0.15)
                ax.axhline(
                    report["optimal_start_value"],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                )
                ax.set(
                    title=option.replace("_", " "),
                    xlabel="Planning look-aheads",
                    ylabel="Planned value" if row == 0 else "Actual task return",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle(
            f"Milestone 12 — {source.replace('_', ' ')}; 100 paired seeds, mean ± SE\nSigned values; gaps indicate unresolved metrics"
        )
        path = directory / f"milestone_12_{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    for row, rep in enumerate(("one_hot", "tile_coding")):
        for col, option in enumerate(CONFIGURATIONS):
            ax = axes[row, col]
            for step in (0, 10000, 20000, 50000):
                stats = report["cases"][f"{rep}/learned_{step:05d}/{option}"][
                    "summary"
                ]["actual_return"]
                mean, se = (
                    np.array(stats["mean"], dtype=float),
                    np.array(stats["se"], dtype=float),
                )
                ax.plot(
                    report["evaluation_budgets"],
                    mean,
                    marker="o",
                    label=f"{step:,} model steps",
                )
                ax.fill_between(
                    report["evaluation_budgets"], mean - se, mean + se, alpha=0.12
                )
            ax.axhline(
                report["optimal_start_value"],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax.set(
                title=f"{rep}: {option}".replace("_", " "),
                xlabel="Planning look-aheads",
                ylabel="Actual task return",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Milestone 12 — Model maturity; 100 paired seeds, mean ± SE\nReselect at every primitive step; signed returns, no clipping"
    )
    path = directory / "milestone_12_model_maturity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
