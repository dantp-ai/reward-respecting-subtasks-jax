"""Per-model learning curves in native and common feature units."""

from rrs.experiments.tile_coded_models import NAMES


def plot_results(cases, steps, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for key, title, filename in (
        ("reward_rmse", "Environment reward RMSE", "reward_models"),
        (
            "common_successor_rmse",
            "Common-state successor RMSE (decoded vector norm)",
            "successor_models",
        ),
        (
            "native_successor_rmse",
            "Native successor-feature RMSE (different output geometries)",
            "native_successor_models",
        ),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.7), layout="constrained")
        for ax, option in zip(axes.flat, NAMES):
            for name, label in (
                ("one_hot", "One-hot model"),
                ("tile_coding", "Tile-coded model"),
            ):
                stats = cases[name]["summary"][option][key]
                mean, se = stats["mean"], stats["se"]
                ax.plot(steps, mean, label=label)
                ax.fill_between(
                    steps,
                    [max(1e-12, m - e) for m, e in zip(mean, se, strict=True)],
                    [m + e for m, e in zip(mean, se, strict=True)],
                    alpha=0.15,
                )
            ax.set(
                title=option.replace("_", " "),
                xlabel="Model-training transitions",
                yscale="log",
            )
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
        fig.suptitle(
            f"Milestone 11 — {title}\nIdentical fixed options; 100 paired model-training seeds; mean ± SE"
        )
        path = directory / f"milestone_11_{filename}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)
    return outputs
