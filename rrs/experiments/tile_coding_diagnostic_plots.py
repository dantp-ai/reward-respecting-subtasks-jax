"""Figures for the separately declared Milestone 10 investigation."""


def _values(stats, key):
    return [s[key] if s[key] is not None else float("nan") for s in stats]


def plot_results(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    cases = report["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name, label in (("one_hot", "One-hot"), ("tile_coding", "Tile coding")):
        for ax, x, keys in (
            (
                axes[0],
                [50_000, 100_000, 150_000, 200_000],
                [f"{name}/{t}" for t in (50_000, 100_000, 150_000, 200_000)],
            ),
            (
                axes[1],
                [0.025, 0.05, 0.1, 0.2],
                [
                    f"{name}_alpha_{a}/50000" if a != 0.1 else f"{name}/50000"
                    for a in (0.025, 0.05, 0.1, 0.2)
                ],
            ),
        ):
            stats = [cases[key]["summary"]["safe_hallway_probability"] for key in keys]
            ax.errorbar(
                x,
                _values(stats, "mean"),
                yerr=_values(stats, "se"),
                marker="o",
                capsize=4,
                label=label,
            )
    axes[0].set(
        xlabel="Training transitions (both step sizes 0.1)",
        title="More training at fixed step sizes",
    )
    axes[1].set(
        xlabel="Joint actor/critic step size (50,000 steps)",
        title="Complete declared step-size comparison",
        xscale="log",
    )
    axes[1].set_xticks([0.025, 0.05, 0.1, 0.2], ["0.025", "0.05", "0.1", "0.2"])
    for ax in axes:
        ax.set(ylabel="Safe hallway attainment probability", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "Milestone 10 investigation — 100 paired seeds; error bars ±1 standard error"
    )
    if report["evaluation_status"]["unresolved"]:
        fig.text(
            0.5,
            0,
            "Missing points indicate unresolved evaluation; see report",
            ha="center",
        )
    sensitivity = directory / "milestone_10_investigation_sensitivity.png"
    fig.savefig(sensitivity, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    names = [
        "one_hot",
        "tile_coding",
        "tiled_critic_only",
        "tiled_actor_only",
        "split_penalty_tiles",
    ]
    stats = [
        cases[f"{name}/50000"]["summary"]["safe_hallway_probability"] for name in names
    ]
    axes[0].bar(
        range(len(names)),
        _values(stats, "mean"),
        yerr=_values(stats, "se"),
        capsize=4,
        color=["C0", "C1", "C2", "C3", "C4"],
    )
    axes[0].set_xticks(
        range(len(names)),
        [
            "One-hot",
            "Tiles",
            "Tiled critic\nOne-hot actor",
            "One-hot critic\nTiled actor",
            "Tiles split at\npenalty boundary",
        ],
        rotation=20,
    )
    axes[0].set_title(
        "Representations used during learning\n50,000 steps; both step sizes 0.1"
    )
    combinations = [
        (p, b) for p in ("one_hot", "tile_coding") for b in ("one_hot", "tile_coding")
    ]
    stats = [
        report["policy_stopping_swaps"][f"policy={p},stopping={b}"]["summary"][
            "safe_hallway_probability"
        ]
        for p, b in combinations
    ]
    axes[1].bar(
        range(4),
        _values(stats, "mean"),
        yerr=_values(stats, "se"),
        capsize=4,
        color=["C0", "C2", "C3", "C1"],
    )
    axes[1].set_xticks(
        range(4),
        [
            "One-hot policy\nOne-hot stops",
            "One-hot policy\nTile stops",
            "Tile policy\nOne-hot stops",
            "Tile policy\nTile stops",
        ],
        rotation=20,
    )
    axes[1].set_title(
        "Swap final stopping maps without retraining\nOriginal 50,000-step policies"
    )
    for ax in axes:
        ax.set(ylabel="Safe hallway attainment probability", ylim=(0, 1.03))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Milestone 10 investigation — Learning components and final stopping; 100 paired seeds"
    )
    if report["evaluation_status"]["unresolved"]:
        fig.text(
            0.5,
            0,
            "Missing bars indicate unresolved evaluation; see report",
            ha="center",
        )
    components = directory / "milestone_10_investigation_components.png"
    fig.savefig(components, dpi=160)
    plt.close(fig)
    return sensitivity, components
