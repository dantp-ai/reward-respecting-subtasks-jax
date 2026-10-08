"""Scientific model errors and predictions, Milestone 8."""

import jax.numpy as jnp

from rrs.envs import four_room
from rrs.experiments.four_room_models import MODEL_NAMES


def plot_results(data, report, models, targets, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    curves = directory / "milestone_08_model_learning.png"
    predictions = directory / "milestone_08_option_models.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, names in enumerate((MODEL_NAMES[:4], MODEL_NAMES[4:])):
        for col, metric in enumerate(("reward_rmse", "successor_rmse")):
            ax = axes[row, col]
            for name in names:
                values = report["summary"][name][metric]
                mean, se = values["mean"], values["se"]
                ax.plot(report["checkpoints"], mean, label=name)
                ax.fill_between(
                    report["checkpoints"],
                    [max(1e-12, m - e) for m, e in zip(mean, se, strict=True)],
                    [m + e for m, e in zip(mean, se, strict=True)],
                    alpha=0.15,
                )
            limit = (0.15, 0.40)[col] if row == 0 else 0.25
            ax.axhline(limit, color="black", linestyle="--", label="Final limit")
            ax.set(
                title=("Primitive actions" if row == 0 else "Frozen hallway options"),
                xlabel="Environment transitions",
                yscale="log",
                ylabel="Reward RMSE"
                if col == 0
                else "Successor RMSE (vector norm per source)",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
    fig.suptitle(
        "Milestone 8 — Eight expectation models; 30 model-training seeds, mean ± SE\nFixed Milestone 7 options; αᵣ = αₚ = 0.1, γ = 0.99"
    )
    fig.savefig(curves, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(11, 15), layout="constrained")
    for row, name in enumerate(MODEL_NAMES[4:]):
        feature = data.nonterminal_indices.index(
            data.positions.index(four_room.HALLWAYS[row])
        )
        pairs = (
            (
                targets.reward_weights[4 + row],
                models.reward_weights[:, 4 + row],
                "Environment reward",
            ),
            (
                targets.successor_weights[4 + row, feature],
                models.successor_weights[:, 4 + row, feature],
                "Discounted final target feature",
            ),
        )
        for ax, (exact, samples, label) in zip(axes[row], pairs, strict=True):
            mean = samples.mean(axis=0)
            se = samples.std(axis=0, ddof=1) / jnp.sqrt(samples.shape[0])
            ax.errorbar(
                exact.tolist(),
                mean.tolist(),
                yerr=se.tolist(),
                fmt="o",
                markersize=3,
                capsize=2,
                alpha=0.7,
            )
            low = float(jnp.minimum(exact.min(), mean.min()))
            high = float(jnp.maximum(exact.max(), mean.max()))
            ax.plot(
                [low, high], [low, high], "k--", linewidth=1, label="Exact agreement"
            )
            ax.set(
                title=f"{name}: {label}",
                xlabel="Exact frozen-policy model",
                ylabel="Learned mean ± SE",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
    fig.suptitle(
        "Milestone 8 — Final option models across all 103 nonterminal starts\nSuccessor panels show the target coordinate; learning RMSE includes every coordinate"
    )
    fig.savefig(predictions, dpi=160)
    plt.close(fig)
    return curves, predictions
