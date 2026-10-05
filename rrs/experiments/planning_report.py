"""Independent policy returns, frozen comparisons and Milestone 5 figures."""

import math
import platform
from importlib.metadata import version

import jax
import jax.numpy as jnp

from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.two_room import _revision
from rrs.rl.plan_evaluation import evaluate_action_policy, induced_policy

_batched_policy = jax.jit(jax.vmap(induced_policy, in_axes=(0, 0, None, None)))


def first_hit(values, checkpoints, target):
    """Missing hits stay explicit; never average only the successful runs."""
    return next(
        (
            step
            for step, value in zip(checkpoints, values, strict=True)
            if value >= target
        ),
        None,
    )


def summarize_runs(runs):
    summary = {}
    for key in ("planned_start", "value_rmse", "actual_return"):
        columns = [mean_and_se(xs) for xs in zip(*(r[key] for r in runs), strict=True)]
        summary[key] = {k: [c[k] for c in columns] for k in ("mean", "se")}
    hits = [r["first_95_percent"] for r in runs]
    summary["first_95_percent"] = {
        "reached": sum(hit is not None for hit in hits),
        "total": len(hits),
        "mean": mean_and_se(hits)["mean"] if all(h is not None for h in hits) else None,
    }
    summary["final_start_absolute_error"] = mean_and_se(
        [r["final_start_absolute_error"] for r in runs]
    )
    for key in (
        "final_max_value_error",
        "final_actual_error",
        "evaluation_error_bound",
    ):
        summary[key] = max(r[key] for r in runs)
    return summary


def case_report(
    data,
    case,
    result,
    seeds,
    model_seeds,
    environment,
    optimal_values,
    checkpoints,
    evaluation_budgets,
):
    """Keep estimated values and actual environment returns on separate axes."""
    start = data.positions.index((3, 1))
    start_feature = data.nonterminal_indices.index(start)
    optimal = optimal_values[start]
    target = jnp.array([optimal_values[i] for i in data.nonterminal_indices])
    starts = result.history[:, :, start_feature].tolist()
    rmse = jnp.sqrt(jnp.mean((result.history - target) ** 2, axis=-1)).tolist()
    weights = result.weights.tolist()
    returns = [[] for _ in seeds]
    choices = [[] for _ in seeds]
    bounds = [0.0 for _ in seeds]
    reachable = [[] for _ in seeds]
    for budget in evaluation_budgets:
        history_index = checkpoints.index(budget)
        policies, selected = _batched_policy(
            result.history[:, history_index],
            case.models,
            case.option_policies,
            jnp.array(data.features),
        )
        for i, (pi, selection) in enumerate(
            zip(policies.tolist(), selected.tolist(), strict=True)
        ):
            evaluation = evaluate_action_policy(environment, pi, start, data.terminal)
            returns[i].append(evaluation.value)
            choices[i].append(selection)
            reachable[i].append(evaluation.reachable_states)
            bounds[i] = max(bounds[i], evaluation.error_bound)
    runs = [
        {
            "planning_seed": seed,
            "model_seed": model_seeds[i]
            if case.source.startswith("learned_")
            else None,
            "planned_start": starts[i],
            "value_rmse": rmse[i],
            "actual_return": returns[i],
            "selected_models": choices[i],
            "reachable_policy_states": reachable[i],
            "evaluation_error_bound": bounds[i],
            "first_95_percent": first_hit(starts[i], checkpoints, 0.95 * optimal),
            "final_start_absolute_error": abs(starts[i][-1] - optimal),
            "final_actual_error": abs(returns[i][-1] - optimal),
            "final_max_value_error": max(
                abs(w - optimal_values[s])
                for w, s in zip(weights[i], data.nonterminal_indices, strict=True)
            ),
            "final_weights": weights[i],
        }
        for i, seed in enumerate(seeds)
    ]
    n_options = case.models.reward_weights.shape[1]
    terminal_features = jnp.array(data.features)[jnp.array(data.terminal)]
    return {
        "source": case.source,
        "configuration": case.configuration,
        "models_per_backup": n_options,
        "planning_updates_per_run": int(result.updates[0]),
        "planning_lookaheads_per_run": int(result.lookaheads[0]),
        "diagnostic_lookaheads_per_run": len(evaluation_budgets)
        * len(data.positions)
        * n_options,
        "finite": bool(jnp.isfinite(result.history).all())
        and all(math.isfinite(v) for values in returns for v in values),
        "terminal_zero": bool(jnp.all(result.history @ terminal_features.T == 0)),
        "summary": summarize_runs(runs),
        "runs": runs,
    }


def acceptance_checks(cases, optimal):
    """Apply the protocol committed before the first planning experiment."""
    checks = {
        "finite": all(c["finite"] for c in cases.values()),
        "terminal_zero": all(c["terminal_zero"] for c in cases.values()),
        "evaluation_error_bound": all(
            c["summary"]["evaluation_error_bound"] <= 1e-8 for c in cases.values()
        ),
    }
    for name, case in cases.items():
        if case["source"].startswith("exact_"):
            summary = case["summary"]
            checks[f"{name}/all_final_values_within_1e-5"] = (
                summary["final_max_value_error"] <= 1e-5
            )
            checks[f"{name}/all_final_returns_within_1e-5"] = (
                summary["final_actual_error"] <= 1e-5
            )
            hit = summary["first_95_percent"]
            checks[f"{name}/all_reach_95_percent"] = hit["reached"] == hit["total"]
    hits = {
        name: cases[f"exact_optimal/{name}"]["summary"]["first_95_percent"]["mean"]
        for name in ("primitives", "shortest_path", "reward_respecting")
    }
    for alternative in ("primitives", "shortest_path"):
        checks[f"exact_reward_respecting_at_most_80_percent_of_{alternative}"] = (
            hits["reward_respecting"] is not None
            and hits[alternative] is not None
            and hits["reward_respecting"] <= 0.8 * hits[alternative]
        )
    for name in ("primitives", "shortest_path", "reward_respecting"):
        summary = cases[f"learned_50000/{name}"]["summary"]
        for metric in ("planned_start", "actual_return"):
            checks[f"learned_50000/{name}/{metric}_within_0.05"] = (
                abs(summary[metric]["mean"][-1] - optimal) <= 0.05
            )
    earlier = cases["learned_10000/reward_respecting"]["summary"]
    final = cases["learned_50000/reward_respecting"]["summary"]
    checks["learned_reward_respecting_start_error_improves"] = (
        final["final_start_absolute_error"]["mean"]
        < earlier["final_start_absolute_error"]["mean"]
    )
    checks["learned_reward_respecting_return_no_worse_by_0.01"] = (
        final["actual_return"]["mean"][-1]
        >= earlier["actual_return"]["mean"][-1] - 0.01
    )
    return checks


def build_report(
    data, inputs, cases, seeds, checkpoints, evaluation_budgets, optimal_values
):
    revision, dirty = _revision()
    optimal = optimal_values[data.positions.index((3, 1))]
    hits = {
        name: cases[f"exact_optimal/{name}"]["summary"]["first_95_percent"]["mean"]
        for name in ("primitives", "shortest_path", "reward_respecting")
    }
    return {
        "schema_version": 1,
        "milestone": "05-option-planning",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/9",
        "canonical_protocol": seeds == list(range(2000, 2100))
        and checkpoints == list(range(0, 20_001, 100))
        and tuple(evaluation_budgets) == (0, 2000, 5000, 10_000, 20_000),
        "planning_seeds": seeds,
        "gamma": 0.99,
        "alpha": 1.0,
        "initial_weights": 0.0,
        "checkpoints": checkpoints,
        "evaluation_budgets": evaluation_budgets,
        "key_schedule": "key(seed); split(next, state); randint(state, (), 0, 72)",
        "shared_state_prefix": True,
        "model_order": "up, down, left, right, optional hallway option",
        "tie_rule": "first maximum",
        "execution": "reselect a model each primitive step; sample one action from its policy",
        "evaluation": "Python-double discounted linear solve in independent environment; no truncation",
        "evaluation_tolerance": 1e-8,
        "policy_probability_roundoff": "normalize float32 rows in double precision",
        "precision": "float32 planning; Python double policy evaluation",
        "diagnostic_queries": "all states (including zero-feature goal), all models, each evaluation budget; excluded from planning axis",
        "start_error_definition": "mean over runs of abs(final planned start - optimum)",
        "optimal_start_value": optimal,
        "optimal_values": optimal_values,
        "feature_positions": [data.positions[i] for i in data.nonterminal_indices],
        "policy_positions": data.positions,
        "model_provenance": inputs.provenance,
        "cases": cases,
        "shortest_path_to_primitive_first_hit_ratio": (
            hits["shortest_path"] / hits["primitives"]
            if hits["shortest_path"] is not None and hits["primitives"]
            else None
        ),
        "acceptance": acceptance_checks(cases, optimal),
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

    directory.mkdir(parents=True, exist_ok=True)
    labels = {
        "primitives": "Primitives",
        "shortest_path": "+ Shortest-path option",
        "reward_respecting": "+ Reward-respecting option",
    }
    comparisons = [
        (
            "exact_planning",
            "Exact optimal-option models",
            [(f"exact_optimal/{name}", label) for name, label in labels.items()],
        ),
        (
            "learned_planning",
            "Models learned for 50,000 transitions",
            [(f"learned_50000/{name}", label) for name, label in labels.items()],
        ),
        (
            "model_maturity",
            "Reward-respecting set: model-training duration",
            [
                *[
                    (f"learned_{step:05d}/reward_respecting", f"{step:,} transitions")
                    for step in (0, 10_000, 20_000, 50_000)
                ],
                ("exact_frozen/reward_respecting", "Exact model of frozen policy"),
            ],
        ),
    ]
    paths = []
    for suffix, title, series in comparisons:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for ax, metric, x, ylabel in zip(
            axes,
            ("planned_start", "actual_return"),
            (report["checkpoints"], report["evaluation_budgets"]),
            ("Planned value from S", "Actual discounted return from S"),
            strict=True,
        ):
            for name, label in series:
                values = report["cases"][name]["summary"][metric]
                mean, se = values["mean"], values["se"]
                (line,) = ax.plot(
                    x,
                    mean,
                    label=label,
                    marker="o" if metric == "actual_return" else None,
                )
                ax.fill_between(
                    x,
                    [m - e for m, e in zip(mean, se)],
                    [m + e for m, e in zip(mean, se)],
                    color=line.get_color(),
                    alpha=0.18,
                )
            ax.axhline(
                report["optimal_start_value"],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Main-task optimum",
            )
            ax.set(xlabel="Planning look-aheads (one state-model query)", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="best")
        axes[1].set_title("Reselect every primitive step; signed returns")
        axes[0].set_title("Zero initialization; uniform state sampling")
        fig.suptitle(
            f"Milestone 5 — {title}\n{len(report['planning_seeds'])} seeds; shading ±1 standard error",
            fontsize=12,
        )
        path = directory / f"milestone_05_{suffix}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
