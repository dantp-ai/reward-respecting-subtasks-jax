"""Milestone 12: feature-facing planning with frozen Milestone 11 models."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.planning import run_plan, sample_states
from rrs.experiments.tile_coded_model_report import save_json
from rrs.rl.plan_evaluation import evaluate_action_policy
from rrs.rl.planning import backed_up_values

SEEDS = tuple(range(12000, 12100))
CONFIGURATIONS = {
    "primitives": (0, 1, 2, 3),
    "shortest_path": (0, 1, 2, 3, 4),
    "one_hot_option": (0, 1, 2, 3, 5),
    "tile_option": (0, 1, 2, 3, 6),
}
BUDGETS = (0, 2000, 5000, 10_000, 20_000)
CHECKPOINTS = tuple(range(0, 20_001, 100))


@partial(jax.jit, static_argnames=("budget", "checkpoint"))
def plan_batch(models, orders, features, budget=20_000, checkpoint=100):
    return jax.vmap(lambda m, s: run_plan(m, s, budget, checkpoint, 1.0, features))(
        models, orders
    )


@jax.jit
def policies_with_validity(weights, models, policies, features):
    def one(w, m):
        q = jax.vmap(backed_up_values, in_axes=(None, 0, None))(w, features, m)
        choices = jnp.argmax(q, axis=1)
        return (
            policies[choices, jnp.arange(len(features))],
            choices,
            jnp.isfinite(q).all(),
        )

    return jax.vmap(one)(weights, models)


def safe_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summary_columns(rows):
    return {
        stat: [
            mean_and_se(column)[stat] if all(v is not None for v in column) else None
            for column in zip(*rows, strict=True)
        ]
        for stat in ("mean", "se")
    }


def hit_summary(values, budgets, target):
    hits = [
        next(
            (
                b
                for b, v in zip(budgets, row, strict=True)
                if v is not None and v >= target
            ),
            None,
        )
        for row in values
    ]
    return {
        "per_seed": hits,
        "reached": sum(v is not None for v in hits),
        "total": len(hits),
        "mean": mean_and_se(hits)["mean"] if all(v is not None for v in hits) else None,
    }


def explicit_nonfinite(value):
    """Parameter archives retain NaN/Infinity explicitly as strings in valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [explicit_nonfinite(v) for v in value]
    return value


def report_case(
    data,
    models,
    result,
    features,
    policies,
    indices,
    environment,
    optimum,
    seeds=SEEDS,
    checkpoints=CHECKPOINTS,
    budgets=BUDGETS,
    cache=None,
):
    cache = {} if cache is None else cache
    full_x = np.asarray(features, dtype=float)
    weights = np.asarray(result.history, dtype=float)
    with np.errstate(invalid="ignore", over="ignore"):
        values = weights @ full_x.T
        rmse = np.sqrt(
            np.mean(
                (
                    values[:, :, list(data.nonterminal_indices)]
                    - np.array(optimum)[list(data.nonterminal_indices)]
                )
                ** 2,
                axis=-1,
            )
        )
    start = data.positions.index((3, 1))
    runs = [
        {
            "planning_seed": seed,
            "planned_start": [safe_number(v) for v in values[i, :, start]],
            "value_rmse": [safe_number(v) for v in rmse[i]],
            "actual_return": [],
            "selected_models": [],
            "evaluation_bounds": [],
            "evaluation_errors": [],
            "trajectory_finite": bool(np.isfinite(weights[i]).all()),
            "final_max_value_error": safe_number(
                np.max(np.abs(values[i, -1] - optimum))
            ),
        }
        for i, seed in enumerate(seeds)
    ]
    for budget in budgets:
        k = checkpoints.index(budget)
        pi, choices, finite = policies_with_validity(
            result.history[:, k], models, policies, features
        )
        for run, p, chosen, valid in zip(
            runs, pi.tolist(), choices.tolist(), finite.tolist(), strict=True
        ):
            global_choices = [indices[c] for c in chosen]
            error, evaluation = None, None
            if not valid:
                error = "Nonfinite model backups; policy not evaluated"
            else:
                key = tuple(global_choices)
                try:
                    if key not in cache:
                        cache[key] = evaluate_action_policy(
                            environment, p, start, data.terminal
                        )
                    evaluation = cache[key]
                except (ValueError, RuntimeError) as exception:
                    error = str(exception)
            run["actual_return"].append(evaluation.value if evaluation else None)
            run["selected_models"].append(global_choices if valid else None)
            run["evaluation_bounds"].append(
                evaluation.error_bound if evaluation else None
            )
            run["evaluation_errors"].append(error)
    summary = {
        key: summary_columns([r[key] for r in runs])
        for key in ("planned_start", "value_rmse", "actual_return")
    }
    target = 0.95 * optimum[start]
    summary["planned_first_95_percent"] = hit_summary(
        [r["planned_start"] for r in runs], checkpoints, target
    )
    summary["actual_first_95_percent"] = hit_summary(
        [r["actual_return"] for r in runs], budgets, target
    )
    return {
        "runs": runs,
        "summary": summary,
        "models_per_backup": len(indices),
        "updates_per_run": int(result.updates[0]),
        "lookaheads_per_run": int(result.lookaheads[0]),
        "diagnostic_queries_per_run": len(budgets) * len(data.positions) * len(indices),
        "trajectory_finite": all(r["trajectory_finite"] for r in runs),
        "terminal_zero": bool(np.all(values[:, :, np.array(data.terminal)] == 0)),
        "evaluation_failures": sum(
            e is not None for r in runs for e in r["evaluation_errors"]
        ),
    }


def performance_checks(cases, optimum):
    checks = {}
    for name, case in cases.items():
        checks[f"{name}/finite"] = (
            case["trajectory_finite"] and case["evaluation_failures"] == 0
        )
        if case["source"] == "exact":
            checks[f"{name}/all_values_within_1e-5"] = all(
                r["final_max_value_error"] is not None
                and r["final_max_value_error"] <= 1e-5
                for r in case["runs"]
            )
            checks[f"{name}/all_returns_within_1e-5"] = all(
                r["actual_return"][-1] is not None
                and abs(r["actual_return"][-1] - optimum) <= 1e-5
                for r in case["runs"]
            )
        elif case["source"] == "learned_50000":
            for metric in ("planned_start", "actual_return"):
                mean = case["summary"][metric]["mean"][-1]
                checks[f"{name}/{metric}_within_0.05"] = (
                    mean is not None and abs(mean - optimum) <= 0.05
                )
    for representation in ("one_hot", "tile_coding"):
        for option in ("one_hot_option", "tile_option"):
            a, b = [
                cases[f"{representation}/learned_{step:05d}/{option}"]["summary"][
                    "actual_return"
                ]["mean"][-1]
                for step in (10000, 50000)
            ]
            checks[f"{representation}/{option}/maturity_return"] = (
                a is not None and b is not None and b >= a - 0.01
            )
    return checks


def paired_comparisons(cases):
    paired = {}
    for name, one in cases.items():
        if not name.startswith("one_hot/"):
            continue
        suffix = name.removeprefix("one_hot/")
        tile = cases[f"tile_coding/{suffix}"]
        paired[suffix] = {}
        for key in ("planned_start", "value_rmse", "actual_return"):
            rows = [
                [
                    a - b if a is not None and b is not None else None
                    for a, b in zip(t[key], o[key], strict=True)
                ]
                for t, o in zip(tile["runs"], one["runs"], strict=True)
            ]
            paired[suffix][key] = summary_columns(rows)
    return paired


def run_experiment(model_dir, output_dir, figures_dir):
    from rrs.experiments.hallway import build_baselines
    from rrs.experiments.learning_report import reference_model
    from rrs.experiments.tile_coded_models import SEEDS as MODEL_SEEDS
    from rrs.experiments.tile_coded_planning_inputs import exact_models, load_inputs
    from rrs.experiments.tile_coded_planning_plots import plot_results
    from rrs.experiments.two_room import _revision
    from rrs.rl.exact import value_iteration

    data = build_baselines()
    inputs = load_inputs(model_dir, data)
    exact, reference_checks = exact_models(data, inputs)
    environment = reference_model(data.positions)
    optimum = value_iteration(environment).values
    orders = jax.vmap(lambda seed: sample_states(seed, 72, 5000))(jnp.array(SEEDS))
    revision, dirty = _revision()
    cases, cache = {}, {}
    for representation, features in inputs["features"].items():
        sources = {
            "exact": jax.tree.map(
                lambda x: jnp.broadcast_to(x, (len(SEEDS),) + x.shape),
                exact[representation],
            ),
            **{
                f"learned_{s:05d}": m
                for s, m in inputs["checkpoints"][representation].items()
            },
        }
        for source, models in sources.items():
            for configuration, indices in CONFIGURATIONS.items():
                name = f"{representation}/{source}/{configuration}"
                selected = jax.tree.map(lambda m: m[:, jnp.array(indices)], models)
                result = plan_batch(
                    selected, orders, features[jnp.array(data.nonterminal_indices)]
                )
                artifact = save_json(
                    output_dir / f"{name.replace('/', '_')}_weights.json.gz",
                    {
                        "planning_seeds": SEEDS,
                        "model_seeds": MODEL_SEEDS if source != "exact" else None,
                        "case": name,
                        "budgets": BUDGETS,
                        "axes": ["run", "evaluation_budget", "feature"],
                        "weights": explicit_nonfinite(
                            result.history[
                                :, jnp.array([CHECKPOINTS.index(b) for b in BUDGETS])
                            ].tolist()
                        ),
                    },
                    compressed=True,
                )
                case = report_case(
                    data,
                    selected,
                    result,
                    features,
                    inputs["policies"][jnp.array(indices)],
                    indices,
                    environment,
                    optimum,
                    cache=cache,
                )
                case.update(
                    {
                        "representation": representation,
                        "source": source,
                        "configuration": configuration,
                        "weights_artifact": artifact,
                    }
                )
                cases[name] = case
                save_json(output_dir / f"{name.replace('/', '_')}.json", case)
                print(
                    f"{name}: planned={case['summary']['planned_start']['mean'][-1]}, actual={case['summary']['actual_return']['mean'][-1]}, failures={case['evaluation_failures']}",
                    flush=True,
                )
    import platform
    from importlib.metadata import version

    report = {
        "milestone": "12-tile-coded-planning",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/23",
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "software": {
            "python": platform.python_version(),
            "packages": {
                p: version(p) for p in ("jax", "jaxlib", "numpy", "matplotlib")
            },
            "backend": jax.default_backend(),
        },
        "planning_seeds": SEEDS,
        "model_seeds": MODEL_SEEDS,
        "checkpoints": CHECKPOINTS,
        "evaluation_budgets": BUDGETS,
        "alpha": 1.0,
        "gamma": 0.99,
        "execution": "Reselect every primitive step, first-max ties, frozen option policies",
        "input_provenance": inputs["provenance"],
        "reference_checks": reference_checks,
        "optimal_values": optimum,
        "optimal_start_value": optimum[data.positions.index((3, 1))],
        "cases": cases,
        "paired_tile_minus_one_hot": paired_comparisons(cases),
        "performance_checks": performance_checks(
            cases, optimum[data.positions.index((3, 1))]
        ),
    }
    report["evaluation_checks"] = {
        "all_policy_evaluations_resolved": all(
            c["evaluation_failures"] == 0 for c in cases.values()
        ),
        "all_available_return_bounds_pass": all(
            b <= 1e-8
            for c in cases.values()
            for r in c["runs"]
            for b in r["evaluation_bounds"]
            if b is not None
        ),
        "terminal_zero": all(c["terminal_zero"] for c in cases.values()),
    }
    save_json(output_dir / "planning.json", report)
    plot_results(report, figures_dir)
    print(
        "Failed performance checks:",
        [k for k, v in report["performance_checks"].items() if not v],
        flush=True,
    )


def main():
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/tile_coded_models")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tile_coded_planning")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if not args.model_dir.exists() or not any(args.model_dir.iterdir()):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rrs.experiments.tile_coded_models",
                "--output-dir",
                str(args.model_dir),
                "--figures-dir",
                str(args.figures_dir),
            ],
            check=True,
        )
    try:
        run_experiment(args.model_dir, args.output_dir, args.figures_dir)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        save_json(args.output_dir / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
