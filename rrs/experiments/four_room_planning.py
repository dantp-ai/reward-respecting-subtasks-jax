"""Four-room planning with primitive models and four stochastic options."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from rrs.envs import four_room
from rrs.experiments.four_room_model_report import build_data, build_references
from rrs.experiments.four_room_model_report import save_options as save_frozen_options
from rrs.experiments.four_room_models import MODEL_NAMES as MODEL_NAMES
from rrs.experiments.four_room_planning_inputs import (
    MODEL_SEEDS,
    MODEL_STEPS,
    PlanningInputs,
    ensure_inputs,
    load_inputs,
)
from rrs.experiments.four_room_options import training_problem
from rrs.experiments.learning_report import mean_and_se
from rrs.experiments.model_learning import FixedOptions, option_tables
from rrs.experiments.planning import PlanningRun, run_plan, sample_states
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.plan_evaluation import evaluate_action_policy, induced_policy
from rrs.rl.stochastic_exact import value_iteration
from rrs.rl.stochastic_subtasks import solve_subtask

PLANNING_SEEDS = tuple(range(3000, 3030))
LOOKAHEAD_BUDGET = 1_600_000
CHECKPOINT = 1_600
CHECKPOINTS = tuple(range(0, LOOKAHEAD_BUDGET + 1, CHECKPOINT))
EVALUATION_BUDGETS = (0, 200_000, 400_000, 600_000, LOOKAHEAD_BUDGET)
MODEL_SET_NAMES = tuple(MODEL_NAMES)
HALLWAYS = four_room.HALLWAYS
GAMMA = 0.99


@dataclass(frozen=True)
class ReferenceSet:
    models: LinearExpectationModel
    option_policies: jax.Array
    provenance: dict


def _stack_models(models):
    return LinearExpectationModel(
        jnp.stack([model.reward_weights for model in models]),
        jnp.stack([model.successor_weights for model in models]),
    )


def _policies_from_options(options, data):
    policies, _ = option_tables(options, data)
    return policies


def _full_reference(data, options, label):
    oracles, targets, _ = build_references(data, options)
    return ReferenceSet(
        targets,
        _policies_from_options(options, data),
        {
            "kind": "exact_stochastic_expectation_models",
            "label": label,
            "iterations": [o.iterations for o in oracles],
            "bellman_residuals": [o.bellman_residual for o in oracles],
            "error_bounds": [o.error_bound for o in oracles],
        },
    )


def build_reference_sets(data, inputs):
    """Build exact primitive, optimal-option and frozen-policy models."""
    primitive_policy = jnp.broadcast_to(
        jnp.eye(4)[:, None, :], (4, len(data.positions), 4)
    )
    policy_weights = jnp.concatenate(
        (
            jnp.broadcast_to(
                jnp.eye(4)[:, :, None],
                (4, 4, len(data.nonterminal_indices)),
            ),
            inputs.option_policies[:, data.nonterminal_indices, :]
            .transpose(0, 2, 1),
        )
    )
    stopping_weights = jnp.concatenate(
        (
            jnp.ones((4, len(data.nonterminal_indices))),
            inputs.option_stopping[:, data.nonterminal_indices],
        )
    )
    frozen_options = FixedOptions(policy_weights, stopping_weights)
    frozen_oracles, frozen_targets, _ = build_references(data, frozen_options)
    primitive_models = LinearExpectationModel(
        frozen_targets.reward_weights[:4], frozen_targets.successor_weights[:4]
    )
    frozen_exact = ReferenceSet(
        frozen_targets,
        _policies_from_options(frozen_options, data),
        {
            "kind": "exact_stochastic_expectation_models",
            "label": "Milestone 7 seed 7000 frozen options",
            "iterations": [o.iterations for o in frozen_oracles],
            "bellman_residuals": [o.bellman_residual for o in frozen_oracles],
            "error_bounds": [o.error_bound for o in frozen_oracles],
        },
    )

    main = value_iteration(data.model, gamma=GAMMA)
    optimal_solutions = []
    for hallway in HALLWAYS:
        target_state = data.positions.index(hallway)
        z = tuple(float(s == target_state) for s in range(len(data.positions)))
        optimal_solutions.append(
            solve_subtask(data.model, z, data.terminal, gamma=GAMMA)
        )
    optimal_policies = jnp.stack(
        [
            jnp.array(
                [
                    float(a == (result.greedy_actions[s][0] if result.greedy_actions[s] else 0))
                    for s in range(len(data.positions))
                    for a in range(4)
                ]
            ).reshape((len(data.positions), 4))
            for result in optimal_solutions
        ]
    )
    optimal_stopping = jnp.array([r.stopping for r in optimal_solutions], dtype=jnp.float32)
    optimal_policy_weights = optimal_policies[:, data.nonterminal_indices, :].transpose(0, 2, 1)
    optimal_stop_weights = optimal_stopping[:, data.nonterminal_indices]
    optimal_options = FixedOptions(
        jnp.concatenate(
            (
                jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, len(data.nonterminal_indices))),
                optimal_policy_weights,
            )
        ),
        jnp.concatenate(
            (jnp.ones((4, len(data.nonterminal_indices))), optimal_stop_weights)
        ),
    )
    optimal_exact = _full_reference(data, optimal_options, "Equation 2 optimal hallway options; first greedy tie")

    return (
        primitive_models,
        primitive_policy,
        optimal_exact,
        frozen_exact,
        main,
    )


def build_cases(data, inputs, references):
    primitive_models, primitive_policy, optimal_exact, frozen_exact, _ = references
    n_runs = len(PLANNING_SEEDS)
    cases = []
    for source, models, policies in (
        ("exact_primitive", primitive_models, primitive_policy),
        ("exact_optimal_options", optimal_exact.models, optimal_exact.option_policies),
        ("exact_frozen_options", frozen_exact.models, frozen_exact.option_policies),
    ):
        cases.append(
            (
                source,
                "actions" if models.reward_weights.shape[0] == 4 else "actions_and_options",
                jax.tree.map(lambda x: jnp.broadcast_to(x, (n_runs,) + x.shape), models),
                jnp.broadcast_to(policies, (n_runs,) + policies.shape),
                None,
            )
        )
    for step, models in inputs.checkpoints.items():
        cases.extend(
            (
                (
                    f"learned_{step:06d}",
                    "actions",
                    LinearExpectationModel(models.reward_weights[:, :4], models.successor_weights[:, :4]),
                    jnp.broadcast_to(primitive_policy, (n_runs,) + primitive_policy.shape),
                    step,
                ),
                (
                    f"learned_{step:06d}",
                    "actions_and_options",
                    models,
                    jnp.concatenate(
                        (
                            jnp.broadcast_to(primitive_policy, (n_runs, 4, *primitive_policy.shape[1:])),
                            jnp.broadcast_to(inputs.option_policies, (n_runs,) + inputs.option_policies.shape),
                        ),
                        axis=1,
                    ),
                    step,
                ),
            )
        )
    return cases


def sample_state_orders(seeds=PLANNING_SEEDS, lookaheads=LOOKAHEAD_BUDGET):
    max_updates = lookaheads // 4
    return jax.vmap(lambda seed: sample_states(seed, 103, max_updates))(jnp.array(seeds))


@partial(jax.jit, static_argnames=("checkpoint_indices",))
def _evaluate_policies(weights, models, policies, features, checkpoint_indices):
    def evaluate(weights_at_checkpoint):
        return jax.vmap(induced_policy, in_axes=(0, 0, 0, None))(
            weights_at_checkpoint, models, policies, features
        )

    selected = weights[:, jnp.array(checkpoint_indices)].transpose((1, 0, 2))
    return jax.vmap(evaluate)(selected)


def plan_case(data, case, result, optimal_values, evaluation_environment):
    source, configuration, models, policies, model_step = case
    n_options = int(models.reward_weights.shape[1])
    lookaheads = LOOKAHEAD_BUDGET
    if lookaheads % CHECKPOINT or CHECKPOINT % n_options:
        raise ValueError("Frozen look-ahead grid must accommodate each complete model set")
    evaluation_indices = tuple(budget // CHECKPOINT for budget in EVALUATION_BUDGETS)
    all_policies, selections = _evaluate_policies(
        result.history,
        models,
        policies,
        jnp.array(data.features, dtype=jnp.float32),
        evaluation_indices,
    )
    start = data.positions.index((4, 1))
    start_feature = data.nonterminal_indices.index(start)
    exact_targets = jnp.array([optimal_values[i] for i in data.nonterminal_indices])
    planned_starts = result.history[:, :, start_feature]
    value_rmse = jnp.sqrt(jnp.mean((result.history - exact_targets) ** 2, axis=-1))
    actual = []
    selection_stats = []
    bounds = []
    reachable = []
    for budget_index in range(len(EVALUATION_BUDGETS)):
        run_returns, run_stats, run_bounds, run_reachable = [], [], [], []
        for seed_index in range(len(PLANNING_SEEDS)):
            policy = all_policies[budget_index, seed_index].tolist()
            selected = selections[budget_index, seed_index].tolist()
            evaluation = evaluate_action_policy(
                evaluation_environment,
                policy,
                start,
                data.terminal,
                gamma=GAMMA,
            )
            counts = [selected.count(i) for i in range(n_options)]
            run_returns.append(evaluation.value)
            run_stats.append(counts)
            run_bounds.append(evaluation.error_bound)
            run_reachable.append(evaluation.reachable_states)
        actual.append(run_returns)
        selection_stats.append(run_stats)
        bounds.append(run_bounds)
        reachable.append(run_reachable)
    optimum = float(optimal_values[start])
    start_curves = planned_starts.tolist()
    hit_steps = []
    for curve in start_curves:
        hit_steps.append(next((CHECKPOINTS[i] for i, value in enumerate(curve) if value >= 0.95 * optimum), None))
    rmse_curves = value_rmse.tolist()
    actual_by_run = [list(values) for values in zip(*actual, strict=True)]
    selection_by_run = [list(values) for values in zip(*selection_stats, strict=True)]
    runs = []
    planning_updates = result.updates.tolist()
    planning_queries = result.lookaheads.tolist()
    for i, seed in enumerate(PLANNING_SEEDS):
        runs.append(
            {
                "planning_seed": seed,
                "model_seed": MODEL_SEEDS[i] if model_step is not None else None,
                "planned_start": start_curves[i],
                "value_rmse": rmse_curves[i],
                "actual_return": actual_by_run[i],
                "selected_model_counts": selection_by_run[i],
                "first_95_percent_lookahead": hit_steps[i],
                "final_start_absolute_error": abs(start_curves[i][-1] - optimum),
                "final_max_value_error": max(
                    abs(float(result.weights[i, feature]) - float(optimal_values[state]))
                    for feature, state in enumerate(data.nonterminal_indices)
                ),
                "final_weights": result.weights[i].tolist(),
                "evaluation_error_bounds": [bounds[t][i] for t in range(len(bounds))],
                "reachable_states": [reachable[t][i] for t in range(len(reachable))],
            }
        )
    summaries = {}
    for metric, values in (
        ("planned_start", start_curves),
        ("value_rmse", rmse_curves),
        ("actual_return", actual_by_run),
    ):
        columns = [mean_and_se([run[t] for run in values]) for t in range(len(values[0]))]
        summaries[metric] = {key: [c[key] for c in columns] for key in ("mean", "se")}
    hits = [value for value in hit_steps if value is not None]
    hit_se = mean_and_se(hits) if len(hits) == len(hit_steps) else None
    terminal_features = jnp.asarray(data.features)[jnp.asarray(data.terminal)]
    return {
        "source": source,
        "configuration": configuration,
        "model_checkpoint": model_step,
        "models_per_backup": n_options,
        "planning_updates_per_run": planning_updates[0],
        "planning_lookaheads_per_run": planning_queries[0],
        "diagnostic_lookaheads_per_run": (
            len(EVALUATION_BUDGETS) * len(data.positions) * n_options
        ),
        "finite": bool(jnp.isfinite(result.history).all())
        and all(jnp.isfinite(jnp.asarray(v)).all() for v in (actual, start_curves, rmse_curves)),
        "terminal_zero": bool(jnp.all(result.history @ terminal_features.T == 0)),
        "first_95_percent": {
            "reached": len(hits),
            "total": len(PLANNING_SEEDS),
            "mean_lookaheads": hit_se["mean"] if hit_se else None,
            "se_lookaheads": hit_se["se"] if hit_se else None,
        },
        "summary": summaries,
        "runs": runs,
    }


def main():
    import argparse
    import json
    import subprocess
    import sys

    from rrs.experiments.four_room_planning_plots import plot_results
    from rrs.experiments.four_room_planning_report import build_report, save_trace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/four_room_planning"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data = build_data()
    model_dir = ensure_inputs(args.model_dir, args.figures_dir)
    inputs = load_inputs(model_dir, data.positions, data.nonterminal_indices)
    refs = build_reference_sets(data, inputs)
    environment = data.model
    optimum = value_iteration(environment, gamma=GAMMA).values
    state_orders = sample_state_orders()
    cases, traces = {}, {}
    for case in build_cases(data, inputs, refs):
        name = f"{case[0]}/{case[1]}"
        print(f"Planning {name}", flush=True)
        # case_report also performs the independent execution-policy audit.
        result = jax.vmap(
            lambda models, order: run_plan(
                models,
                order,
                lookahead_budget=LOOKAHEAD_BUDGET,
                checkpoint=CHECKPOINT,
            )
        )(case[2], state_orders)
        cases[name] = plan_case(data, case, result, optimum, environment)
        traces[name] = result.history
        print(
            f"{name}: planned={cases[name]['summary']['planned_start']['mean'][-1]:.6f}, "
            f"actual={cases[name]['summary']['actual_return']['mean'][-1]:.6f}",
            flush=True,
        )
    provenance = {"milestone_8": inputs.provenance}
    for name, history in traces.items():
        safe_name = name.replace("/", "_")
        provenance[name] = save_trace(
            args.output_dir / f"{safe_name}_weights.json.gz", history,
            name, CHECKPOINTS, PLANNING_SEEDS, inputs.provenance,
        )
    report = build_report(data, inputs, cases, refs, optimum, provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "planning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    figures = plot_results(report, args.figures_dir)
    print(json.dumps({"report": str(path), "figures": [str(p) for p in figures],
                      "acceptance": report["acceptance"]}, indent=2), flush=True)
    if not all(report["acceptance"].values()):
        raise RuntimeError("Frozen planning criteria failed; results preserved for diagnosis")


if __name__ == "__main__":
    main()
