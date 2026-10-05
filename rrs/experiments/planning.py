"""Seeded planning comparisons: uv run python -m rrs.experiments.planning."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.experiments.model_learning import MODEL_NAMES, option_tables
from rrs.experiments.model_report import build_references
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.planning import planning_step

CONFIGURATIONS = {
    "primitives": (0, 1, 2, 3),
    "shortest_path": (0, 1, 2, 3, 5),
    "reward_respecting": (0, 1, 2, 3, 4),
}
EVALUATION_BUDGETS = (0, 2000, 5000, 10_000, 20_000)


@partial(jax.jit, static_argnames=("count",))
def sample_states(seed, n_features, count):
    """Uniform feature indices with a shape-independent PRNG prefix."""

    def sample(key, _):
        key, state_key = jax.random.split(key)
        return key, jax.random.randint(state_key, (), 0, n_features)

    _, states = jax.lax.scan(sample, jax.random.key(seed), None, length=count)
    return states


class PlanningRun(NamedTuple):
    weights: jax.Array
    history: jax.Array
    updates: int
    lookaheads: int


@partial(jax.jit, static_argnames=("lookahead_budget", "checkpoint"))
def run_plan(models, state_order, lookahead_budget=20_000, checkpoint=100, alpha=1.0):
    """Use the supplied state prefix, spending one query per available model."""
    n_options, n_features = models.reward_weights.shape
    if (
        lookahead_budget < 0
        or checkpoint < 1
        or lookahead_budget % checkpoint
        or checkpoint % n_options
    ):
        raise ValueError(
            "Budget and checkpoint must accommodate complete model backups"
        )
    updates = lookahead_budget // n_options
    if state_order.shape[0] < updates:
        raise ValueError("State order is shorter than the requested planning budget")
    blocks = state_order[:updates].reshape(
        (lookahead_budget // checkpoint, checkpoint // n_options)
    )
    initial = jnp.zeros(n_features)

    def update(weights, state):
        features = jax.nn.one_hot(state, n_features)
        updated, _ = planning_step(weights, features, models, alpha)
        return updated, None

    def block(weights, states):
        weights, _ = jax.lax.scan(update, weights, states)
        return weights, weights

    weights, history = jax.lax.scan(block, initial, blocks)
    return PlanningRun(
        weights, jnp.concatenate((initial[None], history)), updates, updates * n_options
    )


@dataclass(frozen=True)
class PlanningCase:
    source: str
    configuration: str
    models: LinearExpectationModel
    option_policies: jax.Array

    @property
    def name(self):
        return f"{self.source}/{self.configuration}"


def build_cases(data, inputs):
    """Keep option quality separate from model approximation in comparisons."""
    n_runs = len(inputs.provenance["model_seeds"])
    optimal = [
        data.models[name].as_linear_model(data.nonterminal_indices)
        for name in MODEL_NAMES
    ]
    optimal = LinearExpectationModel(
        jnp.stack([m.reward_weights for m in optimal]),
        jnp.stack([m.successor_weights for m in optimal]),
    )
    _, frozen_exact = build_references(data, inputs.options)
    optimal_pi = jax.nn.one_hot(
        jnp.array([data.options[name].policy for name in MODEL_NAMES]), 4
    )
    frozen_pi, _ = option_tables(inputs.options, data)
    sources = {
        "exact_optimal": jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_runs,) + x.shape), optimal
        ),
        "exact_frozen": jax.tree.map(
            lambda x: jnp.broadcast_to(x, (n_runs,) + x.shape), frozen_exact
        ),
        **{
            f"learned_{step:05d}": models for step, models in inputs.checkpoints.items()
        },
    }
    cases = []
    for source, models in sources.items():
        for name, indices in CONFIGURATIONS.items():
            if source == "exact_frozen" and name != "reward_respecting":
                continue  # Primitive/shortest-path exact models already appear above.
            indices = jnp.array(indices)
            subset = jax.tree.map(lambda x: x[:, indices], models)
            pi = optimal_pi if source == "exact_optimal" else frozen_pi
            cases.append(PlanningCase(source, name, subset, pi[indices]))
    return cases


def main():
    import argparse
    import json
    import subprocess
    import sys
    from pathlib import Path

    from rrs.experiments.hallway import build_baselines
    from rrs.experiments.learning_report import reference_model
    from rrs.experiments.planning_inputs import load_inputs
    from rrs.experiments.planning_report import build_report, case_report, plot_results
    from rrs.rl.exact import value_iteration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/model_learning")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/planning"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if not (args.model_dir / "learning.json").exists():
        print(
            "Milestone 4 report absent; regenerating its canonical experiment",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rrs.experiments.model_learning",
                "--output-dir",
                str(args.model_dir),
                "--figures-dir",
                str(args.figures_dir),
            ],
            check=True,
        )
    data = build_baselines()
    inputs = load_inputs(
        args.model_dir, [data.positions[i] for i in data.nonterminal_indices]
    )
    environment = reference_model(data.positions)
    optimum = value_iteration(environment).values
    seeds = list(range(2000, 2100))
    checkpoints = list(range(0, 20_001, 100))
    state_orders = jax.vmap(
        lambda seed: sample_states(seed, len(data.nonterminal_indices), 5000)
    )(jnp.array(seeds))
    cases = {}
    for case in build_cases(data, inputs):
        result = jax.vmap(run_plan)(case.models, state_orders)
        cases[case.name] = case_report(
            data,
            case,
            result,
            seeds,
            inputs.provenance["model_seeds"],
            environment,
            optimum,
            checkpoints,
            EVALUATION_BUDGETS,
        )
        summary = cases[case.name]["summary"]
        print(
            f"{case.name}: planned={summary['planned_start']['mean'][-1]:.6f}, "
            f"actual={summary['actual_return']['mean'][-1]:.6f}",
            flush=True,
        )
    report = build_report(
        data, inputs, cases, seeds, checkpoints, EVALUATION_BUDGETS, optimum
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "planning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    figures = plot_results(report, args.figures_dir)
    print(
        json.dumps(
            {
                "report": str(path),
                "figures": [str(p) for p in figures],
                "acceptance": report["acceptance"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not all(report["acceptance"].values()):
        raise RuntimeError(
            "Frozen acceptance criteria failed; results preserved for diagnosis"
        )


if __name__ == "__main__":
    main()
