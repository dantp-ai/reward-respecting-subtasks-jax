"""Frozen-option model learning: uv run python -m rrs.experiments.model_learning."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.experiments.learning_report import policy_tables
from rrs.experiments.option_learning import TrainingProblem
from rrs.representations.tabular import one_hot
from rrs.rl.model_learning import (
    ModelLearningParameters,
    ModelLearningState,
    initial_model,
    model_update,
)

MODEL_NAMES = ("up", "down", "left", "right", "reward_respecting", "shortest_path")


class FixedOptions(NamedTuple):
    # Coefficients over one-hot features, not softmax preference weights.
    probability_weights: jax.Array  # (options, actions, input features)
    stopping_weights: jax.Array  # (options, input features)


class ModelTrainingState(NamedTuple):
    environment: two_room.State
    learners: ModelLearningState
    key: jax.Array
    episodes: jax.Array


def freeze_options(data, learned_option) -> FixedOptions:
    learned_pi, learned_beta = policy_tables(
        learned_option.critic, learned_option.actor, data
    )
    probabilities, stopping = [], []
    for name in MODEL_NAMES:
        if name == "reward_respecting":
            pi, beta = jnp.array(learned_pi), jnp.array(learned_beta)
        else:
            option = data.options[name]
            pi, beta = (
                jax.nn.one_hot(jnp.array(option.policy), 4),
                jnp.array(option.stopping),
            )
        indices = jnp.array(data.nonterminal_indices)
        probabilities.append(pi[indices].T)
        stopping.append(beta[indices].astype(jnp.float32))
    return FixedOptions(jnp.stack(probabilities), jnp.stack(stopping))


def option_tables(options: FixedOptions, data):
    """Export full oracle rows, including harmless uniform terminal policies."""
    features, terminal = jnp.array(data.features), jnp.array(data.terminal)
    pi = jnp.einsum("oaf,sf->osa", options.probability_weights, features)
    beta = jnp.einsum("of,sf->os", options.stopping_weights, features)
    return jnp.where(terminal[None, :, None], 0.25, pi), jnp.where(
        terminal[None, :], 1.0, beta
    )


def initialize(
    seed, problem: TrainingProblem, n_options: int = 6
) -> ModelTrainingState:
    key = jax.random.key(seed)
    learners = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_options,) + x.shape),
        initial_model(len(problem.observations)),
    )
    return ModelTrainingState(
        two_room.reset(key, problem.params), learners, key, jnp.array(0)
    )


def experience_step(
    state: ModelTrainingState,
    action,
    step_key,
    reset_key,
    problem: TrainingProblem,
    options: FixedOptions,
    parameters: ModelLearningParameters = ModelLearningParameters(),
):
    next_state, observation, reward, done = two_room.step(
        step_key, state.environment, action, problem.params
    )
    x = one_hot(two_room.observe(state.environment), problem.observations)
    nx = one_hot(observation, problem.observations)
    target_probabilities = (options.probability_weights @ x)[:, action]
    beta = options.stopping_weights @ nx
    learners, errors = jax.vmap(
        model_update, in_axes=(0, None, None, None, 0, None, 0, None, None)
    )(state.learners, x, reward, nx, target_probabilities, 0.25, beta, done, parameters)
    reset_state = two_room.reset(reset_key, problem.params)
    environment = two_room.State(
        jnp.where(done, reset_state.position, next_state.position)
    )
    return ModelTrainingState(
        environment, learners, state.key, state.episodes + done
    ), errors


def behavior_step(
    state: ModelTrainingState,
    problem: TrainingProblem,
    options: FixedOptions,
    parameters: ModelLearningParameters = ModelLearningParameters(),
):
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, errors = experience_step(
        state, action, step_key, reset_key, problem, options, parameters
    )
    return updated._replace(key=next_key), errors


@partial(jax.jit, static_argnames=("steps",))
def train_block(
    state: ModelTrainingState,
    problem: TrainingProblem,
    options: FixedOptions,
    steps: int = 500,
    parameters: ModelLearningParameters = ModelLearningParameters(),
) -> ModelTrainingState:
    """Advance tested behavior steps, without oracle tables or metric computation."""
    if steps < 1:
        raise ValueError("steps must be positive")

    def transition(state, _):
        updated, _ = behavior_step(state, problem, options, parameters)
        return updated, None

    state, _ = jax.lax.scan(transition, state, None, length=steps)
    return state


def main():
    import argparse
    import json
    from pathlib import Path

    from rrs.experiments.hallway import build_baselines
    from rrs.experiments.model_audit import audit_hallway_model
    from rrs.experiments.model_report import (
        build_references,
        build_report,
        model_rmse,
        plot_results,
        save_checkpoint,
    )
    from rrs.experiments.option_learning import train as train_option, training_problem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/model_learning")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    problem, data = training_problem(), build_baselines()
    print("Regenerating and freezing the Milestone 3 option (seed 0)", flush=True)
    frozen = train_option(0, problem).final.weights
    options = freeze_options(data, frozen)
    oracles, targets = build_references(data, options)
    pi, beta = option_tables(options, data)
    print("Auditing the frozen model with 40,000 independent rollouts", flush=True)
    audits = audit_hallway_model(
        data, pi[4].tolist(), beta[4].tolist(), oracles[4].model
    )
    seeds, steps, checkpoint = tuple(range(1000, 1100)), 50_000, 500
    parameters = ModelLearningParameters()
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array(seeds))
    advance = jax.jit(
        jax.vmap(
            lambda state: train_block(state, problem, options, checkpoint, parameters)
        )
    )
    history, artifacts, finite = [], [], True
    print(
        f"Training {len(seeds)} seeds for {steps:,} model-learning transitions each",
        flush=True,
    )
    for step in range(0, steps + 1, checkpoint):
        if step:
            state = advance(state)
        errors = model_rmse(state.learners.model, targets)
        history.append(errors)
        finite = finite and all(
            bool(jnp.isfinite(x).all())
            for x in jax.tree.leaves((state.learners, errors))
        )
        if step in (0, 10_000, 20_000, 50_000):
            path = args.output_dir / f"models_step_{step:05d}.json.gz"
            artifacts.append(
                save_checkpoint(path, state.learners.model, step, seeds, data)
            )
        if step % 5000 == 0:
            print(
                f"Model training: {step:,}/{steps:,} transitions per seed", flush=True
            )
    report = build_report(
        data,
        frozen,
        options,
        oracles,
        targets,
        state,
        history,
        seeds,
        steps,
        checkpoint,
        parameters,
        audits,
        artifacts,
        finite,
    )
    figures = plot_results(
        data, report, state.learners.model, targets, args.figures_dir
    )
    report["figures"] = [str(path) for path in figures]
    report_path = args.output_dir / "learning.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "final_mean_errors": {
                    name: {key: value["mean"][-1] for key, value in metrics.items()}
                    for name, metrics in report["summary"].items()
                },
                "all_acceptance_passed": all(report["acceptance"].values()),
            },
            indent=2,
        )
    )
    print(f"Report: {report_path}")
    for path in figures:
        print(f"Figure: {path}")
    if not all(report["acceptance"].values()):
        raise SystemExit(
            "Frozen acceptance criteria failed; results preserved for diagnosis"
        )


if __name__ == "__main__":
    main()
