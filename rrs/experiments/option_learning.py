"""Uniform-behavior hallway option learning for Milestone 3."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.representations.tabular import one_hot
from rrs.rl.learning import (
    ActorCritic,
    LearningParameters,
    actor_critic_update,
    initial_weights,
)
from rrs.rl.subtasks import stopping_value


class TrainingProblem(NamedTuple):
    params: two_room.Params
    observations: jax.Array
    main_weights: jax.Array
    hallway_feature: int


class TrainingState(NamedTuple):
    environment: two_room.State
    weights: ActorCritic
    key: jax.Array
    episodes: jax.Array


class TrainingResult(NamedTuple):
    final: TrainingState
    critics: jax.Array


def training_problem() -> TrainingProblem:
    """Construct features and frozen subtask inputs without an optimal oracle."""
    params = two_room.default_params()
    positions = [
        p for p in two_room.legal_positions(params) if p != tuple(params.goal.tolist())
    ]
    return TrainingProblem(
        params, jnp.array(positions), jnp.zeros(len(positions)), positions.index((3, 7))
    )


def initialize(seed, problem: TrainingProblem) -> TrainingState:
    key = jax.random.key(seed)
    return TrainingState(
        two_room.reset(key, problem.params),
        initial_weights(len(problem.observations), 4),
        key,
        jnp.array(0),
    )


def experience_step(
    state: TrainingState,
    action,
    step_key,
    reset_key,
    problem: TrainingProblem,
    parameters: LearningParameters = LearningParameters(),
):
    """Learn from a single action; reset only on environment termination."""
    next_state, observation, reward, done = two_room.step(
        step_key, state.environment, action, problem.params
    )
    features = one_hot(two_room.observe(state.environment), problem.observations)
    next_features = one_hot(observation, problem.observations)
    z = stopping_value(problem.main_weights, next_features, problem.hallway_feature)
    weights, info = actor_critic_update(
        state.weights,
        features,
        action,
        reward,
        next_features,
        z,
        0.25,
        done,
        parameters,
    )
    reset_state = two_room.reset(reset_key, problem.params)
    environment = two_room.State(
        jnp.where(done, reset_state.position, next_state.position)
    )
    return TrainingState(environment, weights, state.key, state.episodes + done), info


def behavior_step(
    state: TrainingState,
    problem: TrainingProblem,
    parameters: LearningParameters = LearningParameters(),
):
    """One uniform random action; option stopping never resets the behavior."""
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, info = experience_step(
        state, action, step_key, reset_key, problem, parameters
    )
    return updated._replace(key=next_key), info


@partial(jax.jit, static_argnames=("steps", "checkpoint"))
def train(
    seed,
    problem: TrainingProblem,
    steps: int = 50_000,
    checkpoint: int = 500,
    parameters: LearningParameters = LearningParameters(),
) -> TrainingResult:
    """Scan the tested one-transition update, retaining checkpoint critic values.

    The result includes zero initialization. This function can be vmapped over
    seeds; each seed owns its environment, weights and explicit PRNG stream.
    """
    if steps < 1 or checkpoint < 1 or steps % checkpoint:
        raise ValueError("steps must be positive and divisible by checkpoint")
    initial = initialize(seed, problem)

    def transition(state, _):
        updated, _ = behavior_step(state, problem, parameters)
        return updated, None

    def block(state, _):
        updated, _ = jax.lax.scan(transition, state, None, length=checkpoint)
        return updated, updated.weights.critic

    final, critics = jax.lax.scan(block, initial, None, length=steps // checkpoint)
    critics = jnp.concatenate((initial.weights.critic[None], critics))
    return TrainingResult(final, critics)


def main():
    import argparse
    import json
    from pathlib import Path

    from rrs.experiments.hallway import build_baselines
    from rrs.experiments.learning_report import build_report, plot_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/option_learning")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    seeds, steps, checkpoint = tuple(range(100)), 50_000, 500
    parameters, problem = LearningParameters(), training_problem()
    print(f"Training {len(seeds)} seeds for {steps:,} transitions each", flush=True)
    result = jax.vmap(lambda seed: train(seed, problem, steps, checkpoint, parameters))(
        jnp.array(seeds)
    )
    result.critics.block_until_ready()
    print("Training complete; evaluating stochastic policies", flush=True)
    data = build_baselines()
    report = build_report(
        data, result, seeds, steps, checkpoint, parameters, progress=True
    )
    figures = plot_results(data, report, args.figures_dir)
    report["figures"] = [str(path) for path in figures]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "learning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "summary": {
                    key: {
                        k: v[-1] if isinstance(v, list) else v for k, v in value.items()
                    }
                    for key, value in report["summary"].items()
                },
                "acceptance": report["acceptance"],
            },
            indent=2,
        )
    )
    print(f"Report: {path}")
    for figure in figures:
        print(f"Figure: {figure}")
    if not all(report["acceptance"].values()):
        raise SystemExit(
            "Frozen acceptance criteria failed; results preserved for diagnosis"
        )


if __name__ == "__main__":
    main()
