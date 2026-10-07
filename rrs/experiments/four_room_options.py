"""Four hallway learners sharing stochastic experience, Milestone 7."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import four_room
from rrs.representations.tabular import one_hot
from rrs.rl.learning import (
    ActorCritic,
    LearningParameters,
    actor_critic_update,
    initial_weights,
)
from rrs.rl.subtasks import stopping_value

PARAMETERS = LearningParameters(alpha=0.05, alpha_actor=0.05)
SEEDS = tuple(range(7000, 7030))
STEPS = 1_000_000
CHECKPOINT = 2000
EVALUATION_STEPS = (0, 200_000, 1_000_000)
OPTION_NAMES = ("H1", "H2", "H3", "H4")


class TrainingProblem(NamedTuple):
    params: four_room.Params
    observations: jax.Array
    main_weights: jax.Array
    hallway_features: jax.Array


class TrainingState(NamedTuple):
    environment: four_room.State
    weights: ActorCritic
    key: jax.Array
    episodes: jax.Array


def training_problem():
    params = four_room.default_params()
    observations = [
        p for p in four_room.legal_positions(params) if p != tuple(params.goal.tolist())
    ]
    return TrainingProblem(
        params,
        jnp.array(observations),
        jnp.zeros(len(observations)),
        jnp.array([observations.index(p) for p in four_room.HALLWAYS]),
    )


def initialize(seed, problem):
    key = jax.random.key(seed)
    weights = jax.vmap(lambda _: initial_weights(len(problem.observations), 4))(
        problem.hallway_features
    )
    return TrainingState(
        four_room.reset(key, problem.params), weights, key, jnp.array(0)
    )


def experience_step(state, action, step_key, reset_key, problem, parameters=PARAMETERS):
    """One environment transition, four independent Equation 10 updates."""
    next_state, observation, reward, done = four_room.step(
        step_key, state.environment, action, problem.params
    )
    x = one_hot(four_room.observe(state.environment), problem.observations)
    nx = one_hot(observation, problem.observations)
    z = jax.vmap(lambda feature: stopping_value(problem.main_weights, nx, feature))(
        problem.hallway_features
    )
    weights, info = jax.vmap(
        lambda w, stop: actor_critic_update(
            w, x, action, reward, nx, stop, 0.25, done, parameters
        )
    )(state.weights, z)
    reset_state = four_room.reset(reset_key, problem.params)
    environment = four_room.State(
        jnp.where(done, reset_state.position, next_state.position)
    )
    return TrainingState(environment, weights, state.key, state.episodes + done), info


def behavior_step(state, problem, parameters=PARAMETERS):
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, info = experience_step(
        state, action, step_key, reset_key, problem, parameters
    )
    return updated._replace(key=next_key), info


@partial(jax.jit, static_argnames=("steps",))
def train_block(state, problem, steps=CHECKPOINT, parameters=PARAMETERS):
    if steps < 1:
        raise ValueError("steps must be positive")

    def update(carry, _):
        result, _ = behavior_step(carry, problem, parameters)
        return result, None

    result, _ = jax.lax.scan(update, state, None, length=steps)
    return result


def main():
    import argparse
    import json
    from pathlib import Path

    from rrs.experiments.four_room_option_plots import plot_results
    from rrs.experiments.four_room_option_report import (
        build_references,
        build_report,
        save_snapshot,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/four_room_options")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    problem, data = training_problem(), build_references()
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array(SEEDS))
    advance = jax.jit(jax.vmap(lambda state: train_block(state, problem)))
    history, snapshots, artifacts = [state.weights.critic], {0: state}, []
    print(
        f"Training {len(SEEDS)} seeds, four options each, for {STEPS:,} transitions",
        flush=True,
    )
    for step in range(CHECKPOINT, STEPS + 1, CHECKPOINT):
        state = advance(state)
        history.append(state.weights.critic)
        if step in EVALUATION_STEPS:
            snapshots[step] = state
            artifacts.append(
                save_snapshot(
                    args.output_dir / f"options_step_{step:07d}.json.gz",
                    state,
                    step,
                    SEEDS,
                    data,
                )
            )
        if step % 100_000 == 0:
            state.weights.critic.block_until_ready()
            print(f"Trained {step:,}/{STEPS:,} transitions", flush=True)
    report = build_report(
        data, jnp.stack(history, axis=1), snapshots, SEEDS, artifacts, progress=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "learning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = plot_results(data, report, snapshots, args.figures_dir)
    print(
        json.dumps(
            {
                "report": str(path),
                "figures": [str(p) for p in paths],
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
