"""Paired one-hot and tile-coded hallway-option learning (Milestone 10)."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.experiments.option_learning import TrainingResult, TrainingState
from rrs.representations import tile_coding
from rrs.representations.tabular import one_hot
from rrs.rl.learning import LearningParameters, actor_critic_update, initial_weights

SEEDS = tuple(range(100))
STEPS = 50_000
CHECKPOINT = 500
PARAMETERS = LearningParameters()


class Problem(NamedTuple):
    environment: two_room.Params
    representation: object


def tabular_features(params, observation):
    return one_hot(observation, params)


def hallway_signal(observation):
    """Fixed target, independent of actor/critic representation; main value=0."""
    return jnp.all(observation == jnp.array([3, 7])).astype(jnp.float32)


def training_problem(name):
    environment = two_room.default_params()
    observations = tuple(
        p
        for p in two_room.legal_positions(environment)
        if p != tuple(environment.goal.tolist())
    )
    if name == "one_hot":
        return Problem(environment, jnp.asarray(observations)), tabular_features
    if name == "tile_coding":
        return Problem(
            environment, tile_coding.build_params(observations)
        ), tile_coding.encode
    raise ValueError(f"Unknown representation: {name}")


def initialize(seed, problem, encode):
    key = jax.random.key(seed)
    n_features = encode(problem.representation, problem.environment.start).shape[0]
    return TrainingState(
        two_room.reset(key, problem.environment),
        initial_weights(n_features, 4),
        key,
        jnp.array(0),
    )


def experience_step(
    state, action, step_key, reset_key, problem, encode, parameters=PARAMETERS
):
    next_state, observation, reward, done = two_room.step(
        step_key, state.environment, action, problem.environment
    )
    x = encode(problem.representation, two_room.observe(state.environment))
    nx = encode(problem.representation, observation)
    weights, info = actor_critic_update(
        state.weights,
        x,
        action,
        reward,
        nx,
        hallway_signal(observation),
        0.25,
        done,
        parameters,
    )
    reset = two_room.reset(reset_key, problem.environment)
    environment = two_room.State(jnp.where(done, reset.position, next_state.position))
    return TrainingState(environment, weights, state.key, state.episodes + done), info


def behavior_step(state, problem, encode, parameters=PARAMETERS):
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, info = experience_step(
        state, action, step_key, reset_key, problem, encode, parameters
    )
    return updated._replace(key=next_key), info


@partial(jax.jit, static_argnames=("encode", "steps", "checkpoint"))
def train(
    seed, problem, encode, steps=STEPS, checkpoint=CHECKPOINT, parameters=PARAMETERS
):
    if steps < 1 or checkpoint < 1 or steps % checkpoint:
        raise ValueError("steps must be positive and divisible by checkpoint")
    initial = initialize(seed, problem, encode)

    def transition(state, _):
        updated, _ = behavior_step(state, problem, encode, parameters)
        return updated, None

    def block(state, _):
        updated, _ = jax.lax.scan(transition, state, None, length=checkpoint)
        return updated, updated.weights.critic

    final, critics = jax.lax.scan(block, initial, None, length=steps // checkpoint)
    return TrainingResult(
        final, jnp.concatenate((initial.weights.critic[None], critics))
    )


def main():
    import argparse
    import json
    from pathlib import Path

    from rrs.experiments.hallway import build_baselines
    from rrs.experiments.tile_coded_option_plots import plot_results
    from rrs.experiments.tile_coded_option_report import (
        build_report,
        case_report,
        save_training,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tile_coded_options")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    data, cases, results, artifacts = build_baselines(), {}, {}, {}
    for name in ("one_hot", "tile_coding"):
        problem, encode = training_problem(name)
        print(
            f"Training {name}: {len(SEEDS)} seeds × {STEPS:,} transitions", flush=True
        )
        result = jax.vmap(lambda seed: train(seed, problem, encode))(jnp.array(SEEDS))
        result.critics.block_until_ready()
        results[name] = result
        # Preserve learned parameters before independent evaluation can fail.
        artifacts[name] = save_training(
            args.output_dir / f"{name}_weights.json.gz", result, SEEDS
        )
        try:
            cases[name] = case_report(
                data, name, problem, encode, result, SEEDS, progress=True
            )
        except (ValueError, RuntimeError) as error:
            (args.output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "representation": name,
                        "error": str(error),
                        "artifacts": artifacts,
                    },
                    indent=2,
                )
                + "\n"
            )
            raise
    report = build_report(data, cases, results, artifacts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "learning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = plot_results(data, report, args.figures_dir)
    print(
        json.dumps(
            {
                "report": str(path),
                "figures": [str(p) for p in paths],
                "acceptance": report["acceptance"],
                "paired_differences": {
                    metric: {k: v for k, v in stats.items() if k != "per_seed"}
                    for metric, stats in report["paired_differences"].items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if not all(report["acceptance"].values()):
        raise RuntimeError(
            "Milestone 10 criteria failed; report and parameters preserved"
        )


if __name__ == "__main__":
    main()
