"""Eight fixed-policy expectation models sharing four-room experience."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import four_room
from rrs.experiments.four_room_options import training_problem
from rrs.representations.tabular import one_hot
from rrs.rl.model_learning import (
    ModelLearningParameters,
    ModelLearningState,
    initial_model,
    model_update,
)

MODEL_NAMES = ("up", "down", "left", "right", "H1", "H2", "H3", "H4")
PARAMETERS = ModelLearningParameters()
SEEDS = tuple(range(8000, 8030))
STEPS = 200_000
CHECKPOINT = 1000
SAVE_STEPS = (0, 10_000, 20_000, 30_000, 40_000, STEPS)


class TrainingState(NamedTuple):
    environment: four_room.State
    learners: ModelLearningState
    key: jax.Array
    episodes: jax.Array


def initialize(seed, problem):
    key = jax.random.key(seed)
    learners = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (len(MODEL_NAMES),) + x.shape),
        initial_model(len(problem.observations)),
    )
    return TrainingState(
        four_room.reset(key, problem.params), learners, key, jnp.array(0)
    )


def experience_step(
    state, action, step_key, reset_key, problem, options, parameters=PARAMETERS
):
    next_state, observation, reward, done = four_room.step(
        step_key, state.environment, action, problem.params
    )
    x = one_hot(four_room.observe(state.environment), problem.observations)
    nx = one_hot(observation, problem.observations)
    pi = (options.probability_weights @ x)[:, action]
    beta = options.stopping_weights @ nx
    learners, errors = jax.vmap(
        model_update, in_axes=(0, None, None, None, 0, None, 0, None, None)
    )(state.learners, x, reward, nx, pi, 0.25, beta, done, parameters)
    reset_state = four_room.reset(reset_key, problem.params)
    environment = four_room.State(
        jnp.where(done, reset_state.position, next_state.position)
    )
    return TrainingState(
        environment, learners, state.key, state.episodes + done
    ), errors


def behavior_step(state, problem, options, parameters=PARAMETERS):
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, errors = experience_step(
        state, action, step_key, reset_key, problem, options, parameters
    )
    return updated._replace(key=next_key), errors


@partial(jax.jit, static_argnames=("steps",))
def train_block(state, problem, options, steps=CHECKPOINT, parameters=PARAMETERS):
    if steps < 1:
        raise ValueError("steps must be positive")

    def update(carry, _):
        result, _ = behavior_step(carry, problem, options, parameters)
        return result, None

    return jax.lax.scan(update, state, None, length=steps)[0]


def main():
    import argparse
    import json
    from pathlib import Path

    from rrs.experiments.four_room_model_audit import audit_models
    from rrs.experiments.four_room_model_inputs import ensure_snapshot, load_options
    from rrs.experiments.four_room_model_plots import plot_results
    from rrs.experiments.four_room_model_report import (
        build_data,
        build_references,
        build_report,
        save_checkpoint,
        save_options,
    )
    from rrs.experiments.model_report import model_rmse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/four_room_models")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--option-snapshot", type=Path)
    args = parser.parse_args()
    problem, data = training_problem(), build_data()
    options, source = load_options(
        ensure_snapshot(args.option_snapshot, args.figures_dir), data
    )
    frozen = save_options(
        args.output_dir / "frozen_options.json.gz", options, source, data
    )
    oracles, targets, identities = build_references(data, options, progress=True)
    audits = audit_models(data, options, oracles, progress=True)
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array(SEEDS))
    advance = jax.jit(jax.vmap(lambda state: train_block(state, problem, options)))
    history, artifacts, finite, mass_ok = [], [], True, True
    print(
        f"Training {len(SEEDS)} seeds, eight models each, for {STEPS:,} transitions",
        flush=True,
    )
    for step in range(0, STEPS + 1, CHECKPOINT):
        if step:
            state = advance(state)
        errors = model_rmse(state.learners.model, targets)
        history.append(errors)
        finite = finite and all(
            bool(jnp.isfinite(x).all())
            for x in jax.tree.leaves((state.learners, errors))
        )
        n = state.learners.model.successor_weights
        mass_ok = mass_ok and bool(
            (n >= -1e-7).all() & (n.sum(axis=-2) <= PARAMETERS.gamma + 1e-6).all()
        )
        if step in SAVE_STEPS:
            artifacts.append(
                save_checkpoint(
                    args.output_dir / f"models_step_{step:06d}.json.gz",
                    state.learners.model,
                    step,
                    SEEDS,
                    data,
                    frozen,
                )
            )
        if step % 10_000 == 0:
            print(f"Model training: {step:,}/{STEPS:,}", flush=True)
    report = build_report(
        data,
        source,
        frozen,
        oracles,
        identities,
        state,
        history,
        audits,
        artifacts,
        finite,
        mass_ok,
    )
    path = args.output_dir / "learning.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = plot_results(data, report, state.learners.model, targets, args.figures_dir)
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
