"""Milestone 11: learn models of identical frozen options in two feature spaces."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.envs import two_room
from rrs.experiments.model_learning import ModelTrainingState
from rrs.experiments.tile_coded_options import hallway_signal, train, training_problem
from rrs.rl.learning import stopping_rule
from rrs.rl.model_learning import ModelLearningParameters, initial_model, model_update

NAMES = (
    "up",
    "down",
    "left",
    "right",
    "shortest_path",
    "one_hot_option",
    "tile_option",
)
SEEDS = tuple(range(2000, 2100))
STEPS, CHECKPOINT = 50_000, 500
PARAMETERS = ModelLearningParameters()


class Problem(NamedTuple):
    environment: two_room.Params
    positions: jax.Array
    features: jax.Array  # state, feature; lookup belongs to experiment only
    probabilities: jax.Array  # option, state, action
    stopping: jax.Array  # option, state


def features(data, name):
    problem, encode = training_problem(name)
    return jax.vmap(lambda obs: encode(problem.representation, obs))(
        jnp.array(data.positions)
    )


def freeze_options(data, seed=100, steps=200_000):
    """Fresh, predetermined option pair; no selection by resulting quality."""
    policies, stops, learned = [], [], {}
    for name in NAMES[:5]:
        option = data.options[name]
        policies.append(jax.nn.one_hot(jnp.array(option.policy), 4))
        stops.append(jnp.array(option.stopping))
    for name in ("one_hot", "tile_coding"):
        problem, encode = training_problem(name)
        result = train(seed, problem, encode, steps=steps, checkpoint=steps)
        weights = result.final.weights
        x = features(data, name)
        pi = jax.nn.softmax(x @ weights.actor.T, axis=-1)
        beta = stopping_rule(
            jax.vmap(hallway_signal)(jnp.array(data.positions)),
            x @ weights.critic,
            jnp.array(data.terminal),
        )
        policies.append(pi)
        stops.append(beta)
        learned[name] = {
            "seed": seed,
            "steps": steps,
            "critic_weights": weights.critic.tolist(),
            "actor_weights": weights.actor.tolist(),
            "episodes": int(result.final.episodes),
        }
    terminal = jnp.array(data.terminal)
    return (
        jnp.where(terminal[None, :, None], 0.25, jnp.stack(policies)),
        jnp.where(terminal[None, :], 1.0, jnp.stack(stops)),
        learned,
    )


def initialize(seed, problem):
    key = jax.random.key(seed)
    learners = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (problem.probabilities.shape[0],) + x.shape),
        initial_model(problem.features.shape[1]),
    )
    return ModelTrainingState(
        two_room.reset(key, problem.environment), learners, key, jnp.array(0)
    )


def experience_step(state, action, step_key, reset_key, problem, parameters=PARAMETERS):
    destination, obs, reward, done = two_room.step(
        step_key, state.environment, action, problem.environment
    )
    s = jnp.argmax(jnp.all(problem.positions == state.environment.position, axis=1))
    ns = jnp.argmax(jnp.all(problem.positions == obs, axis=1))
    learners, errors = jax.vmap(
        model_update, in_axes=(0, None, None, None, 0, None, 0, None, None)
    )(
        state.learners,
        problem.features[s],
        reward,
        problem.features[ns],
        problem.probabilities[:, s, action],
        0.25,
        problem.stopping[:, ns],
        done,
        parameters,
    )
    reset = two_room.reset(reset_key, problem.environment)
    return ModelTrainingState(
        two_room.State(jnp.where(done, reset.position, destination.position)),
        learners,
        state.key,
        state.episodes + done,
    ), errors


def behavior_step(state, problem, parameters=PARAMETERS):
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    updated, errors = experience_step(
        state, action, step_key, reset_key, problem, parameters
    )
    return updated._replace(key=next_key), errors


@partial(jax.jit, static_argnames=("steps",))
def train_block(state, problem, steps=CHECKPOINT, parameters=PARAMETERS):
    if steps < 1:
        raise ValueError("steps must be positive")

    def transition(state, _):
        return behavior_step(state, problem, parameters)[0], None

    return jax.lax.scan(transition, state, None, length=steps)[0]


def main():
    from rrs.experiments.tile_coded_model_report import run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
