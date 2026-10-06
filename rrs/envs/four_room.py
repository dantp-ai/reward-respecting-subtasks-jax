"""Stochastic four-room grid from paper Section 7 / Figure 6.

Coordinates, reward timing and probability semantics: notes/four_room_contract.md.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

LAYOUT = (
    "#############",
    "#.....#.....#",
    "#.....#.....#",
    "#.....2.....#",
    "#S....#.....#",
    "#.....#.....#",
    "##1####.....#",
    "#.PPP.###3###",
    "#.PPP.#.....#",
    "#.PPP.#G....#",
    "#.PPP.4.....#",
    "#.....#.....#",
    "#############",
)
HALLWAYS = ((6, 2), (3, 6), (7, 9), (10, 6))


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Params(NamedTuple):
    walls: jax.Array
    penalties: jax.Array
    start: jax.Array
    goal: jax.Array
    motion_probabilities: jax.Array


class State(NamedTuple):
    position: jax.Array


class Outcomes(NamedTuple):
    """Branches ordered by absolute realized direction, including blocked moves."""

    probabilities: jax.Array
    positions: jax.Array
    rewards: jax.Array
    terminated: jax.Array


def default_params() -> Params:
    probabilities = jnp.full((4, 4), 1 / 9).at[jnp.arange(4), jnp.arange(4)].set(2 / 3)
    return Params(
        walls=jnp.array([[cell == "#" for cell in row] for row in LAYOUT]),
        penalties=jnp.array([[cell == "P" for cell in row] for row in LAYOUT]),
        start=jnp.array([4, 1], dtype=jnp.int32),
        goal=jnp.array([9, 7], dtype=jnp.int32),
        motion_probabilities=probabilities,
    )


def reset(key: jax.Array, params: Params) -> State:
    del key
    return State(params.start)


def observe(state: State) -> jax.Array:
    return state.position


def outcomes(state: State, action: jax.Array | int, params: Params) -> Outcomes:
    """Enumerate all movements before sampling; no renormalization at walls.

    Preconditions: legal state, action in [0,4), row-stochastic motion table.
    The four entries may contain duplicate positions/outcomes.
    """
    moves = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
    candidates = state.position + moves
    bounds = jnp.array(params.walls.shape, dtype=jnp.int32)
    inside = jnp.all((candidates >= 0) & (candidates < bounds), axis=1)
    safe = jnp.clip(candidates, 0, bounds - 1)
    blocked = (~inside) | params.walls[safe[:, 0], safe[:, 1]]
    already_terminal = jnp.all(state.position == params.goal)
    positions = jnp.where(
        (blocked | already_terminal)[:, None], state.position, candidates
    )
    terminated = jnp.all(positions == params.goal, axis=1)
    rewards = jnp.where(
        already_terminal,
        0.0,
        jnp.where(
            terminated,
            1.0,
            jnp.where(params.penalties[positions[:, 0], positions[:, 1]], -1.0, 0.0),
        ),
    )
    return Outcomes(params.motion_probabilities[action], positions, rewards, terminated)


def step(key, state: State, action, params: Params):
    """Sample one realized direction; emit arrival reward and never reset."""
    branches = outcomes(state, action, params)
    direction = jax.random.categorical(key, jnp.log(branches.probabilities))
    next_state = State(branches.positions[direction])
    return (
        next_state,
        observe(next_state),
        branches.rewards[direction],
        branches.terminated[direction],
    )


def legal_positions(params: Params) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r, c)
        for r, row in enumerate(params.walls.tolist())
        for c, wall in enumerate(row)
        if not wall
    )
