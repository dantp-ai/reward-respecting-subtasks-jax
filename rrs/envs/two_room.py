"""Deterministic two-room grid from paper Figures 1–2.

See notes/two_room_contract.md for coordinates, timing, and API preconditions.
"""

from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

LAYOUT = (
    "###############",
    "#.PPPP.#......#",
    "#.PPPP.#......#",
    "#SPPPP.H......#",
    "#.PPPP.#......#",
    "#.PPPP.#......#",
    "#......#..G...#",
    "###############",
)


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


class State(NamedTuple):
    position: jax.Array


def default_params() -> Params:
    """Construct the paper map as dynamic JAX pytree leaves."""
    return Params(
        walls=jnp.array([[cell == "#" for cell in row] for row in LAYOUT]),
        penalties=jnp.array([[cell == "P" for cell in row] for row in LAYOUT]),
        start=jnp.array([3, 1], dtype=jnp.int32),
        goal=jnp.array([6, 10], dtype=jnp.int32),
    )


def reset(key: jax.Array, params: Params) -> State:
    """Start a new episode; the explicit key is unused in this environment."""
    del key
    return State(params.start)


def observe(state: State) -> jax.Array:
    """Return coordinates. Learning features are constructed separately."""
    return state.position


def step(
    key: jax.Array, state: State, action: jax.Array | int, params: Params
) -> tuple[State, jax.Array, jax.Array, jax.Array]:
    """Move once, then emit reward and termination; never automatically reset.

    State must be traversable and action must be in [0, 4). Goal states are
    absorbing with zero reward. This function works eagerly, under JIT, and VMAP.
    """
    del key
    moves = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)
    candidate = state.position + moves[action]
    bounds = jnp.array(params.walls.shape, dtype=jnp.int32)
    inside = jnp.all((candidate >= 0) & (candidate < bounds))
    safe_index = jnp.clip(candidate, 0, bounds - 1)
    blocked = (~inside) | params.walls[safe_index[0], safe_index[1]]
    already_terminal = jnp.all(state.position == params.goal)
    position = jnp.where(blocked | already_terminal, state.position, candidate)
    terminated = jnp.all(position == params.goal)
    reward = jnp.where(
        already_terminal,
        0.0,
        jnp.where(
            terminated,
            1.0,
            jnp.where(params.penalties[position[0], position[1]], -1.0, 0.0),
        ),
    )
    next_state = State(position)
    return next_state, observe(next_state), reward, terminated


def legal_positions(params: Params) -> tuple[tuple[int, int], ...]:
    """Enumerate traversable positions, including the goal, for host-side oracles."""
    return tuple(
        (row, column)
        for row, cells in enumerate(params.walls.tolist())
        for column, is_wall in enumerate(cells)
        if not is_wall
    )
