"""Feature-attainment stopping values and exact GVF optimal control.

The oracle solves paper Equation 2; see notes/hallway_options_contract.md for
the distinction between its optimal-stopping comparison and Equation 9.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from rrs.rl.exact import DeterministicModel


def stopping_value(
    weights: jax.Array, features: jax.Array, feature_index: int, bonus: float = 1.0
) -> jax.Array:
    """Equation 4: replace one feature weight; preserve the other contributions."""
    return (
        jnp.dot(weights, features)
        - weights[feature_index] * features[feature_index]
        + bonus * features[feature_index]
    )


@dataclass(frozen=True)
class Subtask:
    cumulants: tuple[tuple[float, ...], ...]
    stopping_values: tuple[float, ...]
    can_stop: tuple[bool, ...]


@dataclass(frozen=True)
class SubtaskResult:
    values: tuple[float, ...]
    greedy_actions: tuple[tuple[int, ...], ...]
    stopping: tuple[bool, ...]
    iterations: int
    bellman_residual: float


def _action_values(model, subtask, values, gamma):
    # Eq. 2: after arrival, choose stopping now or discounted continuation.
    arrival = tuple(
        max(z, gamma * v) if allowed else gamma * v
        for z, v, allowed in zip(
            subtask.stopping_values, values, subtask.can_stop, strict=True
        )
    )
    return tuple(
        tuple(
            c + (0.0 if done else arrival[next_state])
            for next_state, c, done in zip(
                next_row, cumulant_row, done_row, strict=True
            )
        )
        for next_row, cumulant_row, done_row in zip(
            model.next_states, subtask.cumulants, model.terminated, strict=True
        )
    )


def solve_subtask(
    model: DeterministicModel,
    subtask: Subtask,
    terminal_states: tuple[bool, ...],
    gamma: float = 0.99,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> SubtaskResult:
    """Optimize policy and arrival stopping, requiring one action on initiation.

    Values are forced-action values, including at states where stopping is best
    on arrival. Already-terminal rows are zero. The exact oracle uses Python
    double precision; it does not change JAX's global precision setting.
    """
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not math.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive and finite")
    n_states = len(model.next_states)
    if any(
        len(row) != n_states
        for row in (subtask.stopping_values, subtask.can_stop, terminal_states)
    ):
        raise ValueError("Subtask and terminal vectors must match the state count")
    # Reuse the exact model's rectangular/finite cumulant checks.
    DeterministicModel(model.next_states, subtask.cumulants, model.terminated)
    if any(not math.isfinite(z) for z in subtask.stopping_values):
        raise ValueError(
            "Stopping values must be finite; use can_stop to forbid stopping"
        )
    if any(
        done and z != 0
        for done, z in zip(terminal_states, subtask.stopping_values, strict=True)
    ):
        raise ValueError("Terminal stopping values must be zero")

    values = (0.0,) * n_states
    action_values = _action_values(model, subtask, values, gamma)
    for iteration in range(1, max_iterations + 1):
        values = tuple(
            0.0 if done else max(row)
            for row, done in zip(action_values, terminal_states, strict=True)
        )
        action_values = _action_values(model, subtask, values, gamma)
        residual = max(
            abs((0.0 if done else max(row)) - value)
            for row, value, done in zip(
                action_values, values, terminal_states, strict=True
            )
        )
        if residual <= tolerance:
            actions = tuple(
                ()
                if done
                else tuple(a for a, q in enumerate(row) if abs(q - max(row)) <= 1e-12)
                for row, done in zip(action_values, terminal_states, strict=True)
            )
            stopping = tuple(
                done or (allowed and z >= gamma * v)
                for done, allowed, z, v in zip(
                    terminal_states,
                    subtask.can_stop,
                    subtask.stopping_values,
                    values,
                    strict=True,
                )
            )
            return SubtaskResult(values, actions, stopping, iteration, residual)
    raise RuntimeError(f"Subtask iteration did not converge (residual {residual:.3g})")
