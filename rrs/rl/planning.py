"""Asynchronous tabular and feature-facing planning, paper Equations 18–19."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.rl.option_models import LinearExpectationModel


class TabularPlan(NamedTuple):
    values: tuple[float, ...]
    updates: int
    lookaheads: int


def tabular_plan(rewards, successors, state_order, initial=None) -> TabularPlan:
    """Equation 18 in Python double, with tables indexed [state][option][next].

    Successors already include duration discounts and exclude terminal mass.
    The explicit update sequence makes asynchronous order and query cost visible.
    """
    n_states = len(rewards)
    n_options = len(rewards[0]) if n_states else 0
    if (
        not n_states
        or not n_options
        or len(successors) != n_states
        or any(len(row) != n_options for row in rewards)
        or any(
            len(row) != n_options or any(len(p) != n_states for p in row)
            for row in successors
        )
    ):
        raise ValueError("Reward and successor tables must have compatible shape")
    values = [0.0] * n_states if initial is None else list(initial)
    if len(values) != n_states or any(not math.isfinite(v) for v in values):
        raise ValueError("Initial values must be finite and match the states")
    updates = 0
    for state in state_order:
        if not 0 <= state < n_states:
            raise ValueError("Update state is out of range")
        values[state] = max(
            r + sum(p * v for p, v in zip(row, values, strict=True))
            for r, row in zip(rewards[state], successors[state], strict=True)
        )
        updates += 1
    return TabularPlan(tuple(values), updates, updates * n_options)


class PlanningInfo(NamedTuple):
    action_values: jax.Array
    delta: jax.Array
    lookaheads: int


def backed_up_values(weights, features, models: LinearExpectationModel):
    """One backup per model; no additional gamma or artificial stopping reward."""
    reward, successor = models.predict(features)
    return reward + successor @ weights


def planning_step(weights, features, models: LinearExpectationModel, alpha=1.0):
    """Equation 19: compute the target first, then take the linear semi-gradient."""
    action_values = backed_up_values(weights, features, models)
    delta = jnp.max(action_values) - jnp.dot(weights, features)
    updated = weights + alpha * delta * features
    return updated, PlanningInfo(action_values, delta, action_values.shape[0])
