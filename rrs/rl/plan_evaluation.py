"""Appendix B execution policy and independent discounted policy evaluation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.rl.planning import backed_up_values
from rrs.rl.policy_evaluation import _policy_rows


def induced_policy(weights, models, option_policies, features):
    """Equation 20: reselect g(s) at every primitive step, with first-max ties.

    option_policies is [option, state, primitive action]; stopping is not used
    during this evaluation. A selected option supplies one action distribution.
    """
    values = jax.vmap(backed_up_values, in_axes=(None, 0, None))(
        weights, features, models
    )
    choices = jnp.argmax(values, axis=1)
    return option_policies[choices, jnp.arange(features.shape[0])], choices


class PolicyReturn(NamedTuple):
    value: float
    error_bound: float
    reachable_states: int


def _solve(matrix, rhs):
    """Small independent double-precision Gaussian elimination with pivoting."""
    matrix, rhs = [list(row) for row in matrix], list(rhs)
    size = len(rhs)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        rhs[column], rhs[pivot] = rhs[pivot], rhs[column]
        if abs(matrix[column][column]) < 1e-15:
            raise ValueError("Policy evaluation system is singular")
        for row in range(column + 1, size):
            scale = matrix[row][column] / matrix[column][column]
            if scale == 0:
                continue
            matrix[row][column] = 0.0
            for j in range(column + 1, size):
                matrix[row][j] -= scale * matrix[column][j]
            rhs[row] -= scale * rhs[column]
    values = [0.0] * size
    for row in reversed(range(size)):
        values[row] = (
            rhs[row] - sum(matrix[row][j] * values[j] for j in range(row + 1, size))
        ) / matrix[row][row]
    return values


def _reachable(model, pi, start, terminal):
    seen, pending = {start}, [start]
    while pending:
        state = pending.pop()
        for a, p in enumerate(pi[state]):
            ns = model.next_states[state][a]
            if (
                p > 0
                and not model.terminated[state][a]
                and not terminal[ns]
                and ns not in seen
            ):
                seen.add(ns)
                pending.append(ns)
    return sorted(seen)


def evaluate_action_policy(
    model, probabilities, start, terminal_states, gamma=0.99
) -> PolicyReturn:
    """Solve (I-gamma*P)V=r on reachable nonterminals, including closed cycles.

    This host-side oracle uses Python doubles and true environment transitions.
    A verified residual bounds return error; no sampled or truncated trajectories.
    """
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0,1)")
    if len(terminal_states) != len(model.next_states) or not 0 <= start < len(
        terminal_states
    ):
        raise ValueError("Terminal mask and start must match the model")
    pi = _policy_rows(model, probabilities)
    if terminal_states[start]:
        return PolicyReturn(0.0, 0.0, 0)
    states = _reachable(model, pi, start, terminal_states)
    indices = {s: i for i, s in enumerate(states)}
    matrix, rewards = [], []
    for s in states:
        row, reward = [0.0] * len(states), 0.0
        row[indices[s]] = 1.0
        for a, p in enumerate(pi[s]):
            ns = model.next_states[s][a]
            reward += p * model.rewards[s][a]
            if p and not model.terminated[s][a] and not terminal_states[ns]:
                row[indices[ns]] -= gamma * p
        matrix.append(row)
        rewards.append(reward)
    values = _solve(matrix, rewards)
    residual = max(
        abs(sum(a * v for a, v in zip(row, values, strict=True)) - r)
        for row, r in zip(matrix, rewards, strict=True)
    )
    error = residual / (1 - gamma)
    if error > 1e-8:
        raise RuntimeError(
            f"Policy evaluation error bound exceeds tolerance: {error:.3g}"
        )
    return PolicyReturn(values[indices[start]], error, len(states))
