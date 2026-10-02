"""Exact models of proper deterministic options and feature-facing predictions."""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.rl.exact import DeterministicModel


@dataclass(frozen=True)
class DeterministicOption:
    """Exact-oracle policy action and arrival-stopping flag for each model row."""

    policy: tuple[int, ...]
    stopping: tuple[bool, ...]


class LinearExpectationModel(NamedTuple):
    """Equation 16: predict from features, without environment state indices."""

    reward_weights: jax.Array
    successor_weights: jax.Array

    def predict(self, features: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(self.reward_weights, features), self.successor_weights @ features


@dataclass(frozen=True)
class ExactOptionModel:
    rewards: tuple[float, ...]
    successors: tuple[tuple[float, ...], ...]

    def as_linear_model(
        self, nonterminal_indices: tuple[int, ...]
    ) -> LinearExpectationModel:
        """Export coefficients for one-hot inputs in the supplied feature order.

        Each source state's successor prediction is a COLUMN of W, not a row.
        JAX uses the caller's normal floating-point precision for this export.
        """
        return LinearExpectationModel(
            jnp.array([self.rewards[s] for s in nonterminal_indices]),
            jnp.array([self.successors[s] for s in nonterminal_indices]).T,
        )


def exact_option_model(
    model: DeterministicModel,
    option: DeterministicOption,
    features: tuple[tuple[float, ...], ...],
    terminal_states: tuple[bool, ...],
    gamma: float = 0.99,
) -> ExactOptionModel:
    """Evaluate Eq. 12 and Eq. 15 by memoized one-step recurrences.

    Use actual environment rewards in model.rewards, never subtask cumulants or
    stopping bonuses. Beta is checked AFTER the first action. Reject an option
    with a nonterminating cycle from any nonterminal state instead of truncating.
    This host-side oracle is intended for small finite deterministic problems.
    """
    n_states = len(model.next_states)
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if any(
        len(row) != n_states
        for row in (option.policy, option.stopping, features, terminal_states)
    ):
        raise ValueError(
            "Option, features, and terminal mask must match the state count"
        )
    if any(
        not isinstance(a, int) or not 0 <= a < len(model.next_states[s])
        for s, a in enumerate(option.policy)
    ):
        raise ValueError("Policy actions must be valid action indices")
    width = len(features[0])
    if not width or any(len(row) != width for row in features):
        raise ValueError("Features must have a nonempty rectangular shape")
    if any(not math.isfinite(x) for row in features for x in row):
        raise ValueError("Features must be finite")
    if any(
        done and any(row) for done, row in zip(terminal_states, features, strict=True)
    ):
        raise ValueError("Terminal features must be zero")

    zero = (0.0,) * width
    cache = {}
    active = set()

    def evaluate(state):
        if state in cache:
            return cache[state]
        if terminal_states[state]:
            return 0.0, zero
        if state in active:
            raise ValueError(f"Option has a nonterminating cycle through state {state}")
        active.add(state)
        action = option.policy[state]
        next_state = model.next_states[state][action]
        reward = model.rewards[state][action]
        if model.terminated[state][action] or terminal_states[next_state]:
            successors = zero
        elif option.stopping[next_state]:
            # Eq. 15 discounts once even if K=1, unlike the GVF stopping bonus.
            successors = tuple(gamma * x for x in features[next_state])
        else:
            future_reward, future_features = evaluate(next_state)
            reward += gamma * future_reward
            successors = tuple(gamma * x for x in future_features)
        active.remove(state)
        cache[state] = reward, successors
        return cache[state]

    rows = tuple(evaluate(state) for state in range(n_states))
    return ExactOptionModel(
        tuple(row[0] for row in rows), tuple(row[1] for row in rows)
    )
