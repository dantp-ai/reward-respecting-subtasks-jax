"""Python-double Bellman and fixed-policy oracles for sparse stochastic MDPs."""

import math
from dataclasses import dataclass
from typing import NamedTuple

from rrs.rl.exact import ValueIterationResult
from rrs.rl.plan_evaluation import _solve


class Transition(NamedTuple):
    probability: float
    next_state: int
    reward: float
    terminated: bool


@dataclass(frozen=True)
class StochasticModel:
    """Outcome lists indexed [state][action]; terminal flags belong to outcomes."""

    transitions: tuple[tuple[tuple[Transition, ...], ...], ...]

    def __post_init__(self):
        n_states = len(self.transitions)
        n_actions = len(self.transitions[0]) if n_states else 0
        if (
            not n_states
            or not n_actions
            or any(len(row) != n_actions for row in self.transitions)
        ):
            raise ValueError(
                "Model must have a nonempty rectangular state-action table"
            )
        for row in self.transitions:
            for outcomes in row:
                if not outcomes or any(
                    not math.isfinite(o.probability) or o.probability < 0
                    for o in outcomes
                ):
                    raise ValueError(
                        "Outcome probabilities must be finite and nonnegative"
                    )
                if abs(math.fsum(o.probability for o in outcomes) - 1) > 1e-12:
                    raise ValueError("Outcome probabilities must sum to one")
                if any(
                    not isinstance(o.next_state, int)
                    or not 0 <= o.next_state < n_states
                    for o in outcomes
                ):
                    raise ValueError("Next-state indices must be in range")
                if any(not math.isfinite(o.reward) for o in outcomes):
                    raise ValueError("Rewards must be finite")


def _validate_discount(gamma):
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0,1)")


def action_values(model: StochasticModel, values, gamma=0.99):
    """Expected Bellman backup; terminal arrival rewards are retained."""
    return tuple(
        tuple(
            math.fsum(
                o.probability
                * (o.reward + (0.0 if o.terminated else gamma * values[o.next_state]))
                for o in outcomes
            )
            for outcomes in row
        )
        for row in model.transitions
    )


def value_iteration(
    model: StochasticModel, gamma=0.99, tolerance=1e-12, max_iterations=20_000
):
    """Synchronous backups with an independently recomputed final residual."""
    _validate_discount(gamma)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    values = (0.0,) * len(model.transitions)
    q = action_values(model, values, gamma)
    for iteration in range(1, max_iterations + 1):
        values = tuple(max(row) for row in q)
        q = action_values(model, values, gamma)
        residual = max(abs(max(row) - v) for row, v in zip(q, values, strict=True))
        if residual <= tolerance:
            greedy = tuple(
                tuple(
                    a for a, value in enumerate(row) if abs(value - max(row)) <= 1e-12
                )
                for row in q
            )
            return ValueIterationResult(values, greedy, iteration, residual)
    raise RuntimeError(
        f"Stochastic value iteration did not converge; residual {residual:.3g}"
    )


class PolicyValue(NamedTuple):
    values: tuple[float, ...]
    bellman_residual: float
    error_bound: float


def evaluate_policy(model: StochasticModel, probabilities, gamma=0.99) -> PolicyValue:
    """Solve (I-gamma*P_pi)V=r_pi, including cycles and terminal outcomes."""
    _validate_discount(gamma)
    if len(probabilities) != len(model.transitions):
        raise ValueError("Policy dimensions must match the model")
    matrix, rewards = [], []
    for s, (pi, row) in enumerate(zip(probabilities, model.transitions, strict=True)):
        if len(pi) != len(row):
            raise ValueError("Policy dimensions must match the model")
        if (
            any(not math.isfinite(p) or p < 0 for p in pi)
            or abs(math.fsum(pi) - 1) > 1e-12
        ):
            raise ValueError("Policy rows must be finite probability distributions")
        coefficients = [0.0] * len(model.transitions)
        coefficients[s] = 1.0
        reward = 0.0
        for p_action, outcomes in zip(pi, row, strict=True):
            for outcome in outcomes:
                p = p_action * outcome.probability
                reward += p * outcome.reward
                if not outcome.terminated:
                    coefficients[outcome.next_state] -= gamma * p
        matrix.append(coefficients)
        rewards.append(reward)
    values = tuple(_solve(matrix, rewards))
    q = action_values(model, values, gamma)
    residual = max(
        abs(math.fsum(p * v for p, v in zip(pi, row, strict=True)) - value)
        for pi, row, value in zip(probabilities, q, values, strict=True)
    )
    return PolicyValue(values, residual, residual / (1 - gamma))
