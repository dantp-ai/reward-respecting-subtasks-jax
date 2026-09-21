"""Double-precision tabular oracles for finite deterministic MDPs."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeterministicModel:
    """Rows are states, columns are actions; terminal transitions do not bootstrap."""

    next_states: tuple[tuple[int, ...], ...]
    rewards: tuple[tuple[float, ...], ...]
    terminated: tuple[tuple[bool, ...], ...]

    def __post_init__(self):
        n_states = len(self.next_states)
        if not n_states or not self.next_states[0]:
            raise ValueError("Model must contain states and actions")
        n_actions = len(self.next_states[0])
        if any(
            len(table) != n_states or any(len(row) != n_actions for row in table)
            for table in (self.next_states, self.rewards, self.terminated)
        ):
            raise ValueError("Model tables must have the same rectangular shape")
        if any(not 0 <= s < n_states for row in self.next_states for s in row):
            raise ValueError("Next-state indices must be in range")
        if any(not math.isfinite(r) for row in self.rewards for r in row):
            raise ValueError("Rewards must be finite")


@dataclass(frozen=True)
class ValueIterationResult:
    values: tuple[float, ...]
    greedy_actions: tuple[tuple[int, ...], ...]
    iterations: int
    bellman_residual: float


def _action_values(model, values, gamma):
    # Bellman optimality backup for the return in paper Eq. 1:
    # Q(s,a) = r(s,a) + gamma * (1 - done(s,a)) * V(next(s,a)).
    return tuple(
        tuple(
            reward + (0.0 if done else gamma * values[next_state])
            for next_state, reward, done in zip(
                next_row, reward_row, done_row, strict=True
            )
        )
        for next_row, reward_row, done_row in zip(
            model.next_states, model.rewards, model.terminated, strict=True
        )
    )


def value_iteration(
    model: DeterministicModel,
    gamma: float = 0.99,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    """Solve by synchronous Bellman backups, reporting residual of returned V.

    Python floats retain double precision without changing JAX's global config.
    All actions within 1e-12 of the maximum are reported as greedy ties.
    Raise RuntimeError if the requested residual is not reached.
    """
    if not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    values = (0.0,) * len(model.next_states)
    action_values = _action_values(model, values, gamma)
    for iteration in range(1, max_iterations + 1):
        values = tuple(max(row) for row in action_values)
        action_values = _action_values(model, values, gamma)
        residual = max(
            abs(max(row) - value)
            for row, value in zip(action_values, values, strict=True)
        )
        if residual <= tolerance:
            greedy_actions = tuple(
                tuple(a for a, q in enumerate(row) if abs(q - max(row)) <= 1e-12)
                for row in action_values
            )
            return ValueIterationResult(values, greedy_actions, iteration, residual)
    raise RuntimeError(
        f"Value iteration did not converge in {max_iterations} iterations "
        f"(Bellman residual {residual:.3g})"
    )
