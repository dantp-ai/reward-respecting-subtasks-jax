"""Python-double evaluation of a fixed stochastic option, independently of JAX.

Success means stopping at the target without any negative-reward arrival.
Source stopping never removes the first action. Interval iteration bounds the
undiscounted success probability; no sampled paths or horizon truncation.
"""

import math
from dataclasses import dataclass

from rrs.rl.exact import DeterministicModel


@dataclass(frozen=True)
class PolicyEvaluation:
    values: tuple[float, ...]
    safe_success: tuple[float, ...]
    iterations: int
    value_error_bound: float
    probability_gap: float


def _policy_rows(model, probabilities):
    if len(probabilities) != len(model.next_states):
        raise ValueError("Policy probabilities must match the model shape")
    result = []
    for pi, actions in zip(probabilities, model.next_states, strict=True):
        if (
            len(pi) != len(actions)
            or any(not math.isfinite(p) or p < 0 for p in pi)
            or abs(sum(pi) - 1) > 1e-6
        ):
            raise ValueError(
                "Policy probabilities must be finite, nonnegative and sum to one"
            )
        # Remove only roundoff from a float32 softmax export.
        result.append(tuple(p / sum(pi) for p in pi))
    return result


def _kernels(model, probabilities, stopping, z, terminal, target, gamma):
    rewards, successes, value_kernel, safe_kernel = [], [], [], []
    for s, pi in enumerate(probabilities):
        reward, success, value_row, safe_row = 0.0, 0.0, {}, {}
        if not terminal[s]:
            for a, p in enumerate(pi):
                ns, r, done = (
                    model.next_states[s][a],
                    model.rewards[s][a],
                    model.terminated[s][a],
                )
                done = done or terminal[ns]
                stop = done or stopping[ns]
                reward += p * (r + (z[ns] if stop and not done else 0.0))
                if not stop:
                    value_row[ns] = value_row.get(ns, 0.0) + gamma * p
                if r >= 0 and not done:
                    success += p * (stop and ns == target)
                    if not stop:
                        safe_row[ns] = safe_row.get(ns, 0.0) + p
        rewards.append(reward)
        successes.append(success)
        value_kernel.append(tuple(value_row.items()))
        safe_kernel.append(tuple(safe_row.items()))
    return rewards, successes, value_kernel, safe_kernel


def _backup(offsets, kernel, values):
    return tuple(
        offset + sum(p * values[ns] for ns, p in row)
        for offset, row in zip(offsets, kernel, strict=True)
    )


def evaluate_policy(
    model: DeterministicModel,
    probabilities,
    stopping,
    stopping_values,
    terminal_states,
    target_state: int,
    gamma: float = 0.99,
    tolerance: float = 1e-8,
    max_iterations: int = 20_000,
) -> PolicyEvaluation:
    """Equation 2 under a frozen policy/beta, plus bounded safe attainment.

    A residual divided by (1-gamma) bounds value error. Lower/upper probability
    iterates bound safe success; unresolved recurrent classes raise explicitly.
    """
    n = len(model.next_states)
    if (
        not 0 <= gamma < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or max_iterations < 1
    ):
        raise ValueError(
            "Require gamma in [0,1), positive tolerance and iteration budget"
        )
    if not 0 <= target_state < n or any(
        len(v) != n for v in (stopping, stopping_values, terminal_states)
    ):
        raise ValueError("Stopping/terminal vectors and target must match the model")
    if any(
        not math.isfinite(z) or (done and z != 0)
        for z, done in zip(stopping_values, terminal_states, strict=True)
    ):
        raise ValueError("Stopping values must be finite and zero at terminals")
    probabilities = _policy_rows(model, probabilities)
    rewards, successes, value_kernel, safe_kernel = _kernels(
        model,
        probabilities,
        stopping,
        stopping_values,
        terminal_states,
        target_state,
        gamma,
    )
    values, lower = (0.0,) * n, (0.0,) * n
    upper = tuple(0.0 if done else 1.0 for done in terminal_states)
    error, gap = math.inf, math.inf
    for iteration in range(1, max_iterations + 1):
        if error > tolerance:
            next_values = _backup(rewards, value_kernel, values)
            error = max(
                abs(v - w) for v, w in zip(values, next_values, strict=True)
            ) / (1 - gamma)
            values = next_values
        if gap > tolerance:
            lower = _backup(successes, safe_kernel, lower)
            upper = _backup(successes, safe_kernel, upper)
            gap = max(u - lo for lo, u in zip(lower, upper, strict=True))
        if error <= tolerance and gap <= tolerance:
            return PolicyEvaluation(values, lower, iteration, error, gap)
    raise RuntimeError(
        f"Policy evaluation did not converge: value bound={error:.3g}, success gap={gap:.3g}"
    )
