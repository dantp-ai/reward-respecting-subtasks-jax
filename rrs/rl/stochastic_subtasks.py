"""Exact Equation 2 control and fixed-option evaluation in stochastic MDPs."""

import math
from dataclasses import dataclass

from rrs.rl.subtasks import SubtaskResult


def _validate(model, z, terminal, gamma, tolerance, max_iterations):
    n = len(model.transitions)
    if (
        not 0 <= gamma < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or max_iterations < 1
    ):
        raise ValueError(
            "Require gamma in [0,1), positive tolerance and iteration budget"
        )
    if len(z) != n or len(terminal) != n:
        raise ValueError("Stopping and terminal vectors must match the model")
    if any(
        not math.isfinite(v) or (done and v != 0)
        for v, done in zip(z, terminal, strict=True)
    ):
        raise ValueError("Stopping values must be finite and zero at terminals")


def action_values(model, values, z, can_stop, terminal, gamma):
    """Outcome rewards are cumulants; stopping is chosen after arrival."""
    arrival = tuple(
        max(stop, gamma * v) if allowed else gamma * v
        for stop, v, allowed in zip(z, values, can_stop, strict=True)
    )
    return tuple(
        tuple(
            math.fsum(
                o.probability
                * (
                    o.reward
                    + (
                        0.0
                        if o.terminated or terminal[o.next_state]
                        else arrival[o.next_state]
                    )
                )
                for o in outcomes
            )
            for outcomes in row
        )
        for row in model.transitions
    )


def solve_subtask(
    model,
    stopping_values,
    terminal_states,
    can_stop=None,
    gamma=0.99,
    tolerance=1e-12,
    max_iterations=20_000,
):
    """Optimize forced-action returns and arrival stopping in Python doubles."""
    z, terminal = tuple(stopping_values), tuple(terminal_states)
    _validate(model, z, terminal, gamma, tolerance, max_iterations)
    allowed = (True,) * len(z) if can_stop is None else tuple(can_stop)
    if len(allowed) != len(z):
        raise ValueError("Stopping permissions must match the model")
    values = (0.0,) * len(z)
    q = action_values(model, values, z, allowed, terminal, gamma)
    for iteration in range(1, max_iterations + 1):
        values = tuple(
            0.0 if done else max(row) for row, done in zip(q, terminal, strict=True)
        )
        q = action_values(model, values, z, allowed, terminal, gamma)
        residual = max(
            abs((0.0 if done else max(row)) - v)
            for row, v, done in zip(q, values, terminal, strict=True)
        )
        if residual <= tolerance:
            greedy = tuple(
                ()
                if done
                else tuple(a for a, v in enumerate(row) if abs(v - max(row)) <= 1e-12)
                for row, done in zip(q, terminal, strict=True)
            )
            stopping = tuple(
                done or (allow and stop >= gamma * v)
                for done, allow, stop, v in zip(
                    terminal, allowed, z, values, strict=True
                )
            )
            return SubtaskResult(values, greedy, stopping, iteration, residual)
    raise RuntimeError(
        f"Stochastic subtask iteration did not converge; residual {residual:.3g}"
    )


@dataclass(frozen=True)
class OptionEvaluation:
    values: tuple[float, ...]
    target_probability: tuple[float, ...]
    safe_target_probability: tuple[float, ...]
    penalty_probability: tuple[float, ...]
    value_error_bound: float
    probability_gap: float
    iterations: int


def _policy_rows(model, probabilities):
    if len(probabilities) != len(model.transitions):
        raise ValueError("Policy dimensions must match the model")
    result = []
    for pi, row in zip(probabilities, model.transitions, strict=True):
        if (
            len(pi) != len(row)
            or any(not math.isfinite(p) or p < 0 for p in pi)
            or abs(math.fsum(pi) - 1) > 1e-6
        ):
            raise ValueError("Policy rows must be finite probability distributions")
        total = math.fsum(pi)
        result.append(tuple(p / total for p in pi))
    return result


def _kernels(model, pi, beta, z, terminal, target):
    rewards, hits, safe_hits, penalties, continuation, safe_continuation = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for s, row in enumerate(pi):
        reward, hit, safe_hit, penalty = 0.0, 0.0, 0.0, 0.0
        kernel, safe_kernel = {}, {}
        for a, p_action in enumerate(row):
            if terminal[s]:
                break
            for o in model.transitions[s][a]:
                p, ns = p_action * o.probability, o.next_state
                done = o.terminated or terminal[ns]
                stop = 1.0 if done else beta[ns]
                reward += p * (o.reward + (0.0 if done else stop * z[ns]))
                target_mass = p * stop if ns == target and not done else 0.0
                hit += target_mass
                weight = p * (1 - stop)
                if weight:
                    kernel[ns] = kernel.get(ns, 0.0) + weight
                if o.reward < 0:
                    penalty += p
                else:
                    safe_hit += target_mass
                    if weight:
                        safe_kernel[ns] = safe_kernel.get(ns, 0.0) + weight
        rewards.append(reward)
        hits.append(hit)
        safe_hits.append(safe_hit)
        penalties.append(penalty)
        continuation.append(tuple(kernel.items()))
        safe_continuation.append(tuple(safe_kernel.items()))
    return rewards, (hits, safe_hits, penalties), continuation, safe_continuation


def _backup(offsets, kernel, values, discount=1.0):
    return tuple(
        offset + discount * math.fsum(p * values[ns] for ns, p in row)
        for offset, row in zip(offsets, kernel, strict=True)
    )


def evaluate_option(
    model,
    probabilities,
    stopping,
    stopping_values,
    terminal_states,
    target_state,
    gamma=0.99,
    tolerance=1e-8,
    max_iterations=20_000,
):
    """Independent stochastic option return and bounded outcome probabilities.

    Termination is checked after arrival; initial beta does not suppress action.
    Safe target success excludes every penalty, including on the stopping action.
    """
    z, terminal, beta = tuple(stopping_values), tuple(terminal_states), tuple(stopping)
    _validate(model, z, terminal, gamma, tolerance, max_iterations)
    if not 0 <= target_state < len(z) or terminal[target_state]:
        raise ValueError("Target must be a nonterminal state")
    if len(beta) != len(z) or any(
        not math.isfinite(b) or not 0 <= b <= 1 for b in beta
    ):
        raise ValueError("Stopping probabilities must match states and lie in [0,1]")
    pi = _policy_rows(model, probabilities)
    rewards, offsets, continuation, safe = _kernels(
        model, pi, beta, z, terminal, target_state
    )
    kernels = (continuation, safe, safe)
    values = (0.0,) * len(z)
    lower = ((0.0,) * len(z),) * 3
    upper = (tuple(0.0 if done else 1.0 for done in terminal),) * 3
    error, gap = math.inf, math.inf
    for iteration in range(1, max_iterations + 1):
        if error > tolerance:
            values = _backup(rewards, continuation, values, gamma)
            next_values = _backup(rewards, continuation, values, gamma)
            error = max(
                abs(a - b) for a, b in zip(values, next_values, strict=True)
            ) / (1 - gamma)
        if gap > tolerance:
            lower = tuple(
                _backup(off, kernel, v)
                for off, kernel, v in zip(offsets, kernels, lower, strict=True)
            )
            upper = tuple(
                _backup(off, kernel, v)
                for off, kernel, v in zip(offsets, kernels, upper, strict=True)
            )
            gap = max(
                hi - lo
                for low, high in zip(lower, upper, strict=True)
                for lo, hi in zip(low, high, strict=True)
            )
        if error <= tolerance and gap <= tolerance:
            return OptionEvaluation(values, *lower, error, gap, iteration)
    raise RuntimeError(
        f"Option evaluation did not converge: value bound={error:.3g}, probability gap={gap:.3g}"
    )
