"""Independent Python-double oracle for proper stochastic option models."""

import math
from dataclasses import dataclass

from rrs.rl.exact import DeterministicModel
from rrs.rl.option_models import ExactOptionModel
from rrs.rl.stochastic_exact import StochasticModel, Transition
from rrs.rl.stochastic_subtasks import _policy_rows


def _stochastic(model):
    """Embed deterministic dynamics without changing existing callers."""
    if isinstance(model, DeterministicModel):
        return StochasticModel(
            tuple(
                tuple(
                    (Transition(1.0, ns, model.rewards[s][a], model.terminated[s][a]),)
                    for a, ns in enumerate(row)
                )
                for s, row in enumerate(model.next_states)
            )
        )
    return model


@dataclass(frozen=True)
class StochasticModelResult:
    model: ExactOptionModel
    iterations: int
    bellman_residual: float
    error_bound: float


def _assert_proper(model, pi, beta, terminal):
    """Finite chains are proper iff every state has a positive-probability stop path."""
    reachable = {s for s, done in enumerate(terminal) if done}
    while True:
        previous = len(reachable)
        for s, row in enumerate(pi):
            for a, p in enumerate(row):
                for outcome in model.transitions[s][a]:
                    ns = outcome.next_state
                    if (
                        p > 0
                        and outcome.probability > 0
                        and (
                            outcome.terminated
                            or terminal[ns]
                            or beta[ns] > 0
                            or ns in reachable
                        )
                    ):
                        reachable.add(s)
        if len(reachable) == previous:
            break
    if len(reachable) != len(terminal):
        raise ValueError("Option can enter a class of states that never stop")


def _model_system(model, pi, beta, features, terminal, gamma):
    width = 1 + len(features[0])  # reward followed by successor features
    offsets, kernel = [], []
    for s, row in enumerate(pi):
        offset, continuation = [0.0] * width, {}
        if not terminal[s]:
            for a, p_action in enumerate(row):
                for outcome in model.transitions[s][a]:
                    ns = outcome.next_state
                    p = p_action * outcome.probability
                    offset[0] += p * outcome.reward
                    if outcome.terminated or terminal[ns]:
                        continue
                    for j, x in enumerate(features[ns], start=1):
                        offset[j] += p * gamma * beta[ns] * x
                    coefficient = p * gamma * (1 - beta[ns])
                    if coefficient:
                        continuation[ns] = continuation.get(ns, 0.0) + coefficient
        offsets.append(tuple(offset))
        kernel.append(tuple(continuation.items()))
    return offsets, kernel


def _backup(offsets, kernel, values):
    return tuple(
        tuple(
            value + sum(p * values[ns][j] for ns, p in transitions)
            for j, value in enumerate(offset)
        )
        for offset, transitions in zip(offsets, kernel, strict=True)
    )


def stochastic_option_model(
    model: DeterministicModel | StochasticModel,
    probabilities,
    stopping,
    features,
    terminal_states,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 20_000,
) -> StochasticModelResult:
    """Equations 12/15 under fixed pi/beta, including an action at stopping sources.

    Returns model tables in state-row order; as_linear_model exports feature
    columns for the existing prediction API. Residual/(1-gamma) bounds the
    absolute error in every reward and successor coefficient.
    """
    model = _stochastic(model)
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
    if any(len(v) != n for v in (stopping, features, terminal_states)):
        raise ValueError("Option, features and terminal vectors must match the model")
    if any(not math.isfinite(b) or not 0 <= b <= 1 for b in stopping):
        raise ValueError("Stopping probabilities must be finite and in [0,1]")
    width = len(features[0])
    if not width or any(
        len(row) != width or any(not math.isfinite(x) for x in row) for row in features
    ):
        raise ValueError("Features must have nonempty finite rectangular shape")
    if any(
        done and any(row) for done, row in zip(terminal_states, features, strict=True)
    ):
        raise ValueError("Terminal features must be zero")
    pi = _policy_rows(model, probabilities)
    _assert_proper(model, pi, stopping, terminal_states)
    offsets, kernel = _model_system(
        model, pi, stopping, features, terminal_states, gamma
    )
    values = ((0.0,) * (width + 1),) * n
    targets = _backup(offsets, kernel, values)
    for iteration in range(1, max_iterations + 1):
        values = targets
        targets = _backup(offsets, kernel, values)
        residual = max(
            abs(v - t)
            for row, target in zip(values, targets, strict=True)
            for v, t in zip(row, target, strict=True)
        )
        error = residual / (1 - gamma)
        if error <= tolerance:
            exact = ExactOptionModel(
                tuple(row[0] for row in values), tuple(row[1:] for row in values)
            )
            return StochasticModelResult(exact, iteration, residual, error)
    raise RuntimeError(f"Stochastic model did not converge (error bound {error:.3g})")
