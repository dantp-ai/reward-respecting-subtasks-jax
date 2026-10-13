"""Independent Python rollouts auditing frozen option reward/feature models."""

import bisect
import itertools
import math
import random

from rrs.envs import reference


def _mean_and_se(total, squares, count):
    mean = total / count
    variance = max(0.0, (squares - total * mean) / (count - 1))
    return mean, math.sqrt(variance / count)


def sample_model(
    positions,
    probabilities,
    stopping,
    start,
    seed,
    samples=10_000,
    gamma=0.99,
    max_steps=10_000,
    feature_rows=None,
):
    """Sample environment rewards and gamma**K final features (default one-hot).

    Sampling uses Python's explicit seeded RNG and the independent environment.
    A horizon overrun raises instead of silently biasing a truncated model.
    """
    if samples < 2 or max_steps < 1 or not 0 <= gamma < 1:
        raise ValueError(
            "Require at least two samples, positive horizon and gamma in [0,1)"
        )
    state_index = {position: i for i, position in enumerate(positions)}
    features = {
        position: i for i, position in enumerate(p for p in positions if p != (6, 10))
    }
    if feature_rows is None:
        feature_rows = [
            [float(features.get(p) == j) for j in range(len(features))]
            for p in positions
        ]
    width = len(feature_rows[0]) if len(feature_rows) else 0
    if (
        len(feature_rows) != len(positions)
        or not width
        or any(
            len(row) != width or any(not math.isfinite(x) for x in row)
            for row in feature_rows
        )
    ):
        raise ValueError("Feature rows must be finite and match positions")
    if any(feature_rows[state_index[(6, 10)]]):
        raise ValueError("Terminal features must be zero")
    cumulative = [tuple(itertools.accumulate(row)) for row in probabilities]
    rng = random.Random(seed)
    reward_sum, reward_squares = 0.0, 0.0
    successor_sum, successor_squares = [0.0] * width, [0.0] * width
    longest = 0
    for _ in range(samples):
        position, total, discount = start, 0.0, 1.0
        if position == (6, 10):
            continue
        for count in range(1, max_steps + 1):
            row = cumulative[state_index[position]]
            action = bisect.bisect_right(row, rng.random() * row[-1])
            position, reward, done = reference.step(position, action)
            total += discount * reward
            discount *= gamma
            if done or rng.random() < stopping[state_index[position]]:
                if not done:
                    for feature, x in enumerate(feature_rows[state_index[position]]):
                        successor_sum[feature] += discount * x
                        successor_squares[feature] += (discount * x) ** 2
                longest = max(longest, count)
                break
        else:
            raise RuntimeError("Rollout exceeded max_steps before stopping")
        reward_sum += total
        reward_squares += total * total
    reward, reward_se = _mean_and_se(reward_sum, reward_squares, samples)
    pairs = [
        _mean_and_se(total, square, samples)
        for total, square in zip(successor_sum, successor_squares, strict=True)
    ]
    return {
        "start": start,
        "seed": seed,
        "samples": samples,
        "max_steps": max_steps,
        "longest_rollout": longest,
        "reward_mean": reward,
        "reward_se": reward_se,
        "successor_mean": [p[0] for p in pairs],
        "successor_se": [p[1] for p in pairs],
    }


def audit_hallway_model(data, probabilities, stopping, exact):
    """Frozen audit starts, seeds and tolerances from model_learning_contract.md."""
    audits = []
    for seed, start in zip(
        range(20260, 20264), ((3, 1), (1, 13), (3, 7), (3, 3)), strict=True
    ):
        result = sample_model(data.positions, probabilities, stopping, start, seed)
        index = data.positions.index(start)
        result["reward_exact"] = exact.rewards[index]
        result["successor_exact"] = exact.successors[index]
        result["reward_passed"] = (
            abs(result["reward_mean"] - exact.rewards[index])
            <= 0.01 + 6 * result["reward_se"]
        )
        result["successor_passed"] = all(
            abs(mean - target) <= 0.01 + 6 * se
            for mean, target, se in zip(
                result["successor_mean"],
                exact.successors[index],
                result["successor_se"],
                strict=True,
            )
        )
        audits.append(result)
    return audits
