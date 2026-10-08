"""Independent sampled rewards and gamma**K final features in four rooms."""

import bisect
import itertools
import random

from rrs.envs import four_room_reference
from rrs.experiments.model_audit import _mean_and_se
from rrs.experiments.model_learning import option_tables

STARTS = ((4, 1), (6, 2), (3, 6), (7, 9), (10, 6), (9, 3))


def sample_model(
    positions,
    probabilities,
    stopping,
    start,
    seed,
    samples=10_000,
    gamma=0.99,
    max_steps=10_000,
):
    if samples < 2 or max_steps < 1 or not 0 <= gamma < 1:
        raise ValueError("Require two samples, positive horizon and gamma in [0,1)")
    indices = {p: i for i, p in enumerate(positions)}
    features = {p: i for i, p in enumerate(p for p in positions if p != (9, 7))}
    cumulative = [tuple(itertools.accumulate(row)) for row in probabilities]
    rng = random.Random(seed)
    totals, squares = [0.0] * (1 + len(features)), [0.0] * (1 + len(features))
    longest = 0
    for _ in range(samples):
        position, reward_sum, discount = start, 0.0, 1.0
        if position == (9, 7):
            continue
        for count in range(1, max_steps + 1):
            row = cumulative[indices[position]]
            action = bisect.bisect_right(row, rng.random() * row[-1])
            position, reward, done = four_room_reference.step(position, action, rng)
            reward_sum += discount * reward
            discount *= gamma
            if done or rng.random() < stopping[indices[position]]:
                if not done:
                    j = 1 + features[position]
                    totals[j] += discount
                    squares[j] += discount * discount
                longest = max(longest, count)
                break
        else:
            raise RuntimeError("Rollout exceeded max_steps before stopping")
        totals[0] += reward_sum
        squares[0] += reward_sum * reward_sum
    moments = [
        _mean_and_se(t, q, samples) for t, q in zip(totals, squares, strict=True)
    ]
    return {
        "start": start,
        "seed": seed,
        "samples": samples,
        "max_steps": max_steps,
        "longest_rollout": longest,
        "reward_mean": moments[0][0],
        "reward_se": moments[0][1],
        "successor_mean": [m[0] for m in moments[1:]],
        "successor_se": [m[1] for m in moments[1:]],
    }


def audit_models(data, options, oracles, progress=False):
    pi, beta = option_tables(options, data)
    pis, betas = pi.tolist(), beta.tolist()
    audits = []
    for option in range(4):
        exact = oracles[4 + option].model
        for k, start in enumerate(STARTS):
            result = sample_model(
                data.positions,
                pis[4 + option],
                betas[4 + option],
                start,
                8100 + 6 * option + k,
            )
            index = data.positions.index(start)
            reward_error = abs(result["reward_mean"] - exact.rewards[index])
            reward_limit = 0.01 + 6 * result["reward_se"]
            errors = [
                abs(m - v)
                for m, v in zip(
                    result["successor_mean"], exact.successors[index], strict=True
                )
            ]
            limits = [0.01 + 6 * se for se in result["successor_se"]]
            result.update(
                {
                    "option": f"H{option + 1}",
                    "reward_exact": exact.rewards[index],
                    "successor_exact": exact.successors[index],
                    "reward_error": reward_error,
                    "reward_limit": reward_limit,
                    "successor_errors": errors,
                    "successor_limits": limits,
                    "reward_passed": reward_error <= reward_limit,
                    "successor_passed": all(
                        e <= t for e, t in zip(errors, limits, strict=True)
                    ),
                }
            )
            audits.append(result)
            if progress:
                print(
                    f"Audited H{option + 1} from {start}: {result['samples']:,} rollouts",
                    flush=True,
                )
    return audits
