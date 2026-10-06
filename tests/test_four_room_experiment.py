import math

import pytest

from rrs.experiments.four_room import audit_policy, build_model, build_report, rollout
from rrs.rl.stochastic_exact import action_values, evaluate_policy, value_iteration


@pytest.fixture(scope="module")
def baseline():
    positions, model = build_model()
    solution = value_iteration(model)
    actions = [row[0] for row in solution.greedy_actions]
    pi = tuple(tuple(float(a == chosen) for a in range(4)) for chosen in actions)
    evaluation = evaluate_policy(model, pi)
    return positions, model, solution, actions, evaluation


def test_four_room_baseline_matches_linear_policy_solve(baseline):
    positions, model, solution, _, evaluation = baseline
    assert len(positions) == len(model.transitions) == 104
    assert solution.values[positions.index((9, 7))] == 0.0
    assert solution.bellman_residual / (1 - 0.99) <= 1e-10
    assert evaluation.error_bound <= 1e-10
    assert evaluation.values == pytest.approx(solution.values, abs=1e-9)
    residual = max(
        abs(max(row) - v)
        for row, v in zip(
            action_values(model, solution.values), solution.values, strict=True
        )
    )
    assert residual == solution.bellman_residual
    assert all(math.isfinite(v) and v <= 1.0 for v in solution.values)


def test_short_return_audit_is_reproducible_and_report_keeps_raw_samples(baseline):
    positions, model, solution, actions, evaluation = baseline
    first = audit_policy(positions, actions, solution.values, episodes=32)
    second = audit_policy(positions, actions, solution.values, episodes=32)
    assert first == second
    assert [a["seed"] for a in first] == [6000, 6001, 6002, 6003]
    for audit in first:
        assert len(audit["returns"]) == len(audit["episode_lengths"]) == 32
        assert audit["truncations"] == 0
        assert audit["tail_bound"] <= 1e-10
        assert audit["tolerance"] == 0.005 + 6 * audit["se"] + audit["tail_bound"]
    report = build_report(positions, model, solution, evaluation, first)
    assert not report["canonical_protocol"]
    assert report["nonterminal_states"] == 103
    assert report["audits"] == first
    assert report["acceptance"]["policy_values_match_optimum_within_1e-9"]
    first[0]["passed"] = False
    first[0]["truncations"] = 1
    failed = build_report(positions, model, solution, evaluation, first)
    assert not failed["acceptance"]["all_four_rollout_audits_pass"]
    assert not failed["acceptance"]["no_truncated_episodes"]


def test_rollout_preserves_signed_rewards_discounts_and_truncation():
    class IntendedDirection:
        def randrange(self, stop):
            assert stop == 9
            return 0

    rng = IntendedDirection()
    # UP is blocked in this penalty cell: three rewards -1, discounted by .5.
    sample = rollout((7, 3), {(7, 3): 0}, rng, gamma=0.5, max_steps=3)
    assert sample == (-1.75, 3, True)
    assert rollout((9, 8), {(9, 8): 2}, rng) == (1.0, 1, False)
    assert rollout((9, 7), {}, rng) == (0.0, 0, False)
