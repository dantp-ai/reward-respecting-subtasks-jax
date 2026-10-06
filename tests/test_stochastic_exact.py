import math

import pytest

from rrs.rl.exact import DeterministicModel, value_iteration as deterministic_vi
from rrs.rl.stochastic_exact import (
    StochasticModel,
    Transition as T,
    action_values,
    evaluate_policy,
    value_iteration,
)


def test_hand_backup_keeps_terminal_reward_and_masks_only_its_bootstrap():
    model = StochasticModel(
        (
            (
                (T(0.25, 1, 4.0, True), T(0.75, 0, -1.0, False)),
                (T(1.0, 1, 0.0, False),),
            ),
            ((T(1.0, 1, 0.0, True),),) * 2,
        )
    )
    # V0=2,V1=10,gamma=.5 => Q0=.25*4+.75*(-1+.5*2)=1; Q1=5.
    assert action_values(model, (2.0, 10.0), 0.5)[0] == (1.0, 5.0)


def test_stochastic_self_loop_value_and_greedy_policy_solve():
    model = StochasticModel(
        (
            ((T(0.5, 0, 0.0, False), T(0.5, 1, 1.0, True)), (T(1.0, 1, 0.5, True),)),
            ((T(1.0, 1, 0.0, True),),) * 2,
        )
    )
    expected = 0.5 / (1 - 0.9 * 0.5)
    result = value_iteration(model, gamma=0.9)
    assert result.values == pytest.approx((expected, 0.0), abs=1e-11)
    assert result.greedy_actions[0] == (0,)
    evaluated = evaluate_policy(model, ((1.0, 0.0), (1.0, 0.0)), gamma=0.9)
    assert evaluated.values == pytest.approx((expected, 0.0), abs=1e-14)
    assert evaluated.error_bound <= 1e-12
    # A stochastic action policy has r=.5 and continuing mass=.25.
    mixed = evaluate_policy(model, ((0.5, 0.5), (0.5, 0.5)), gamma=0.9)
    assert mixed.values[0] == pytest.approx(0.5 / (1 - 0.9 * 0.25))


def test_discounted_closed_cycles_and_nonconvergence():
    model = StochasticModel((((T(1.0, 0, -1.0, False),),),))
    assert evaluate_policy(model, ((1.0,),), gamma=0.5).values == (-2.0,)
    with pytest.raises(RuntimeError, match="converge"):
        value_iteration(model, gamma=0.99, max_iterations=1)


def test_deterministic_model_is_a_special_case_and_ties_are_kept():
    deterministic = DeterministicModel(
        ((1, 1), (1, 1)), ((1.0, 1.0), (0.0, 0.0)), ((True, True), (True, True))
    )
    stochastic = StochasticModel(
        tuple(
            tuple(
                (T(1.0, ns, reward, done),)
                for ns, reward, done in zip(ns_row, r_row, d_row, strict=True)
            )
            for ns_row, r_row, d_row in zip(
                deterministic.next_states,
                deterministic.rewards,
                deterministic.terminated,
                strict=True,
            )
        )
    )
    assert value_iteration(stochastic) == deterministic_vi(deterministic)
    assert value_iteration(stochastic).greedy_actions[0] == (0, 1)


@pytest.mark.parametrize(
    "outcomes,message",
    [
        ((), "probabilities"),
        ((T(-1.0, 0, 0.0, False), T(2.0, 0, 0.0, False)), "probabilities"),
        ((T(math.nan, 0, 0.0, False),), "probabilities"),
        ((T(0.5, 0, 0.0, False),), "sum to one"),
        ((T(1.0, 1, 0.0, False),), "indices"),
        ((T(1.0, 0.5, 0.0, False),), "indices"),
        ((T(1.0, 0, math.inf, False),), "Rewards"),
    ],
)
def test_invalid_outcomes_are_rejected(outcomes, message):
    with pytest.raises(ValueError, match=message):
        StochasticModel(((outcomes,),))


def test_invalid_dimensions_discounts_and_policy_probabilities():
    with pytest.raises(ValueError, match="rectangular"):
        StochasticModel(())
    model = StochasticModel((((T(1.0, 0, 0.0, True),),),))
    for gamma in (-0.1, 1.0, math.nan):
        with pytest.raises(ValueError, match="gamma"):
            value_iteration(model, gamma=gamma)
        with pytest.raises(ValueError, match="gamma"):
            evaluate_policy(model, ((1.0,),), gamma=gamma)
    for pi in ((), ((0.5, 0.5),)):
        with pytest.raises(ValueError, match="dimensions"):
            evaluate_policy(model, pi)
    for pi in (((math.nan,),), ((0.5,),), ((-1.0,),)):
        with pytest.raises(ValueError, match="probability"):
            evaluate_policy(model, pi)
    with pytest.raises(ValueError, match="tolerance"):
        value_iteration(model, tolerance=0.0)
