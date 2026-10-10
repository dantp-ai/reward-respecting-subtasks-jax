import pytest

from rrs.rl.plan_evaluation import evaluate_action_policy
from rrs.rl.stochastic_exact import StochasticModel, Transition


def test_stochastic_cycles_terminal_rewards_and_mixed_outcomes_are_solved():
    # At S, action 0 loops with reward -1 or reaches H with reward 2, equally.
    # H reaches G with reward +1. V(H)=1; V(S)=(0.5 + 0.5*0.9)/(1-0.5*0.9).
    model = StochasticModel(
        (
            (
                (
                    Transition(0.5, 0, -1.0, False),
                    Transition(0.5, 1, 2.0, False),
                ),
                (Transition(1.0, 0, 0.0, False),),
            ),
            (
                (Transition(1.0, 2, 1.0, True),),
                (Transition(1.0, 2, 1.0, True),),
            ),
            (
                (Transition(1.0, 2, 0.0, True),),
                (Transition(1.0, 2, 0.0, True),),
            ),
        )
    )
    result = evaluate_action_policy(
        model, ((1.0, 0.0),) * 3, 0, (False, False, True), gamma=0.9
    )
    assert result.value == pytest.approx(0.95 / 0.55, abs=1e-12)
    assert result.reachable_states == 2
    assert result.error_bound <= 1e-8
    assert evaluate_action_policy(
        model, ((1.0, 0.0),) * 3, 2, (False, False, True)
    ).value == 0


def test_stochastic_policy_keeps_action_probabilities_and_respects_terminal_mask():
    model = StochasticModel(
        (
            (
                (Transition(0.5, 0, -2.0, False), Transition(0.5, 1, 0.0, False)),
                (Transition(1.0, 2, 4.0, True),),
            ),
            ((Transition(1.0, 2, 1.0, False),),) * 2,
            ((Transition(1.0, 2, 0.0, False),),) * 2,
        )
    )
    result = evaluate_action_policy(
        model, ((0.5, 0.5), (1, 0), (1, 0)), 0, (False, False, True), gamma=0.5
    )
    # Immediate mean reward 1.5, with .25 continuation to S and .25 to H.
    # V(H)=1, so V(S) = (1.5 + .125)/(1-.125) = 13/7.
    assert result.value == pytest.approx(13 / 7, abs=1e-12)
    assert result.reachable_states == 2
