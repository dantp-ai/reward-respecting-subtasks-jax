import pytest

from rrs.rl.exact import DeterministicModel
from rrs.rl.policy_evaluation import evaluate_policy


def test_stochastic_self_loop_and_forced_action_at_stopping_source():
    # S: equally likely to stay or reach H. H: goes to goal with reward 1.
    model = DeterministicModel(
        ((0, 1), (2, 2), (2, 2)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0)),
        ((False, False), (True, True), (True, True)),
    )
    result = evaluate_policy(
        model,
        ((0.5, 0.5),) * 3,
        (False, True, True),
        (0.0, 1.0, 0.0),
        (False, False, True),
        target_state=1,
        gamma=0.9,
    )
    assert result.values == pytest.approx((0.5 / (1 - 0.45), 1.0, 0.0), abs=1e-8)
    assert result.safe_success == pytest.approx((1.0, 0.0, 0.0), abs=1e-8)
    assert result.value_error_bound <= 1e-8
    assert result.probability_gap <= 1e-8


def test_penalty_invalidates_success_even_when_hallway_is_reached():
    # First action reaches H safely half the time; otherwise pays -1 before H.
    model = DeterministicModel(
        ((1, 2), (1, 1), (1, 1)),
        ((0.0, -1.0), (0.0, 0.0), (0.0, 0.0)),
        ((False, False),) * 3,
    )
    result = evaluate_policy(
        model,
        ((0.5, 0.5),) * 3,
        (False, True, False),
        (0.0, 1.0, 0.0),
        (False,) * 3,
        target_state=1,
        gamma=0.9,
    )
    assert result.values == pytest.approx((0.45, 1.0, 1.0), abs=1e-8)
    assert result.safe_success == pytest.approx((0.5, 1.0, 1.0), abs=1e-8)


def test_stopping_elsewhere_is_failure_and_bonus_is_not_discounted():
    model = DeterministicModel(
        ((1, 2), (1, 1), (2, 2)), ((0.0, 0.0),) * 3, ((False, False),) * 3
    )
    result = evaluate_policy(
        model,
        ((0.25, 0.75),) * 3,
        (True,) * 3,
        (0.0, 2.0, 0.0),
        (False,) * 3,
        target_state=1,
        gamma=0.5,
    )
    assert result.values[0] == 0.5
    assert result.safe_success[0] == 0.25


def test_nontermination_is_reported_instead_of_truncated_success():
    model = DeterministicModel(((0,),), ((0.0,),), ((False,),))
    with pytest.raises(RuntimeError, match="converge"):
        evaluate_policy(
            model, ((1.0,),), (False,), (0.0,), (False,), 0, max_iterations=10
        )


@pytest.mark.parametrize("probabilities", [((0.8,),), ((-1.0,),), ((float("nan"),),)])
def test_invalid_policy_is_rejected(probabilities):
    model = DeterministicModel(((0,),), ((0.0,),), ((False,),))
    with pytest.raises(ValueError, match="probabilit"):
        evaluate_policy(model, probabilities, (True,), (1.0,), (False,), 0)
