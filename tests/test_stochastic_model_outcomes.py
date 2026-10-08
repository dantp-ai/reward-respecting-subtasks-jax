import pytest

from rrs.rl.stochastic_exact import StochasticModel, Transition as T
from rrs.rl.stochastic_models import stochastic_option_model


def test_mixed_outcomes_and_action_probabilities_use_arrival_stopping():
    # At S: pi=(1/4,3/4); action 0 splits equally between a -2 loop and +4 at H.
    # Action 1 reaches terminal G for +1. beta(S)=1/2, beta(H)=1, gamma=1/2.
    # R(S) = (-1/4+1/2+3/4) + R(S)/32 = 32/31.
    # N(S) = (1/32,1/16) + N(S)/32 = (1/31,2/31).
    model = StochasticModel(
        (
            ((T(0.5, 0, -2, False), T(0.5, 1, 4, False)), (T(1, 2, 1, True),)),
            ((T(1, 2, 3, True),),) * 2,
            ((T(1, 2, 0, True),),) * 2,
        )
    )
    result = stochastic_option_model(
        model,
        ((0.25, 0.75),) * 3,
        (0.5, 1, 1),
        ((1, 0), (0, 1), (0, 0)),
        (False, False, True),
        gamma=0.5,
    )
    assert result.model.rewards == pytest.approx((32 / 31, 3, 0), abs=1e-10)
    assert result.model.successors[0] == pytest.approx((1 / 31, 2 / 31), abs=1e-10)
    assert result.model.successors[1:] == ((0, 0), (0, 0))
    assert result.error_bound <= 1e-10
    linear = result.model.as_linear_model((0, 1))
    assert linear.successor_weights[:, 0].tolist() == pytest.approx((1 / 31, 2 / 31))


def test_duplicate_wall_outcomes_equal_merged_primitive_expectation():
    duplicated = (T(0.2, 0, -1, False), T(0.3, 0, -1, False), T(0.5, 1, 1, True))
    merged = (T(0.5, 0, -1, False), T(0.5, 1, 1, True))
    results = [
        stochastic_option_model(
            StochasticModel(((row,), ((T(1, 1, 0, True),),))),
            ((1,),) * 2,
            (1, 1),
            ((1,), (0,)),
            (False, True),
            gamma=0.8,
        )
        for row in (duplicated, merged)
    ]
    assert results[0].model == results[1].model
    assert results[0].model.rewards == (0, 0)
    assert results[0].model.successors == ((0.4,), (0,))


def test_zero_probability_exit_does_not_make_closed_class_proper():
    model = StochasticModel(
        (((T(1, 0, 0, False), T(0, 1, 1, True)),), ((T(1, 1, 0, True),),))
    )
    with pytest.raises(ValueError, match="never stop"):
        stochastic_option_model(model, ((1,),) * 2, (0, 1), ((1,), (0,)), (False, True))


def test_terminal_state_mask_overrides_outcome_flag_and_stopping():
    model = StochasticModel((((T(1, 1, 7, False),),), ((T(1, 1, 0, False),),)))
    result = stochastic_option_model(
        model, ((1,),) * 2, (0, 0), ((1,), (0,)), (False, True)
    )
    assert result.model.rewards == (7, 0)
    assert result.model.successors == ((0,), (0,))
