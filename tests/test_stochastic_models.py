import pytest

from rrs.experiments.hallway import build_baselines
from rrs.rl.exact import DeterministicModel
from rrs.rl.stochastic_models import stochastic_option_model


def test_stochastic_loop_has_analytic_reward_and_discounted_successor():
    # At S, half the actions loop for reward 2, half reach H for reward 4.
    # Initiation at H takes an action, obtaining reward 1 at terminal G.
    model = DeterministicModel(
        ((0, 1), (2, 2), (2, 2)),
        ((2.0, 4.0), (1.0, 1.0), (0.0, 0.0)),
        ((False, False), (True, True), (True, True)),
    )
    result = stochastic_option_model(
        model,
        ((0.5, 0.5),) * 3,
        (0.0, 1.0, 1.0),
        ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        (False, False, True),
        gamma=0.5,
    )
    assert result.model.rewards == pytest.approx((4.0, 1.0, 0.0), abs=1e-10)
    assert result.model.successors[0] == pytest.approx((0.0, 1 / 3), abs=1e-10)
    assert result.model.successors[1:] == ((0.0, 0.0), (0.0, 0.0))
    assert result.error_bound <= 1e-10
    assert result.bellman_residual / 0.5 == result.error_bound


def test_fractional_stopping_has_geometric_duration():
    model = DeterministicModel(((0,),), ((2.0,),), ((False,),))
    result = stochastic_option_model(
        model, ((1.0,),), (0.5,), ((1.0,),), (False,), gamma=0.5
    )
    assert result.model.rewards == pytest.approx((8 / 3,), abs=1e-10)
    assert result.model.successors[0] == pytest.approx((1 / 3,), abs=1e-10)


def test_zero_discount_does_not_change_properness():
    model = DeterministicModel(
        ((1,), (2,), (2,)), ((1.0,), (2.0,), (0.0,)), ((False,), (False,), (False,))
    )
    result = stochastic_option_model(
        model, ((1.0,),) * 3, (0.0, 0.0, 1.0), ((1.0,),) * 3, (False,) * 3, gamma=0.0
    )
    assert result.model.rewards == (1.0, 2.0, 0.0)
    assert result.model.successors == ((0.0,),) * 3


def test_stochastic_oracle_matches_all_existing_deterministic_models():
    data = build_baselines()
    for name, option in data.options.items():
        pi = tuple(
            tuple(float(a == chosen) for a in range(4)) for chosen in option.policy
        )
        result = stochastic_option_model(
            data.model, pi, option.stopping, data.features, data.terminal
        )
        assert result.model.rewards == pytest.approx(
            data.models[name].rewards, abs=1e-10
        )
        for row, expected in zip(
            result.model.successors, data.models[name].successors, strict=True
        ):
            assert row == pytest.approx(expected, abs=1e-10)


def test_improper_option_is_rejected_even_when_discounted_values_exist():
    model = DeterministicModel(
        ((0, 1), (1, 1)), ((0.0, 0.0),) * 2, ((False, False),) * 2
    )
    # State 0 can stop or enter a closed class at 1; not every start is proper.
    with pytest.raises(ValueError, match="never stop"):
        stochastic_option_model(
            model, ((0.5, 0.5),) * 2, (1.0, 0.0), ((1.0,),) * 2, (False,) * 2
        )


def test_nonconvergence_and_invalid_stopping_probabilities():
    model = DeterministicModel(((0,),), ((2.0,),), ((False,),))
    with pytest.raises(RuntimeError, match="converge"):
        stochastic_option_model(
            model, ((1.0,),), (0.5,), ((1.0,),), (False,), max_iterations=1
        )
    for beta in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="Stopping probabilities"):
            stochastic_option_model(model, ((1.0,),), (beta,), ((1.0,),), (False,))
