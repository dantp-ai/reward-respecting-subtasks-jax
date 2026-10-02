import jax
import jax.numpy as jnp
import pytest

from rrs.rl.exact import DeterministicModel
from rrs.rl.option_models import DeterministicOption, exact_option_model


def test_two_step_model_and_source_stopping_does_not_prevent_initiation():
    model = DeterministicModel(
        ((1,), (2,), (2,)), ((2.0,), (4.0,), (8.0,)), ((False,), (False,), (False,))
    )
    option = DeterministicOption((0, 0, 0), (True, False, True))
    features = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    result = exact_option_model(model, option, features, (False,) * 3, gamma=0.5)
    assert result.rewards == (4.0, 4.0, 8.0)
    assert result.successors == ((0.0, 0.0, 0.25), (0.0, 0.0, 0.5), (0.0, 0.0, 0.5))


def test_primitive_model_and_environment_termination_override():
    model = DeterministicModel(
        ((1,), (2,), (2,)), ((2.0,), (4.0,), (0.0,)), ((False,), (True,), (True,))
    )
    features = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0))
    primitive = DeterministicOption((0, 0, 0), (True,) * 3)
    result = exact_option_model(
        model, primitive, features, (False, False, True), gamma=0.5
    )
    assert result.rewards == (2.0, 4.0, 0.0)
    assert result.successors == ((0.0, 0.5), (0.0, 0.0), (0.0, 0.0))
    never_stops = DeterministicOption((0, 0, 0), (False,) * 3)
    result = exact_option_model(
        model, never_stops, features, (False, False, True), gamma=0.5
    )
    assert result.rewards == (4.0, 4.0, 0.0)
    assert result.successors == ((0.0, 0.0),) * 3


def test_nonterminating_option_is_rejected_even_if_rewards_are_zero():
    model = DeterministicModel(((0,),), ((0.0,),), ((False,),))
    option = DeterministicOption((0,), (False,))
    with pytest.raises(ValueError, match="cycle"):
        exact_option_model(model, option, ((1.0,),), (False,))


def test_linear_model_uses_feature_order_and_accepts_feature_vectors():
    # Terminal row lies between two feature-bearing rows.
    model = DeterministicModel(
        ((2,), (1,), (0,)), ((2.0,), (0.0,), (4.0,)), ((False,), (True,), (False,))
    )
    features = ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
    option = DeterministicOption((0, 0, 0), (True,) * 3)
    exact = exact_option_model(model, option, features, (False, True, False), gamma=0.5)
    linear = exact.as_linear_model((0, 2))
    assert jnp.array_equal(linear.reward_weights, jnp.array([2.0, 4.0]))
    reward, successors = jax.jit(linear.predict)(jnp.array([0.25, 0.75]))
    assert float(reward) == 3.5
    assert jnp.array_equal(successors, jnp.array([0.375, 0.125]))
