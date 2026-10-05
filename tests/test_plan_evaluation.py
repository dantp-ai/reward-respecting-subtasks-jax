import jax.numpy as jnp
import pytest

from rrs.rl.exact import DeterministicModel
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.plan_evaluation import evaluate_action_policy, induced_policy


def test_induced_policy_reselects_after_each_primitive_action():
    # The option wins at S, but the primitive wins at T. Committing to the
    # option would repeatedly take action 1 at T and incur negative rewards.
    model = DeterministicModel(
        ((0, 1), (2, 1), (2, 2)),
        ((0.0, 0.0), (1.0, -1.0), (0.0, 0.0)),
        ((False, False), (True, False), (True, True)),
    )
    models = LinearExpectationModel(
        jnp.array([[0.0, 2.0], [1.0, 0.0]]), jnp.zeros((2, 2, 2))
    )
    option_policies = jnp.array([[[1.0, 0.0]] * 3, [[0.0, 1.0]] * 3])
    pi, choices = induced_policy(
        jnp.zeros(2),
        models,
        option_policies,
        jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    assert choices.tolist() == [1, 0, 0]
    assert pi.tolist() == [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    result = evaluate_action_policy(
        model, pi.tolist(), 0, (False, False, True), gamma=0.9
    )
    assert result.value == pytest.approx(0.9, abs=1e-12)
    assert result.error_bound <= 1e-8


def test_selected_option_retains_its_stochastic_action_probabilities():
    models = LinearExpectationModel(jnp.array([[0.0], [1.0]]), jnp.zeros((2, 1, 1)))
    pi, choices = induced_policy(
        jnp.zeros(1),
        models,
        jnp.array([[[1.0, 0.0]], [[0.25, 0.75]]]),
        jnp.ones((1, 1)),
    )
    assert choices.tolist() == [1]
    assert pi.tolist() == [[0.25, 0.75]]


def test_stochastic_loop_value_is_solved_without_a_horizon():
    model = DeterministicModel(
        ((0, 1), (1, 1)), ((0.0, 1.0), (0.0, 0.0)), ((False, True), (True, True))
    )
    result = evaluate_action_policy(
        model, ((0.5, 0.5),) * 2, 0, (False, True), gamma=0.9
    )
    assert result.value == pytest.approx(0.5 / (1 - 0.45), abs=1e-12)
    assert result.reachable_states == 1


def test_nonterminating_negative_cycle_has_finite_discounted_return():
    model = DeterministicModel(((0,), (1,)), ((-1.0,), (0.0,)), ((False,), (True,)))
    result = evaluate_action_policy(model, ((1.0,),) * 2, 0, (False, True), gamma=0.5)
    assert result.value == -2.0
    assert result.error_bound == 0.0
    assert evaluate_action_policy(model, ((1.0,),) * 2, 1, (False, True)).value == 0.0


def test_invalid_policy_or_discount_is_rejected():
    model = DeterministicModel(((0,),), ((0.0,),), ((False,),))
    with pytest.raises(ValueError, match="probabilit"):
        evaluate_action_policy(model, ((0.9,),), 0, (False,))
    with pytest.raises(ValueError, match="gamma"):
        evaluate_action_policy(model, ((1.0,),), 0, (False,), gamma=1)
