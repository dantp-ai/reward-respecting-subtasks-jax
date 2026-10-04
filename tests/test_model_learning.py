import jax.numpy as jnp
import pytest

from rrs.rl.model_learning import (
    ModelLearningParameters,
    ModelLearningState,
    initial_model,
    model_update,
)
from rrs.rl.option_models import LinearExpectationModel


def test_primitive_uses_environment_reward_and_gamma_next_features():
    state, error = model_update(
        initial_model(2),
        jnp.array([0.25, 0.75]),
        2.0,
        jnp.array([1.0, 0.5]),
        0.25,
        0.25,
        1.0,
        False,
        ModelLearningParameters(0.5, 1.0, 1.0, 0.0),
    )
    assert float(error.reward) == 2.0
    assert jnp.array_equal(error.successor, jnp.array([0.5, 0.25]))
    assert jnp.array_equal(state.model.reward_weights, jnp.array([0.5, 1.5]))
    assert jnp.array_equal(
        state.model.successor_weights, jnp.array([[0.125, 0.375], [0.0625, 0.1875]])
    )


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_model_deltas_and_each_trace_use_preupdate_parameters(beta):
    state = ModelLearningState(
        LinearExpectationModel(
            jnp.array([2.0, 4.0]), jnp.array([[0.2, 0.6], [0.4, 0.8]])
        ),
        jnp.array([0.1, 0.2]),
        jnp.array([[0.1, 0.2], [0.3, 0.4]]),
    )
    result, errors = model_update(
        state,
        jnp.array([1.0, 0.0]),
        3.0,
        jnp.array([0.0, 1.0]),
        0.5,
        0.25,
        beta,
        False,
        ModelLearningParameters(0.5, 0.1, 0.2, 0.25),
    )
    if beta == 0:
        expected = (
            [2.66, 4.12],
            [[0.244, 0.608], [0.4, 0.8]],
            [0.275, 0.05],
            [[0.275, 0.05], [0.325, 0.1]],
        )
        assert float(errors.reward) == 3.0
        assert errors.successor.tolist() == pytest.approx([0.1, 0.0], abs=1e-6)
    else:
        expected = (
            [2.22, 4.04],
            [[0.112, 0.584], [0.452, 0.816]],
            [0.0, 0.0],
            [[0.0, 0.0], [0.0, 0.0]],
        )
        assert float(errors.reward) == 1.0
        assert errors.successor.tolist() == pytest.approx([-0.2, 0.1], abs=1e-6)
    actual = (*result.model, result.reward_trace, result.successor_trace)
    for a, b in zip(actual, expected, strict=True):
        assert jnp.allclose(a, jnp.array(b), atol=1e-6)


def test_fractional_stopping_probability():
    state = initial_model(1)._replace(
        model=LinearExpectationModel(jnp.array([0.2]), jnp.array([[0.3]]))
    )
    state, errors = model_update(
        state,
        jnp.ones(1),
        1.0,
        jnp.array([2.0]),
        0.25,
        0.25,
        0.25,
        False,
        ModelLearningParameters(0.5, 1.0, 1.0, 0.0),
    )
    assert float(errors.reward) == pytest.approx(0.95, abs=1e-6)
    assert float(errors.successor[0]) == pytest.approx(0.175, abs=1e-6)
    assert float(state.model.successor_weights[0, 0]) == pytest.approx(0.475, abs=1e-6)


def test_terminal_masks_successor_even_with_nonzero_supplied_next_features():
    state, errors = model_update(
        initial_model(2),
        jnp.array([1.0, 0.0]),
        1.0,
        jnp.array([999.0, 999.0]),
        0.25,
        0.25,
        0.0,
        True,
        ModelLearningParameters(lambda_=0.8),
    )
    assert float(state.model.reward_weights[0]) == pytest.approx(0.1)
    assert jnp.array_equal(errors.successor, jnp.zeros(2))
    assert jnp.array_equal(state.successor_trace, jnp.zeros((2, 2)))
    assert jnp.array_equal(state.reward_trace, jnp.zeros(2))


def test_zero_importance_ratio_preserves_weights_and_clears_traces():
    state = initial_model(2)._replace(
        reward_trace=jnp.ones(2), successor_trace=jnp.ones((2, 2))
    )
    result, _ = model_update(
        state,
        jnp.ones(2),
        3.0,
        jnp.ones(2),
        0.0,
        0.25,
        0.0,
        False,
        ModelLearningParameters(lambda_=0.8),
    )
    for actual, expected in zip(result.model, state.model, strict=True):
        assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(result.reward_trace, jnp.zeros(2))
    assert jnp.array_equal(result.successor_trace, jnp.zeros((2, 2)))


def test_two_step_option_learns_analytic_discounted_model():
    # 0 --reward 2--> 1 --reward 4--> stopping state 2.
    # Alpha=1 and reverse experience make the hand-computed model exact in 2 updates.
    state = initial_model(3)
    params = ModelLearningParameters(0.5, 1.0, 1.0, 0.0)
    eye = jnp.eye(3)
    for source, reward, destination, beta in [(1, 4.0, 2, 1.0), (0, 2.0, 1, 0.0)]:
        state, _ = model_update(
            state, eye[source], reward, eye[destination], 1.0, 1.0, beta, False, params
        )
    reward, successor = state.model.predict(eye[0])
    assert float(reward) == 4.0
    assert jnp.array_equal(successor, jnp.array([0.0, 0.0, 0.25]))
