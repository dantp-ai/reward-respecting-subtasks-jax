import math

import jax
import jax.numpy as jnp
import pytest

from rrs.rl.learning import (
    ActorCritic,
    LearningParameters,
    actor_critic_update,
    importance_ratio,
    initial_weights,
    linear_value,
    log_policy_gradient,
    softmax_policy,
    stopping_rule,
    td_error,
    update_weights_and_traces,
)


def test_linear_value_accepts_nonbinary_features():
    assert float(linear_value(jnp.array([2.0, -3.0]), jnp.array([0.5, 2.0]))) == -5


@pytest.mark.parametrize("beta, expected", [(0.0, 3.7), (1.0, 6.0), (0.5, 4.85)])
def test_td_error_equation_5(beta, expected):
    assert float(td_error(2.0, 5.0, 1.0, 3.0, beta, 0.9)) == pytest.approx(
        expected, abs=1e-6
    )


def test_equation_9_stops_on_equality_and_terminal_but_does_not_discount():
    assert bool(stopping_rule(3.0, 3.0, False))
    assert not bool(stopping_rule(3.0, 4.0, False))  # even if gamma*4 < 3
    assert bool(stopping_rule(0.0, 100.0, True))


def test_softmax_stability_probabilities_and_analytic_gradient():
    x = jnp.array([0.5, 2.0])
    uniform = softmax_policy(jnp.zeros((4, 2)), x)
    assert jnp.array_equal(uniform, jnp.full(4, 0.25))
    actor = jnp.array([[2000.0, 1.0], [2000.0, -1.0]])
    probabilities = softmax_policy(actor, x)
    assert jnp.isfinite(probabilities).all()
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert float(probabilities[0]) == pytest.approx(1 / (1 + math.exp(-4)))
    gradient = log_policy_gradient(probabilities, 1, x)
    autodiff = jax.grad(lambda w: jnp.log(softmax_policy(w, x)[1]))(actor)
    assert jnp.allclose(gradient, autodiff, atol=1e-6)
    assert jnp.allclose(gradient.sum(axis=0), 0, atol=1e-6)
    assert float(importance_ratio(probabilities, 1, 0.25)) == pytest.approx(
        4 / (1 + math.exp(4)), abs=1e-6
    )


@pytest.mark.parametrize("beta, expected_trace", [(0.0, 0.63), (1.0, 0.0)])
def test_uwt_importance_multiplies_old_trace_and_gradient_before_update(
    beta, expected_trace
):
    # e = 2*(0.3+0.4)=1.4, w = 1 + 0.1*3*1.4=1.42, decay=.9*.5.
    w, e = update_weights_and_traces(
        jnp.array([1.0]),
        jnp.array([0.3]),
        jnp.array([0.4]),
        delta=3.0,
        rho=2.0,
        beta=beta,
        alpha=0.1,
        gamma=0.9,
        lambda_=0.5,
    )
    assert float(w[0]) == pytest.approx(1.42, abs=1e-6)
    assert float(e[0]) == pytest.approx(expected_trace, abs=1e-6)


def test_actor_and_critic_share_preupdate_delta_ratio_and_stopping():
    weights = ActorCritic(
        jnp.array([0.5]),
        jnp.array([[math.log(3)], [0.0]]),
        jnp.array([0.2]),
        jnp.array([[0.1], [-0.1]]),
    )
    parameters = LearningParameters(0.9, 0.1, 0.2, 0.5, 0.25)
    updated, info = actor_critic_update(
        weights, jnp.ones(1), 0, 1.0, jnp.ones(1), 0.0, 0.5, False, parameters
    )
    assert float(info.delta) == pytest.approx(0.95, abs=1e-6)
    assert float(info.rho) == pytest.approx(1.5, abs=1e-6)
    assert not bool(info.beta)
    assert float(updated.critic[0]) == pytest.approx(0.671, abs=1e-6)
    assert float(updated.critic_trace[0]) == pytest.approx(0.81, abs=1e-6)
    assert updated.actor[:, 0].tolist() == pytest.approx(
        [math.log(3) + 0.09975, -0.09975], abs=1e-6
    )
    assert updated.actor_trace[:, 0].tolist() == pytest.approx(
        [0.118125, -0.118125], abs=1e-6
    )


def test_stopping_uses_destination_and_terminal_masks_bonus_and_bootstrap():
    weights = initial_weights(2, 2)._replace(critic=jnp.array([0.0, 2.0]))
    x, next_x = jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    _, info = actor_critic_update(weights, x, 0, 1.0, next_x, 0.0, 0.5, False)
    assert not bool(info.beta)  # source value is zero, destination value is two
    assert float(info.delta) == pytest.approx(2.98, abs=1e-6)
    updated, info = actor_critic_update(
        weights,
        x,
        0,
        1.0,
        next_x,
        999.0,
        0.5,
        True,
        LearningParameters(lambda_=0.8, lambda_actor=0.8),
    )
    assert bool(info.beta)
    assert float(info.delta) == 1.0
    assert jnp.array_equal(updated.critic_trace, jnp.zeros(2))
    assert jnp.array_equal(updated.actor_trace, jnp.zeros((2, 2)))


def test_short_sequence_matches_independent_scalar_updates():
    # Nonbinary features and both nonzero traces exercise more than tabular TD(0).
    weights = initial_weights(2, 2)
    critic, actor = [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]
    ec, ea = [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]
    parameters = LearningParameters(0.9, 0.1, 0.2, 0.4, 0.6)
    transitions = [
        ([1.0, 0.0], 0, -1.0, [0.0, 1.0], 0.0, False),
        ([0.0, 1.0], 1, 0.0, [0.5, 0.5], 1.0, False),
        ([0.5, 0.5], 1, 0.5, [1.0, 0.0], 0.0, False),
        ([1.0, 0.0], 0, 1.0, [0.0, 0.0], 0.0, True),
    ]
    for x, a, r, nx, z, done in transitions:
        preferences = [sum(w * f for w, f in zip(row, x)) for row in actor]
        exps = [math.exp(p) for p in preferences]
        pi = [p / sum(exps) for p in exps]
        v = sum(w * f for w, f in zip(critic, x))
        vn = sum(w * f for w, f in zip(critic, nx))
        stop = done or z >= vn
        delta = r + (z if stop else 0.9 * vn) - v
        rho = pi[a] / 0.5
        for j in range(2):
            ec[j] = rho * (ec[j] + x[j])
            critic[j] += 0.1 * delta * ec[j]
            ec[j] *= 0.9 * 0.4 * (not stop)
            for k in range(2):
                ea[k][j] = rho * (ea[k][j] + ((k == a) - pi[k]) * x[j])
                actor[k][j] += 0.2 * delta * ea[k][j]
                ea[k][j] *= 0.9 * 0.6 * (not stop)
        weights, _ = actor_critic_update(
            weights, jnp.array(x), a, r, jnp.array(nx), z, 0.5, done, parameters
        )
        for actual, expected in zip(weights, (critic, actor, ec, ea), strict=True):
            assert jnp.allclose(actual, jnp.array(expected), atol=1e-6)
