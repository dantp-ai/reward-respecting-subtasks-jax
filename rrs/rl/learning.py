"""Linear off-policy actor-critic, paper Equations 3, 5, 9–11 and UWT.

All updates consume features, not state IDs. See notes/option_learning_contract.md.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class LearningParameters(NamedTuple):
    gamma: float = 0.99
    alpha: float = 0.1
    alpha_actor: float = 0.1
    lambda_: float = 0.0
    lambda_actor: float = 0.0


class ActorCritic(NamedTuple):
    critic: jax.Array
    actor: jax.Array
    critic_trace: jax.Array
    actor_trace: jax.Array


class UpdateInfo(NamedTuple):
    delta: jax.Array
    rho: jax.Array
    beta: jax.Array


def initial_weights(n_features: int, n_actions: int) -> ActorCritic:
    return ActorCritic(
        jnp.zeros(n_features),
        jnp.zeros((n_actions, n_features)),
        jnp.zeros(n_features),
        jnp.zeros((n_actions, n_features)),
    )


def linear_value(weights, features):
    """Equation 3."""
    return jnp.dot(weights, features)


def td_error(c, z, v, v_next, beta, gamma):
    """Equation 5: z and beta belong to the destination state."""
    return c + beta * z + gamma * (1 - beta) * v_next - v


def stopping_rule(z, value, terminal):
    """Equation 9: compare with the undiscounted value; stop on equality."""
    return jnp.asarray(terminal) | (z >= value)


def softmax_policy(actor, features):
    """Equation 11; actor shape is (actions, features)."""
    return jax.nn.softmax(actor @ features)


def log_policy_gradient(probabilities, action, features):
    """Gradient of log pi(action|features) with respect to the actor matrix."""
    return jnp.outer(
        jax.nn.one_hot(action, probabilities.shape[0]) - probabilities, features
    )


def importance_ratio(probabilities, action, behavior_probability):
    """Off-policy ratio; the supplied behavior probability must be positive."""
    return probabilities[action] / behavior_probability


def update_weights_and_traces(
    weights, trace, gradient, delta, rho, beta, alpha, gamma, lambda_
):
    """UWT: accumulate with importance sampling, update, then decay."""
    trace = rho * (trace + gradient)
    weights = weights + alpha * delta * trace
    trace = gamma * lambda_ * (1 - beta) * trace
    return weights, trace


def actor_critic_update(
    weights: ActorCritic,
    features,
    action,
    cumulant,
    next_features,
    next_stopping_value,
    behavior_probability,
    terminal,
    parameters: LearningParameters = LearningParameters(),
) -> tuple[ActorCritic, UpdateInfo]:
    """Equation 10: both UWT calls share quantities computed before either update.

    Environment termination overrides the supplied stopping value and rule.
    Terminal features are zero; the immediate terminal reward is still learned.
    """
    value = linear_value(weights.critic, features)
    next_value = linear_value(weights.critic, next_features)
    z = jnp.where(terminal, 0.0, next_stopping_value)
    beta = stopping_rule(z, next_value, terminal)
    probabilities = softmax_policy(weights.actor, features)
    rho = importance_ratio(probabilities, action, behavior_probability)
    delta = td_error(cumulant, z, value, next_value, beta, parameters.gamma)
    actor_gradient = log_policy_gradient(probabilities, action, features)
    critic, critic_trace = update_weights_and_traces(
        weights.critic,
        weights.critic_trace,
        features,
        delta,
        rho,
        beta,
        parameters.alpha,
        parameters.gamma,
        parameters.lambda_,
    )
    actor, actor_trace = update_weights_and_traces(
        weights.actor,
        weights.actor_trace,
        actor_gradient,
        delta,
        rho,
        beta,
        parameters.alpha_actor,
        parameters.gamma,
        parameters.lambda_actor,
    )
    return ActorCritic(critic, actor, critic_trace, actor_trace), UpdateInfo(
        delta, rho, beta
    )
