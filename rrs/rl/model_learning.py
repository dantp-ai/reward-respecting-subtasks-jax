"""Feature-facing off-policy expectation model learning, Equations 12–17.

The successor target follows gamma**K in Equation 15; see the discount
discrepancy documented in notes/model_learning_contract.md.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rrs.rl.learning import td_error, update_weights_and_traces
from rrs.rl.option_models import LinearExpectationModel


class ModelLearningParameters(NamedTuple):
    gamma: float = 0.99
    alpha_r: float = 0.1
    alpha_p: float = 0.1
    lambda_: float = 0.0


class ModelLearningState(NamedTuple):
    model: LinearExpectationModel
    reward_trace: jax.Array
    successor_trace: jax.Array


class ModelErrors(NamedTuple):
    reward: jax.Array
    successor: jax.Array


def initial_model(n_features: int) -> ModelLearningState:
    return ModelLearningState(
        LinearExpectationModel(
            jnp.zeros(n_features), jnp.zeros((n_features, n_features))
        ),
        jnp.zeros(n_features),
        jnp.zeros((n_features, n_features)),
    )


def model_update(
    state: ModelLearningState,
    features,
    reward,
    next_features,
    target_probability,
    behavior_probability,
    beta,
    terminal,
    parameters: ModelLearningParameters = ModelLearningParameters(),
) -> tuple[ModelLearningState, ModelErrors]:
    """Update one option model from one behavior transition, with pre-update TD errors.

    beta is the destination's stopping probability, in [0,1]. Supplied behavior
    probability must be positive. Reward is actual environment reward, without
    an artificial stopping bonus. A zero target probability yields rho=0.
    """
    beta = jnp.where(terminal, 1.0, beta)
    next_features = jnp.where(terminal, jnp.zeros_like(next_features), next_features)
    current_reward, current_successor = state.model.predict(features)
    next_reward, next_successor = state.model.predict(next_features)
    rho = target_probability / behavior_probability
    delta_r = td_error(reward, 0.0, current_reward, next_reward, beta, parameters.gamma)
    # Eq. 15 requires gamma even when stopping after this single action.
    delta_n = td_error(
        0.0,
        parameters.gamma * next_features,
        current_successor,
        next_successor,
        beta,
        parameters.gamma,
    )
    reward_weights, reward_trace = update_weights_and_traces(
        state.model.reward_weights,
        state.reward_trace,
        features,
        delta_r,
        rho,
        beta,
        parameters.alpha_r,
        parameters.gamma,
        parameters.lambda_,
    )
    successor_weights, successor_trace = update_weights_and_traces(
        state.model.successor_weights,
        state.successor_trace,
        jnp.broadcast_to(features, state.model.successor_weights.shape),
        delta_n[:, None],
        rho,
        beta,
        parameters.alpha_p,
        parameters.gamma,
        parameters.lambda_,
    )
    return ModelLearningState(
        LinearExpectationModel(reward_weights, successor_weights),
        reward_trace,
        successor_trace,
    ), ModelErrors(delta_r, delta_n)
