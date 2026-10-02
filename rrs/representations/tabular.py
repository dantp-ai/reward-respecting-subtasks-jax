"""One-hot features for an explicitly ordered list of nonterminal observations."""

import jax
import jax.numpy as jnp


def one_hot(observation: jax.Array, nonterminal_observations: jax.Array) -> jax.Array:
    """Match a legal coordinate observation; the omitted terminal maps to zeros.

    The supplied observations must be distinct and have shape (features, coordinates).
    No state index or environment layout is encoded here.
    """
    return jnp.all(nonterminal_observations == observation, axis=-1).astype(jnp.float32)
