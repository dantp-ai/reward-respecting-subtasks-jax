"""Finite coordinate tilings with separate, normalized feature blocks."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


class TileParams(NamedTuple):
    observations: jax.Array
    width: jax.Array
    offsets: jax.Array
    feature_tilings: jax.Array
    tile_coordinates: jax.Array


def build_params(observations, width=(2, 2), offsets=((0, 0), (0, 1), (1, 0), (1, 1))):
    """Host-side geometry only; sort occupied tiles within each ordered block."""
    observations = tuple(tuple(p) for p in observations)
    width, offsets = tuple(width), tuple(tuple(o) for o in offsets)
    if not observations or len(set(observations)) != len(observations):
        raise ValueError("Nonterminal observations must be nonempty and distinct")
    if len(width) != 2 or any(not math.isfinite(w) or w <= 0 for w in width):
        raise ValueError("Tile width must contain two positive finite numbers")
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("Offsets must be nonempty and distinct")
    if any(
        len(p) != 2 or any(not math.isfinite(v) for v in p)
        for p in (*observations, *offsets)
    ):
        raise ValueError("Observations and offsets must be finite coordinate pairs")
    tilings, coordinates = [], []
    for t, offset in enumerate(offsets):
        tiles = sorted(
            {
                tuple(math.floor((p[d] - offset[d]) / width[d]) for d in range(2))
                for p in observations
            }
        )
        tilings.extend([t] * len(tiles))
        coordinates.extend(tiles)
    return TileParams(
        jnp.asarray(observations),
        jnp.asarray(width),
        jnp.asarray(offsets),
        jnp.asarray(tilings),
        jnp.asarray(coordinates),
    )


def encode(params: TileParams, observation):
    """One active tile per tiling; excluded/terminal observations map to zero."""
    valid = jnp.any(jnp.all(params.observations == observation, axis=-1))
    tiles = jnp.floor((observation - params.offsets) / params.width)
    active = jnp.all(tiles[params.feature_tilings] == params.tile_coordinates, axis=-1)
    return (active & valid).astype(jnp.float32) / math.sqrt(params.offsets.shape[0])
