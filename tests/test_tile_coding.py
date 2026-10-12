import jax
import jax.numpy as jnp
import pytest

from rrs.representations.tile_coding import build_params, encode


def test_explicit_tiles_negative_coordinates_order_overlap_and_exclusion():
    params = build_params(((0, 0), (1, 1), (2, 2)))
    assert params.tile_coordinates.tolist() == [
        [0, 0],
        [1, 1],  # unshifted
        [0, -1],
        [0, 0],
        [1, 0],  # column offset
        [-1, 0],
        [0, 0],
        [0, 1],  # row offset
        [-1, -1],
        [0, 0],  # both offsets
    ]
    assert params.feature_tilings.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    expected = jnp.zeros((3, 10))
    for row, indices in enumerate(((0, 2, 5, 8), (0, 3, 6, 9), (1, 4, 7, 9))):
        expected = expected.at[row, jnp.array(indices)].set(0.5)
    actual = jax.vmap(encode, in_axes=(None, 0))(params, params.observations)
    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(jnp.sum(actual**2, axis=1), jnp.ones(3))
    assert float(actual[0] @ actual[1]) == 0.25
    assert float(actual[0] @ actual[2]) == 0.0
    # Excluded coordinates can share occupied tiles; exclusion still forces zero.
    assert jnp.array_equal(encode(params, jnp.array([0, 1])), jnp.zeros(10))
    assert jnp.array_equal(encode(params, jnp.array([99, 99])), jnp.zeros(10))
    assert jnp.array_equal(
        jax.jit(jax.vmap(encode, in_axes=(None, 0)))(params, params.observations),
        expected,
    )
    assert jnp.array_equal(jax.jit(encode)(params, params.observations[0]), expected[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observations": []},
        {"observations": [(0, 0), (0, 0)]},
        {"width": (0, 2)},
        {"width": (2, float("nan"))},
        {"offsets": []},
        {"offsets": [(0, 0), (0, 0)]},
        {"offsets": [(0,)]},
    ],
)
def test_invalid_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_params(**({"observations": [(0, 0), (1, 1)]} | kwargs))
