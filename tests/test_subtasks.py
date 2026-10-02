import jax
import jax.numpy as jnp
import pytest

from rrs.representations.tabular import one_hot
from rrs.rl.exact import DeterministicModel
from rrs.rl.subtasks import Subtask, solve_subtask, stopping_value


def test_representation_uses_observations_and_terminal_zero():
    states = jnp.array([[1, 2], [3, 4], [5, 6]])
    observations = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected = jnp.vstack([jnp.eye(3), jnp.zeros(3)])
    encode = jax.vmap(one_hot, in_axes=(0, None))
    assert jnp.array_equal(encode(observations, states), expected)
    assert jnp.array_equal(jax.jit(encode)(observations, states), expected)


@pytest.mark.parametrize(
    "weights, features, index, bonus, expected",
    [
        ([2.0, 3.0], [1.0, 0.0], 1, 10.0, 2.0),
        ([2.0, 3.0], [0.0, 1.0], 1, 10.0, 10.0),
        ([2.0, -99.0], [0.0, 1.0], 1, 10.0, 10.0),
        ([2.0, 3.0], [0.5, 0.25], 1, 10.0, 3.5),
        ([2.0, 3.0], [0.0, 0.0], 1, 10.0, 0.0),
    ],
)
def test_stopping_value_equation_4(weights, features, index, bonus, expected):
    assert (
        float(stopping_value(jnp.array(weights), jnp.array(features), index, bonus))
        == expected
    )


def test_stopping_bonus_is_not_discounted_on_arrival():
    # 0 -> 1 -> terminal 2. At 1, stopping for 3 beats continuing for gamma*4=2,
    # even though 3 is less than the forced-action value V(1)=4.
    model = DeterministicModel(
        ((1,), (2,), (2,)), ((0.0,), (4.0,), (0.0,)), ((False,), (True,), (True,))
    )
    subtask = Subtask(model.rewards, (0.0, 3.0, 0.0), (False, True, True))
    result = solve_subtask(model, subtask, (False, False, True), gamma=0.5)
    assert result.values == (3.0, 4.0, 0.0)
    assert result.stopping == (False, True, True)
    assert result.bellman_residual == 0.0


def test_stop_on_equality_and_take_an_action_from_a_stopping_state():
    model = DeterministicModel(((0,),), ((0.0,),), ((False,),))
    subtask = Subtask(model.rewards, (0.0,), (True,))
    result = solve_subtask(model, subtask, (False,), gamma=0.5)
    assert result.values == (0.0,)
    assert result.stopping == (True,)
    # A blocked action can attain the current feature after one action.
    result = solve_subtask(
        model, Subtask(model.rewards, (1.0,), (True,)), (False,), gamma=0.5
    )
    assert result.values == (1.0,)


def test_terminal_transition_masks_stopping_value_and_continuation():
    model = DeterministicModel(((1,), (1,)), ((7.0,), (0.0,)), ((True,), (True,)))
    result = solve_subtask(
        model, Subtask(model.rewards, (100.0, 0.0), (True, True)), (False, True)
    )
    assert result.values == (7.0, 0.0)
    assert result.stopping[-1]


def test_fixed_shortest_path_cumulant_and_action_ties():
    model = DeterministicModel(
        ((1, 1), (1, 1)), ((99.0, -99.0), (0.0, 0.0)), ((False, False), (False, False))
    )
    subtask = Subtask(((-1.0, -1.0), (-1.0, -1.0)), (0.0, 0.0), (False, True))
    result = solve_subtask(model, subtask, (False, False), gamma=0.5)
    assert result.values == (-1.0, -1.0)
    assert result.greedy_actions == ((0, 1), (0, 1))
    assert result.stopping == (False, True)


def test_subtask_solver_reports_nonconvergence():
    model = DeterministicModel(((0,),), ((1.0,),), ((False,),))
    subtask = Subtask(model.rewards, (0.0,), (False,))
    with pytest.raises(RuntimeError, match="converge"):
        solve_subtask(model, subtask, (False,), gamma=0.5, max_iterations=1)
