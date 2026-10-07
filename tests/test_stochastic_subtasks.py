import pytest

from rrs.rl.exact import DeterministicModel
from rrs.rl.stochastic_exact import StochasticModel, Transition as T
from rrs.rl.stochastic_subtasks import action_values, evaluate_option, solve_subtask
from rrs.rl.subtasks import Subtask, solve_subtask as deterministic_subtask


def test_stochastic_backup_keeps_outcome_reward_and_arrival_stopping():
    model = StochasticModel(
        (
            ((T(0.5, 1, -1.0, False), T(0.5, 2, 1.0, True)),),
            ((T(1.0, 1, 0.0, False),),),
            ((T(1.0, 2, 0.0, True),),),
        )
    )
    z, terminal = (0.0, 1.0, 0.0), (False, False, True)
    q = action_values(model, (3.0, 0.5, 100.0), z, (True,) * 3, terminal, 0.9)
    assert q[0] == (0.5,)
    result = solve_subtask(model, z, terminal)
    assert result.values == (0.5, 1.0, 0.0)
    evaluated = evaluate_option(model, ((1.0,),) * 3, (1.0,) * 3, z, terminal, 1)
    assert evaluated.values == (0.5, 1.0, 0.0)
    assert evaluated.target_probability == (0.5, 1.0, 0.0)
    assert evaluated.safe_target_probability == (0.0, 1.0, 0.0)
    assert evaluated.penalty_probability == (0.5, 0.0, 0.0)


def test_stopping_on_arrival_requires_a_first_action_even_at_target():
    model = StochasticModel((((T(1.0, 1, -1.0, False),),), ((T(1.0, 1, 0.0, False),),)))
    evaluated = evaluate_option(
        model, ((1.0,),) * 2, (1.0, 1.0), (1.0, 0.0), (False, False), 0
    )
    assert evaluated.values[0] == -1.0
    assert evaluated.target_probability[0] == 0.0
    assert evaluated.penalty_probability[0] == 1.0


def test_fractional_stopping_self_loop_matches_closed_form():
    model = StochasticModel((((T(1.0, 0, 0.0, False),),),))
    result = evaluate_option(model, ((1.0,),), (0.5,), (2.0,), (False,), 0, gamma=0.9)
    assert result.values[0] == pytest.approx(1 / (1 - 0.9 * 0.5), abs=1e-8)
    assert result.target_probability[0] == pytest.approx(1.0, abs=1e-8)
    assert result.safe_target_probability[0] == pytest.approx(1.0, abs=1e-8)
    assert result.penalty_probability[0] == 0.0
    assert result.value_error_bound <= 1e-8 and result.probability_gap <= 1e-8


def test_exact_and_equation_nine_stopping_are_distinct_and_deterministic_case_agrees():
    deterministic = DeterministicModel(
        ((1,), (2,), (2,)), ((0.0,), (1.0,), (0.0,)), ((False,), (True,), (True,))
    )
    model = StochasticModel(
        (
            ((T(1.0, 1, 0.0, False),),),
            ((T(1.0, 2, 1.0, True),),),
            ((T(1.0, 2, 0.0, True),),),
        )
    )
    z, terminal = (0.0, 0.95, 0.0), (False, False, True)
    result = solve_subtask(model, z, terminal, gamma=0.9)
    expected = deterministic_subtask(
        deterministic,
        Subtask(deterministic.rewards, z, (True,) * 3),
        terminal,
        gamma=0.9,
    )
    assert result == expected
    assert result.values == (0.95, 1.0, 0.0)
    assert result.stopping[1]
    assert not z[1] >= result.values[1]  # Eq. 9 would continue at this state.


def test_nonterminating_policy_has_no_false_probability_certificate():
    model = StochasticModel((((T(1.0, 0, 0.0, False),),),))
    with pytest.raises(RuntimeError, match="probability gap"):
        evaluate_option(model, ((1.0,),), (0.0,), (1.0,), (False,), 0, max_iterations=8)
    with pytest.raises(RuntimeError, match="converge"):
        solve_subtask(
            StochasticModel((((T(1.0, 0, -1.0, False),),),)),
            (0.0,),
            (False,),
            can_stop=(False,),
            max_iterations=1,
        )


def test_invalid_stopping_vectors_probabilities_and_terminal_values():
    model = StochasticModel((((T(1.0, 0, 0.0, False),),),))
    with pytest.raises(ValueError, match="Stopping probabilities"):
        evaluate_option(model, ((1.0,),), (1.1,), (1.0,), (False,), 0)
    with pytest.raises(ValueError, match="probability"):
        evaluate_option(model, ((0.5,),), (1.0,), (1.0,), (False,), 0)
    with pytest.raises(ValueError, match="zero at terminals"):
        solve_subtask(model, (1.0,), (True,))
    with pytest.raises(ValueError, match="vectors"):
        solve_subtask(model, (), (False,))
    with pytest.raises(ValueError, match="Target"):
        evaluate_option(model, ((1.0,),), (1.0,), (0.0,), (True,), 0)
