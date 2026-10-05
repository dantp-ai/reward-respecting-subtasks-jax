import jax.numpy as jnp
import pytest

from rrs.experiments.hallway import build_baselines
from rrs.rl.exact import value_iteration
from rrs.rl.option_models import LinearExpectationModel
from rrs.rl.planning import backed_up_values, planning_step, tabular_plan


def test_tabular_updates_are_asynchronous_and_count_each_option():
    rewards = ((0.0, -1.0), (1.0, 0.0))
    transitions = (((0.0, 0.5), (0.0, 0.0)), ((0.0, 0.0), (0.0, 0.0)))
    forward = tabular_plan(rewards, transitions, (0, 1))
    reverse = tabular_plan(rewards, transitions, (1, 0))
    assert forward.values == (0.0, 1.0)
    assert reverse.values == (0.5, 1.0)
    assert forward.updates == reverse.updates == 2
    assert forward.lookaheads == reverse.lookaheads == 4


def test_linear_backup_and_semigradient_on_mixed_features():
    weights = jnp.array([1.0, 2.0])
    features = jnp.array([0.25, 0.5])
    models = LinearExpectationModel(
        jnp.array([[2.0, 0.0], [-1.0, 4.0]]),
        jnp.array([[[0.0, 0.5], [0.25, 0.0]], [[0.1, 0.0], [0.0, 0.2]]]),
    )
    updated, info = planning_step(weights, features, models, alpha=0.4)
    assert info.action_values.tolist() == pytest.approx([0.875, 1.975], abs=1e-6)
    assert float(info.delta) == pytest.approx(0.725, abs=1e-6)
    assert updated.tolist() == pytest.approx([1.0725, 2.145], abs=1e-6)
    assert info.lookaheads == 2


def test_successor_already_contains_all_discounts_and_terminal_is_zero():
    model = LinearExpectationModel(
        jnp.array([[2.0, 0.0]]), jnp.array([[[0.0, 0.0], [0.25, 0.0]]])
    )
    weights = jnp.array([0.0, 8.0])
    # Reward 2 followed by a two-step successor gamma**2=0.25, so backup=4.
    assert float(backed_up_values(weights, jnp.array([1.0, 0.0]), model)[0]) == 4.0
    updated, info = planning_step(weights, jnp.zeros(2), model)
    assert jnp.array_equal(updated, weights)
    assert jnp.array_equal(info.action_values, jnp.zeros(1))
    assert float(info.delta) == 0.0


def test_primitive_model_backup_is_the_original_bellman_backup():
    data = build_baselines()
    values = tuple(i / 100 for i in range(len(data.positions)))
    weights = jnp.array([values[s] for s in data.nonterminal_indices])
    models = [
        data.models[name].as_linear_model(data.nonterminal_indices)
        for name in ("up", "down", "left", "right")
    ]
    batched = LinearExpectationModel(
        jnp.stack([m.reward_weights for m in models]),
        jnp.stack([m.successor_weights for m in models]),
    )
    for s in data.nonterminal_indices:
        actual = backed_up_values(weights, jnp.array(data.features[s]), batched)
        expected = [
            r + (0 if done else 0.99 * values[ns])
            for ns, r, done in zip(
                data.model.next_states[s],
                data.model.rewards[s],
                data.model.terminated[s],
                strict=True,
            )
        ]
        assert actual.tolist() == pytest.approx(expected, abs=1e-6)


def test_exact_planning_with_all_three_model_sets_reaches_main_task_oracle():
    data = build_baselines()
    expected = value_iteration(data.model).values
    primitives = ("up", "down", "left", "right")
    for names in (
        primitives,
        primitives + ("reward_respecting",),
        primitives + ("shortest_path",),
    ):
        rewards = tuple(
            tuple(data.models[name].rewards[s] for name in names)
            for s in data.nonterminal_indices
        )
        transitions = tuple(
            tuple(data.models[name].successors[s] for name in names)
            for s in data.nonterminal_indices
        )
        order = tuple(range(72)) * 40
        result = tabular_plan(rewards, transitions, order)
        assert result.values == pytest.approx(
            tuple(expected[s] for s in data.nonterminal_indices), abs=1e-12
        )
        assert result.lookaheads == len(order) * len(names)


def test_empty_schedule_and_invalid_tabular_dimensions():
    result = tabular_plan(((2.0,),), (((0.5,),),), (), initial=(3.0,))
    assert result.values == (3.0,)
    assert result.lookaheads == result.updates == 0
    with pytest.raises(ValueError, match="shape"):
        tabular_plan(((1.0,),), (((0.0, 0.0),),), (0,))
