import gzip
import hashlib
import json

import jax
import jax.numpy as jnp
import pytest

from rrs.experiments import four_room_model_inputs as inputs
from rrs.experiments.four_room_model_report import (
    acceptance_checks,
    build_data,
    build_references,
    save_checkpoint,
    save_options,
)
from rrs.experiments.four_room_models import MODEL_NAMES, initialize, training_problem
from rrs.experiments.model_learning import FixedOptions, option_tables


def test_eight_reference_models_match_one_step_and_subtask_identities():
    from rrs.experiments.four_room_option_report import (
        build_references as option_references,
    )

    data, subtasks = build_data(), option_references()
    n = len(data.nonterminal_indices)
    pi = jnp.broadcast_to(jnp.eye(4)[:, :, None], (4, 4, n))
    hallway_pi = jnp.array(
        [
            [
                [float(a == s.greedy_actions[j][0]) for j in data.nonterminal_indices]
                for a in range(4)
            ]
            for s in subtasks.solutions
        ]
    )
    beta = jnp.array(
        [[s.stopping[j] for j in data.nonterminal_indices] for s in subtasks.solutions]
    )
    options = FixedOptions(
        jnp.concatenate((pi, hallway_pi)), jnp.concatenate((jnp.ones((4, n)), beta))
    )
    oracles, targets, identities = build_references(data, options)
    assert all(o.error_bound <= 1e-10 for o in oracles)
    assert all(v["maximum_error"] <= v["tolerance"] for v in identities.values())
    assert targets.reward_weights.shape == (8, 103)
    assert targets.successor_weights.shape == (8, 103, 103)
    goal = data.positions.index((9, 7))
    assert all(
        o.model.rewards[goal] == 0 and not any(o.model.successors[goal])
        for o in oracles
    )


def test_load_pinned_snapshot_preserves_stochastic_policies_and_checkpoint_link(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from rrs.experiments.four_room_option_report import save_snapshot
    from rrs.experiments.four_room_options import SEEDS, initialize as initialize_option

    problem, data = training_problem(), build_data()
    state = jax.vmap(lambda seed: initialize_option(seed, problem))(jnp.array(SEEDS))
    adapter = SimpleNamespace(
        features=data.features,
        terminal=data.terminal,
        positions=data.positions,
        nonterminal=data.nonterminal_indices,
        stopping_values=tuple(
            tuple(float(p == h) for p in data.positions)
            for h in ((6, 2), (3, 6), (7, 9), (10, 6))
        ),
    )
    path = tmp_path / "input.json.gz"
    artifact = save_snapshot(path, state, 1_000_000, SEEDS, adapter)
    monkeypatch.setattr(inputs, "SNAPSHOT_SHA256", artifact["sha256"])
    options, source = inputs.load_options(path, data)
    assert source["seed"] == 7000
    assert source["sha256"] == artifact["sha256"]
    assert options.probability_weights.shape == (8, 4, 103)
    assert jnp.all(options.probability_weights[4:] == 0.25)
    pi, beta = option_tables(options, data)
    assert jnp.all(beta == 1)
    assert jnp.all(pi[:, data.positions.index((9, 7))] == 0.25)
    frozen = save_options(tmp_path / "frozen.json.gz", options, source, data)
    batch = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array([8000, 8001]))
    checkpoint = tmp_path / "models.json.gz"
    first = save_checkpoint(
        checkpoint, batch.learners.model, 0, [8000, 8001], data, frozen
    )
    raw = checkpoint.read_bytes()
    second = save_checkpoint(
        checkpoint, batch.learners.model, 0, [8000, 8001], data, frozen
    )
    assert first == second and checkpoint.read_bytes() == raw
    payload = json.loads(gzip.decompress(raw))
    assert payload["model_names"] == list(MODEL_NAMES)
    assert payload["successor_axes"] == [
        "run",
        "option",
        "output_feature",
        "input_feature",
    ]
    assert (
        payload["frozen_options"]["sha256"]
        == hashlib.sha256((tmp_path / "frozen.json.gz").read_bytes()).hexdigest()
    )
    assert payload["reward_weights"] == batch.learners.model.reward_weights.tolist()
    # A matching byte hash alone does not excuse wrong feature axes/order.
    broken = json.loads(gzip.decompress(path.read_bytes()))
    broken["feature_positions"].reverse()
    path.write_bytes(gzip.compress(json.dumps(broken).encode(), mtime=0))
    monkeypatch.setattr(
        inputs, "SNAPSHOT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="order"):
        inputs.load_options(path, data)


def test_missing_default_regenerates_but_supplied_missing_or_corrupt_never_does(
    tmp_path, monkeypatch
):
    path = tmp_path / "missing.json.gz"
    monkeypatch.setattr(inputs, "SNAPSHOT", path)
    calls = []

    def regenerate(command, check):
        calls.append((command, check))
        path.write_bytes(b"not the canonical snapshot")

    monkeypatch.setattr(inputs.subprocess, "run", regenerate)
    with pytest.raises(FileNotFoundError):
        inputs.ensure_snapshot(tmp_path / "explicit-missing.json.gz")
    assert not calls
    assert inputs.ensure_snapshot(figures_dir=tmp_path) == path
    assert len(calls) == 1 and calls[0][1] is True
    assert "rrs.experiments.four_room_options" in calls[0][0]
    assert inputs.ensure_snapshot() == path
    with pytest.raises(ValueError, match="SHA-256"):
        inputs.load_options(path, build_data())
    assert len(calls) == 1


def test_per_model_gates_cannot_hide_failure_in_averages():
    summary = {
        name: {
            metric: {"mean": [1.0, 0.1]} for metric in ("reward_rmse", "successor_rmse")
        }
        for name in MODEL_NAMES
    }
    assert all(acceptance_checks(summary).values())
    summary["H3"]["reward_rmse"]["mean"][-1] = 0.26
    checks = acceptance_checks(summary)
    assert len(checks) == 32
    assert [key for key, passed in checks.items() if not passed] == [
        "H3/reward_rmse_at_most_0.25"
    ]
    summary["H3"]["reward_rmse"]["mean"] = [0.2, 0.11]
    assert not acceptance_checks(summary)["H3/reward_rmse_at_most_0.50_initial"]
