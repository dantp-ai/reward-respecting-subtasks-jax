"""Reproducible report and compressed planning traces for Milestone 9."""

import gzip
import hashlib
import json
import platform
from importlib.metadata import version

import jax

from rrs.experiments.two_room import _revision


def save_trace(path, history, case, checkpoints, seeds, provenance):
    payload = {
        "schema_version": 1,
        "case": case,
        "checkpoints": list(checkpoints),
        "planning_seeds": list(seeds),
        "model_seeds": provenance["model_seeds"] if case.startswith("learned_") else None,
        "weight_axes": ["run", "lookahead_checkpoint", "feature"],
        "weights": history.tolist(),
    }
    encoded = gzip.compress(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def build_report(data, inputs, cases, references, optimal_values, provenance):
    revision, dirty = _revision()
    start = data.positions.index((4, 1))
    return {
        "schema_version": 1,
        "milestone": "09-four-room-planning",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/17",
        "canonical_protocol": len(cases) == 15,
        "planning_seeds": list(range(3000, 3030)),
        "model_seeds": list(inputs.provenance["model_seeds"]),
        "gamma": 0.99,
        "alpha": 1.0,
        "initial_weights": 0.0,
        "lookahead_budget": 1_600_000,
        "checkpoints": list(range(0, 1_600_001, 1_600)),
        "evaluation_budgets": [0, 200_000, 400_000, 600_000, 1_600_000],
        "key_schedule": "key(seed); split(next, state); randint(state, (), 0, 103)",
        "shared_state_prefix": True,
        "model_order": "UP, DOWN, LEFT, RIGHT, H1, H2, H3, H4",
        "tie_rule": "first maximum",
        "execution": "reselect a model at every primitive step; sample one action from its policy",
        "evaluation": "Python-double discounted solve in independent stochastic environment; no truncation",
        "evaluation_tolerance": 1e-8,
        "precision": "float32 planning; Python-double references and policy evaluation",
        "start_state": list(data.positions[start]),
        "optimal_start_value": float(optimal_values[start]),
        "optimal_values": list(optimal_values),
        "feature_positions": [list(data.positions[i]) for i in data.nonterminal_indices],
        "policy_positions": [list(p) for p in data.positions],
        "reference_provenance": {
            "optimal_options": references[2].provenance,
            "frozen_options": references[3].provenance,
        },
        "model_provenance": inputs.provenance,
        "artifacts": provenance,
        "cases": cases,
        "acceptance": {
            "all_cases_finite": all(c["finite"] for c in cases.values()),
            "terminal_values_zero": all(c["terminal_zero"] for c in cases.values()),
            "all_queries_match_budget": all(
                c["planning_lookaheads_per_run"] == 1_600_000 for c in cases.values()
            ),
            "all_policy_error_bounds_within_1e-8": all(
                max(max(run["evaluation_error_bounds"]) for run in c["runs"])
                <= 1e-8
                for c in cases.values()
            ),
            "exact_primitive_final_values_within_1e-5": max(
                r["final_max_value_error"] for r in cases["exact_primitive/actions"]["runs"]
            ) <= 1e-5,
            "exact_primitive_final_returns_within_1e-5": max(
                r["actual_return"][-1] for r in cases["exact_primitive/actions"]["runs"]
            ) - float(optimal_values[start]) <= 1e-5
            and float(optimal_values[start]) - min(
                r["actual_return"][-1] for r in cases["exact_primitive/actions"]["runs"]
            ) <= 1e-5,
        },
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("jax", "jaxlib", "matplotlib")},
    }
