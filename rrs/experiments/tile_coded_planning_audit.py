"""Post-run audit of exact primitive planning, independent of feature-model weights.

Replay predetermined planning seed 12000 in double precision directly in state
space: V <- V + (X X.T)[:,s] * (max_o[R_o(s)+P_o(s) V] - V[s]).
This verifies the observed dynamics without JAX planning or pseudoinverse model
fitting. It changes no experimental hyperparameters and tests no alternative.
"""

import gzip
import json
from pathlib import Path

import numpy as np

from rrs.experiments.hallway import build_baselines
from rrs.experiments.planning import sample_states
from rrs.experiments.tile_coded_model_report import save_json
from rrs.experiments.tile_coded_models import features
from rrs.experiments.tile_coded_planning import BUDGETS


def state_space_replay(rewards, successors, gram, order, query_budgets):
    """Pure NumPy arithmetic on state values; no feature-model coefficients."""
    v = np.zeros(gram.shape[0], dtype=float)
    history = [v.copy()]
    n_models = len(rewards)
    for update, s in enumerate(order, 1):
        delta = max(rewards[:, s] + successors[:, s] @ v) - v[s]
        v = v + gram[:, s] * delta
        if update * n_models in query_budgets:
            history.append(v.copy())
    if len(history) != len(query_budgets):
        raise ValueError("Replay order must cover every requested query budget")
    return np.array(history)


def audit(directory, model_directory):
    source = json.loads((model_directory / "learning.json").read_text())
    data = build_baselines()
    ids = list(data.nonterminal_indices)
    start = ids.index(data.positions.index((3, 1)))
    # Independent exact state-space tables were verified by the main experiment.
    r = np.asarray(source["references"]["one_hot"]["reward"])[:4, ids]
    p = np.asarray(source["references"]["one_hot"]["successor"])[:4, ids]
    order = sample_states(12000, len(ids), 5000).tolist()
    cases = {}
    for name in ("one_hot", "tile_coding"):
        x = np.asarray(features(data, name), dtype=float)[ids]
        values = state_space_replay(r, p, x @ x.T, order, BUDGETS)
        with gzip.open(
            directory / f"{name}_exact_primitives_weights.json.gz", "rt"
        ) as stream:
            saved = json.load(stream)
        if saved["planning_seeds"][0] != 12000 or saved["budgets"] != list(BUDGETS):
            raise ValueError("Saved planning run does not match the audit")
        original = np.asarray(saved["weights"][0]) @ x.T
        difference = np.max(np.abs(values - original), axis=1)
        scale = np.maximum(1, np.max(np.abs(values), axis=1))
        cases[name] = {
            "double_start_values": values[:, start].tolist(),
            "saved_start_values": original[:, start].tolist(),
            "double_state_values": values.tolist(),
            "max_absolute_discrepancy": difference.tolist(),
            "max_discrepancy_over_state_scale": (difference / scale).tolist(),
        }
    result = {
        "planning_seed": 12000,
        "budgets": BUDGETS,
        "cases": cases,
        "scope": "Post-run descriptive numerical audit, not a new tuning experiment or predeclared acceptance test",
    }
    save_json(directory / "state_space_audit.json", result)
    print(
        json.dumps(
            {name: c["double_start_values"] for name, c in cases.items()}, indent=2
        )
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tile_coded_planning")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/tile_coded_models")
    )
    args = parser.parse_args()
    audit(args.output_dir, args.model_dir)


if __name__ == "__main__":
    main()
