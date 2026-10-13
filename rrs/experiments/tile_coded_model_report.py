"""Independent references, state-space errors and reproducible Milestone 11 runs."""

import gzip
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rrs.experiments.hallway import build_baselines
from rrs.experiments.learning_report import mean_and_se, reference_model
from rrs.experiments.model_audit import sample_model
from rrs.experiments.tile_coded_models import (
    CHECKPOINT,
    NAMES,
    PARAMETERS,
    SEEDS,
    STEPS,
    Problem,
    features,
    freeze_options,
    initialize,
    train_block,
)
from rrs.experiments.two_room import _revision
from rrs.rl.policy_evaluation import evaluate_policy
from rrs.rl.stochastic_models import stochastic_option_model

METRICS = ("reward_rmse", "native_successor_rmse", "common_successor_rmse")


def save_json(path, payload, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=None if compressed else 2, allow_nan=False) + "\n"
    ).encode()
    if compressed:
        encoded = gzip.compress(encoded, mtime=0)
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def decoder(feature_rows, indices):
    x = np.asarray(feature_rows, dtype=float)[list(indices)]
    d = np.linalg.pinv(x)
    residual = float(np.max(np.abs(x @ d - np.eye(len(indices)))))
    if residual > 1e-10:
        raise ValueError("Common-state decoding requires full row rank")
    return d, {
        "rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)),
        "identity_residual": residual,
    }


def build_references(data, pi, beta, tables):
    model = reference_model(data.positions)
    references = {}
    for name, x in tables.items():
        print(f"Building independent {name} model references", flush=True)
        refs = [
            stochastic_option_model(model, p, b, x.tolist(), data.terminal)
            for p, b in zip(pi.tolist(), beta.tolist(), strict=True)
        ]
        references[name] = {
            "reward": [r.model.rewards for r in refs],
            "successor": [r.model.successors for r in refs],
            "bounds": [r.error_bound for r in refs],
            "iterations": [r.iterations for r in refs],
        }
    x = np.asarray(tables["tile_coding"])[list(data.nonterminal_indices)]
    projected = np.asarray(references["one_hot"]["successor"]) @ x
    discrepancy = float(
        np.max(np.abs(projected - references["tile_coding"]["successor"]))
    )
    reward_discrepancy = float(
        np.max(
            np.abs(
                np.asarray(references["one_hot"]["reward"])
                - references["tile_coding"]["reward"]
            )
        )
    )
    if max(discrepancy, reward_discrepancy) > 1e-8:
        raise ValueError("Independent feature-space references disagree")
    return references, {
        "successor_max_error": discrepancy,
        "reward_max_error": reward_discrepancy,
    }


def rollout_audits(data, pi, beta, tables, references):
    combined = np.concatenate(list(tables.values()), axis=1).tolist()
    audits = []
    for option in (5, 6):
        for k, start in enumerate(((3, 1), (1, 13), (3, 7), (3, 3))):
            sampled = sample_model(
                data.positions,
                pi[option].tolist(),
                beta[option].tolist(),
                start,
                21260 + 4 * (option - 5) + k,
                feature_rows=combined,
            )
            s, offset = data.positions.index(start), 0
            target_r = references["one_hot"]["reward"][option][s]
            sampled.update(
                {
                    "option": NAMES[option],
                    "reward_exact": target_r,
                    "reward_passed": abs(sampled["reward_mean"] - target_r)
                    <= 0.01 + 6 * sampled["reward_se"],
                    "representations": {},
                }
            )
            for name, x in tables.items():
                width = x.shape[1]
                target = np.asarray(references[name]["successor"][option][s])
                means = np.asarray(sampled["successor_mean"][offset : offset + width])
                se = np.asarray(sampled["successor_se"][offset : offset + width])
                sampled["representations"][name] = {
                    "successor_exact": target.tolist(),
                    "successor_passed": bool(
                        np.all(np.abs(means - target) <= 0.01 + 6 * se)
                    ),
                    "slice": [offset, offset + width],
                }
                offset += width
            audits.append(sampled)
    return audits


@jax.jit
def model_errors(models, x, reward_target, successor_target, decode, common_target):
    """x contains nonterminal source rows; predictions, not parameter differences."""
    r = models.reward_weights @ x.T
    n = jnp.swapaxes(models.successor_weights @ x.T, -1, -2)
    return jnp.stack(
        (
            jnp.sqrt(jnp.mean((r - reward_target) ** 2, axis=-1)),
            jnp.sqrt(jnp.mean(jnp.sum((n - successor_target) ** 2, axis=-1), axis=-1)),
            jnp.sqrt(
                jnp.mean(jnp.sum((n @ decode - common_target) ** 2, axis=-1), axis=-1)
            ),
        ),
        axis=-1,
    )


def summarize(history):
    # checkpoint, run, option, metric
    return {
        name: {
            key: {
                stat: [mean_and_se(row[:, o, m].tolist())[stat] for row in history]
                for stat in ("mean", "se")
            }
            for m, key in enumerate(METRICS)
        }
        for o, name in enumerate(NAMES)
    }


def save_checkpoint(path, state, problem, step, source):
    return save_json(
        path,
        {
            "schema_version": 1,
            "step": step,
            "seeds": SEEDS,
            "model_names": NAMES,
            "parameters": PARAMETERS._asdict(),
            "frozen_options": source,
            "positions": problem.positions.tolist(),
            "features": problem.features.tolist(),
            "reward_axes": ["run", "option", "input_feature"],
            "successor_axes": ["run", "option", "output_feature", "input_feature"],
            "reward_weights": state.learners.model.reward_weights.tolist(),
            "successor_weights": state.learners.model.successor_weights.tolist(),
            "reward_traces": state.learners.reward_trace.tolist(),
            "successor_traces": state.learners.successor_trace.tolist(),
            "environment_positions": state.environment.position.tolist(),
            "keys": jax.random.key_data(state.key).tolist(),
            "episodes": state.episodes.tolist(),
        },
        compressed=True,
    )


def train_case(data, name, pi, beta, tables, refs, directory, source):
    problem = Problem(data.params, jnp.array(data.positions), tables[name], pi, beta)
    d, geometry = decoder(problem.features, data.nonterminal_indices)
    ids = jnp.array(data.nonterminal_indices)
    state = jax.vmap(lambda seed: initialize(seed, problem))(jnp.array(SEEDS))
    advance = jax.jit(jax.vmap(lambda s: train_block(s, problem, steps=CHECKPOINT)))
    history, artifacts = [], []
    targets = (
        problem.features[ids],
        jnp.array(refs[name]["reward"])[:, ids],
        jnp.array(refs[name]["successor"])[:, ids],
        jnp.array(d),
        jnp.array(refs["one_hot"]["successor"])[:, ids],
    )
    for step in range(0, STEPS + 1, CHECKPOINT):
        if step:
            state = advance(state)
        errors = model_errors(state.learners.model, *targets)
        if not all(
            bool(jnp.isfinite(v).all())
            for v in jax.tree.leaves((state.learners, errors))
        ):
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"{name}_nonfinite_step_{step}.npz",
                **{
                    f"state_{i}": np.asarray(v)
                    for i, v in enumerate(jax.tree.leaves(state.learners))
                },
            )
            raise RuntimeError(
                f"Nonfinite {name} model at step {step}; parameters preserved"
            )
        history.append(np.asarray(errors))
        if step in (0, 10_000, 20_000, 50_000):
            artifacts.append(
                save_checkpoint(
                    directory / f"{name}_step_{step:05d}.json.gz",
                    state,
                    problem,
                    step,
                    source,
                )
            )
        if step % 5000 == 0:
            print(f"{name}: {step:,}/{STEPS:,} model transitions per seed", flush=True)
    history = np.stack(history)
    summary = summarize(history)
    terminal_x = problem.features[jnp.array(data.terminal)]
    terminal_zero = bool(
        jnp.all(state.learners.model.reward_weights @ terminal_x.T == 0)
        & jnp.all(state.learners.model.successor_weights @ terminal_x.T == 0)
    )
    performance = {}
    for model, metrics in summary.items():
        for metric in METRICS[:2]:
            values = metrics[metric]["mean"]
            performance[f"{model}_{metric}_halved"] = values[-1] <= 0.5 * values[0]
            if name == "one_hot":
                limit = (
                    0.01
                    if model in NAMES[:4]
                    else (0.1 if model == "shortest_path" else 0.2)
                )
                performance[f"{model}_{metric}_within_{limit}"] = values[-1] <= limit
    return {
        "features": problem.features.tolist(),
        "decoder": d.tolist(),
        "geometry": geometry,
        "summary": summary,
        "history_axes": ["checkpoint", "run", "option", "metric"],
        "metric_names": METRICS,
        "history": history.tolist(),
        "checkpoints": artifacts,
        "finite": True,
        "terminal_predictions_zero": terminal_zero,
        "performance_checks": performance,
    }, history


def _run(directory, figures):
    from rrs.experiments.tile_coded_model_plots import plot_results

    revision, dirty = _revision()
    data = build_baselines()
    tables = {name: features(data, name) for name in ("one_hot", "tile_coding")}
    print(
        "Training and freezing both seed-100 options at 200,000 transitions", flush=True
    )
    pi, beta, weights = freeze_options(data)
    provenance = {
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("jax", "jaxlib", "numpy", "matplotlib")
        },
        "backend": jax.default_backend(),
        "jax_enable_x64": jax.config.jax_enable_x64,
    }
    source = save_json(
        directory / "frozen_options.json",
        {
            "provenance": provenance,
            "positions": data.positions,
            "model_names": NAMES,
            "probabilities": pi.tolist(),
            "stopping": beta.tolist(),
            "trained_options": weights,
            "option_parameters": {
                "alpha": 0.1,
                "alpha_actor": 0.1,
                "gamma": 0.99,
                "lambda_": 0.0,
                "lambda_actor": 0.0,
            },
        },
    )
    refs, agreement = build_references(data, pi, beta, tables)
    quality = {}
    for i in (5, 6):
        result = evaluate_policy(
            reference_model(data.positions),
            pi[i].tolist(),
            beta[i].tolist(),
            data.subtasks["reward_respecting"].stopping_values,
            data.terminal,
            data.positions.index((3, 7)),
        )
        start = data.positions.index((3, 1))
        quality[NAMES[i]] = {
            "return": result.values[start],
            "safe_attainment": result.safe_success[start],
            "value_error_bound": result.value_error_bound,
            "probability_gap": result.probability_gap,
        }
    print("Auditing both fixed options with 80,000 independent executions", flush=True)
    audits = rollout_audits(data, pi, beta, tables, refs)
    cases, histories = {}, {}
    for name in tables:
        cases[name], histories[name] = train_case(
            data, name, pi, beta, tables, refs, directory, source
        )
        save_json(directory / f"{name}_metrics.json", cases[name])
    # Only common-unit metrics receive paired representation comparisons.
    difference = histories["tile_coding"] - histories["one_hot"]
    paired = summarize(difference)
    for metrics in paired.values():
        del metrics["native_successor_rmse"]
    validity = {
        "finite": all(c["finite"] for c in cases.values()),
        "terminal_predictions_zero": all(
            c["terminal_predictions_zero"] for c in cases.values()
        ),
        "oracle_bounds": all(b <= 1e-10 for r in refs.values() for b in r["bounds"]),
        "rollout_audits": all(
            a["reward_passed"]
            and all(r["successor_passed"] for r in a["representations"].values())
            for a in audits
        ),
    }
    report = {
        "milestone": "11-tile-coded-models",
        "issue": "https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/21",
        "provenance": provenance,
        "seeds": SEEDS,
        "steps": STEPS,
        "checkpoints": list(range(0, STEPS + 1, CHECKPOINT)),
        "parameters": PARAMETERS._asdict(),
        "model_names": NAMES,
        "frozen_options": source,
        "option_quality": quality,
        "references": refs,
        "reference_agreement": agreement,
        "rollout_audits": audits,
        "cases": cases,
        "paired_tile_minus_one_hot": paired,
        "implementation_checks": validity,
        "figures": [
            str(p)
            for p in plot_results(cases, list(range(0, STEPS + 1, CHECKPOINT)), figures)
        ],
    }
    save_json(directory / "learning.json", report)
    print(
        json.dumps(
            {
                "option_quality": quality,
                "implementation_checks": validity,
                "failed_performance_checks": {
                    name: [k for k, v in c["performance_checks"].items() if not v]
                    for name, c in cases.items()
                },
                "final_errors": {
                    name: {
                        o: {k: v["mean"][-1] for k, v in m.items()}
                        for o, m in c["summary"].items()
                    }
                    for name, c in cases.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if not all(validity.values()):
        raise RuntimeError("Implementation checks failed; report preserved")


def run_experiment():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tile_coded_models")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    try:
        _run(args.output_dir, args.figures_dir)
    except (ValueError, RuntimeError) as error:
        save_json(
            args.output_dir / "failure.json",
            {"error": str(error), "code_revision": _revision()[0]},
        )
        raise
