"""Separate, declared diagnostics for the frozen Milestone 10 result."""

import gzip
import hashlib
import json
import platform
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rrs.envs import reference, two_room
from rrs.experiments.hallway import build_baselines
from rrs.experiments.learning_report import mean_and_se, reference_model
from rrs.experiments.option_learning import TrainingState
from rrs.experiments.tile_coded_options import SEEDS, hallway_signal, training_problem
from rrs.experiments.two_room import _revision
from rrs.rl.learning import (
    ActorCritic,
    LearningParameters,
    importance_ratio,
    log_policy_gradient,
    softmax_policy,
    stopping_rule,
    td_error,
    update_weights_and_traces,
)
from rrs.rl.policy_evaluation import evaluate_policy

SOURCE_SHA256 = "2a0e7c460275b468c61f7383bead4e7a4fa4fafb114297e912f19640293856f7"
SOURCE = Path("artifacts/tile_coded_options/learning.json")
PARAMETERS = LearningParameters()
METRICS = ("critic_rmse", "stochastic_start_value", "safe_hallway_probability")


class Problem(NamedTuple):
    environment: two_room.Params
    positions: jax.Array
    critic_features: jax.Array
    actor_features: jax.Array


class Run(NamedTuple):
    final: TrainingState
    critics: jax.Array
    actors: jax.Array


def feature_tables(data):
    tables = {}
    for name in ("one_hot", "tile_coding"):
        problem, encode = training_problem(name)
        tables[name] = jax.vmap(lambda obs: encode(problem.representation, obs))(
            jnp.asarray(data.positions)
        )
    tables["binary_tiles"] = 2 * tables["tile_coding"]
    penalty = jnp.array(
        [bool(data.params.penalties[p[0], p[1]]) for p in data.positions]
    )
    x = tables["tile_coding"]
    tables["split_penalty_tiles"] = jnp.concatenate(
        (x * (~penalty)[:, None], x * penalty[:, None]), axis=1
    )
    return tables


def make_problem(data, tables, critic, actor):
    return Problem(
        data.params, jnp.asarray(data.positions), tables[critic], tables[actor]
    )


def initialize(seed, problem):
    nc, na = problem.critic_features.shape[1], problem.actor_features.shape[1]
    key = jax.random.key(seed)
    weights = ActorCritic(
        jnp.zeros(nc), jnp.zeros((4, na)), jnp.zeros(nc), jnp.zeros((4, na))
    )
    return TrainingState(
        two_room.reset(key, problem.environment), weights, key, jnp.array(0)
    )


def step(state, problem, parameters=PARAMETERS):
    """Same equations as actor_critic_update, allowing separate feature spaces."""
    next_key, action_key, step_key, reset_key = jax.random.split(state.key, 4)
    action = jax.random.randint(action_key, (), 0, 4)
    destination, observation, reward, done = two_room.step(
        step_key, state.environment, action, problem.environment
    )
    # State indexing is confined to this finite-environment diagnostic runner.
    s = jnp.argmax(jnp.all(problem.positions == state.environment.position, axis=1))
    ns = jnp.argmax(jnp.all(problem.positions == observation, axis=1))
    cx, nx, ax = (
        problem.critic_features[s],
        problem.critic_features[ns],
        problem.actor_features[s],
    )
    weights = state.weights
    value, next_value = weights.critic @ cx, weights.critic @ nx
    z = jnp.where(done, 0.0, hallway_signal(observation))
    beta = stopping_rule(z, next_value, done)
    probabilities = softmax_policy(weights.actor, ax)
    rho = importance_ratio(probabilities, action, 0.25)
    delta = td_error(reward, z, value, next_value, beta, parameters.gamma)
    critic, ec = update_weights_and_traces(
        weights.critic,
        weights.critic_trace,
        cx,
        delta,
        rho,
        beta,
        parameters.alpha,
        parameters.gamma,
        parameters.lambda_,
    )
    actor, ea = update_weights_and_traces(
        weights.actor,
        weights.actor_trace,
        log_policy_gradient(probabilities, action, ax),
        delta,
        rho,
        beta,
        parameters.alpha_actor,
        parameters.gamma,
        parameters.lambda_actor,
    )
    reset = two_room.reset(reset_key, problem.environment)
    return TrainingState(
        two_room.State(jnp.where(done, reset.position, destination.position)),
        ActorCritic(critic, actor, ec, ea),
        next_key,
        state.episodes + done,
    )


@partial(jax.jit, static_argnames=("steps", "checkpoint"))
def train(seed, problem, parameters=PARAMETERS, steps=50_000, checkpoint=50_000):
    if steps < 1 or checkpoint < 1 or steps % checkpoint:
        raise ValueError("steps must be positive and divisible by checkpoint")
    initial = initialize(seed, problem)

    def transition(state, _):
        return step(state, problem, parameters), None

    def block(state, _):
        updated, _ = jax.lax.scan(transition, state, None, length=checkpoint)
        return updated, (updated.weights.critic, updated.weights.actor)

    final, (critics, actors) = jax.lax.scan(
        block, initial, None, length=steps // checkpoint
    )
    return Run(
        final,
        jnp.concatenate((initial.weights.critic[None], critics)),
        jnp.concatenate((initial.weights.actor[None], actors)),
    )


def configurations():
    cases = [
        ("one_hot", "one_hot", "one_hot", 0.1, 200_000),
        ("tile_coding", "tile_coding", "tile_coding", 0.1, 200_000),
        ("binary_tiles", "binary_tiles", "binary_tiles", 0.025, 50_000),
        ("tiled_critic_only", "tile_coding", "one_hot", 0.1, 50_000),
        ("tiled_actor_only", "one_hot", "tile_coding", 0.1, 50_000),
        (
            "split_penalty_tiles",
            "split_penalty_tiles",
            "split_penalty_tiles",
            0.1,
            50_000,
        ),
    ]
    for representation in ("one_hot", "tile_coding"):
        for alpha in (0.025, 0.05, 0.2):
            cases.append(
                (
                    f"{representation}_alpha_{alpha}",
                    representation,
                    representation,
                    alpha,
                    50_000,
                )
            )
    return cases


def evaluate_tables(data, policies, stopping, seeds=SEEDS):
    model = reference_model(data.positions)
    z = data.subtasks["reward_respecting"].stopping_values
    start, target = data.positions.index((3, 1)), data.positions.index((3, 7))
    rows = []
    for seed, pi, beta in zip(seeds, policies, stopping, strict=True):
        failures = []
        result = None
        for budget in (20_000, 200_000):
            try:
                result = evaluate_policy(
                    model, pi, beta, z, data.terminal, target, max_iterations=budget
                )
                break
            except RuntimeError as error:
                failures.append({"iteration_budget": budget, "error": str(error)})
        rows.append(
            {
                "seed": seed,
                "stochastic_start_value": result.values[start] if result else None,
                "safe_hallway_probability": result.safe_success[start]
                if result
                else None,
                "value_error_bound": result.value_error_bound if result else None,
                "probability_gap": result.probability_gap if result else None,
                "evaluation_failures": failures,
                "evaluation_iterations": result.iterations if result else None,
            }
        )
    return rows


def summarize(rows, metrics):
    return {
        key: mean_and_se([row[key] for row in rows])
        if all(row[key] is not None for row in rows)
        else {"mean": None, "se": None}
        for key in metrics
    }


def snapshot_report(data, problem, run, index, seeds=SEEDS):
    values = run.critics[:, index] @ problem.critic_features.T
    preferences = jnp.einsum(
        "raf,sf->rsa", run.actors[:, index], problem.actor_features
    )
    pi = jax.nn.softmax(preferences, axis=-1)
    beta = stopping_rule(
        jnp.array(data.subtasks["reward_respecting"].stopping_values),
        values,
        jnp.array(data.terminal),
    )
    rows = evaluate_tables(data, pi.tolist(), beta.tolist(), seeds)
    ids = jnp.array(data.nonterminal_indices)
    oracle = jnp.array(data.solutions["reward_respecting"].values)[ids]
    errors = jnp.sqrt(jnp.mean((values[:, ids] - oracle) ** 2, axis=1)).tolist()
    for row, error in zip(rows, errors, strict=True):
        row["critic_rmse"] = error
    return {
        "summary": summarize(rows, METRICS),
        "runs": rows,
    }


def swaps(data, source):
    results = {}
    for policy_name in ("one_hot", "tile_coding"):
        for stopping_name in ("one_hot", "tile_coding"):
            policies = [
                r["policy_probabilities"] for r in source["cases"][policy_name]["runs"]
            ]
            stopping = [r["stopping"] for r in source["cases"][stopping_name]["runs"]]
            rows = evaluate_tables(data, policies, stopping)
            results[f"policy={policy_name},stopping={stopping_name}"] = {
                "summary": summarize(rows, METRICS[1:]),
                "runs": rows,
            }
    return results


def sampled_audit(data, pi, beta, samples=10_000, seed=10_000, cap=10_000):
    """Independent sampled executions; forced first action and arrival stopping."""
    rng = np.random.default_rng(seed)
    indices = {p: i for i, p in enumerate(data.positions)}
    start, target = indices[(3, 1)], indices[(3, 7)]
    transitions = [[reference.step(p, a) for a in range(4)] for p in data.positions]
    cdf = np.cumsum(np.array(pi) / np.sum(pi, axis=1)[:, None], axis=1)
    returns, safe, longest = [], [], 0
    z = data.subtasks["reward_respecting"].stopping_values
    for _ in range(samples):
        s, discount, value, penalized = start, 1.0, 0.0, False
        for length in range(1, cap + 1):
            action = int(np.searchsorted(cdf[s], rng.random(), side="right"))
            position, reward, done = transitions[s][action]
            ns = indices[position]
            penalized |= reward < 0
            value += discount * reward
            if done or beta[ns]:
                value += discount * (0.0 if done else z[ns])
                returns.append(value)
                safe.append(float(not done and ns == target and not penalized))
                longest = max(longest, length)
                break
            s, discount = ns, discount * PARAMETERS.gamma
        else:
            raise RuntimeError("Independent rollout exceeded the declared action cap")
    return {
        "return": mean_and_se(returns),
        "safe_success": mean_and_se(safe),
        "samples": samples,
        "rng_seed": seed,
        "longest_execution": longest,
    }


def geometry_audit(data, tables):
    # Independent host construction: do not call the tile encoder or its builder.
    ids = list(data.nonterminal_indices)
    positions = [data.positions[i] for i in ids]
    keys = [
        (t, tile)
        for t, offset in enumerate(((0, 0), (0, 1), (1, 0), (1, 1)))
        for tile in sorted({tuple((np.array(p) - offset) // 2) for p in positions})
    ]
    independent = np.zeros((len(data.positions), len(keys)))
    for s in ids:
        for t, offset in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            independent[
                s, keys.index((t, tuple((np.array(data.positions[s]) - offset) // 2)))
            ] = 0.5
    x = np.asarray(tables["tile_coding"])[ids].astype(float)
    y = np.array(data.solutions["reward_respecting"].values)[ids]
    weights, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    penalty = np.array([bool(data.params.penalties[p[0], p[1]]) for p in positions])
    gram = x @ x.T
    return {
        "independent_features_equal": np.array_equal(
            independent, np.asarray(tables["tile_coding"])
        ),
        "nonterminal_squared_norms_one": bool(np.all(np.sum(x * x, axis=1) == 1)),
        "terminal_features_zero": bool(
            np.all(np.asarray(tables["tile_coding"])[np.array(data.terminal)] == 0)
        ),
        "rank": int(rank),
        "features": x.shape[1],
        "states": x.shape[0],
        "optimal_value_fit_rmse": float(np.sqrt(np.mean((x @ weights - y) ** 2))),
        "feature_matrix_condition_number": float(singular[0] / singular[-1]),
        "shared_state_pairs": int(np.triu(gram > 0, 1).sum()),
        "shared_pairs_crossing_penalty_boundary": int(
            np.triu((gram > 0) & (penalty[:, None] != penalty[None, :]), 1).sum()
        ),
    }


def write_snapshot(path, problem, result, alpha, steps):
    payload = {
        "seeds": SEEDS,
        "checkpoints": list(range(0, steps + 1, 50_000)),
        "alpha_critic": alpha,
        "alpha_actor": alpha,
        "positions": problem.positions.tolist(),
        "critic_features": problem.critic_features.tolist(),
        "actor_features": problem.actor_features.tolist(),
        "critics": result.critics.tolist(),
        "actors": result.actors.tolist(),
        "critic_axes": ["run", "checkpoint", "feature"],
        "actor_axes": ["run", "checkpoint", "action", "feature"],
        "final_positions": result.final.environment.position.tolist(),
        "episodes": result.final.episodes.tolist(),
    }
    encoded = gzip.compress(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0
    )
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest()}


def main():
    import argparse

    from rrs.experiments.tile_coding_diagnostic_plots import plot_results

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/tile_coding_investigation")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    encoded = SOURCE.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != SOURCE_SHA256:
        raise ValueError("The frozen Milestone 10 source report has changed")
    source, data = json.loads(encoded), build_baselines()
    tables, cases, archives, regressions = feature_tables(data), {}, {}, {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = geometry_audit(data, tables)
    print("Geometry audit:", geometry, flush=True)
    for name, critic, actor, alpha, steps in configurations():
        problem = make_problem(data, tables, critic, actor)
        params = PARAMETERS._replace(alpha=alpha, alpha_actor=alpha)
        print(f"Training {name}: {len(SEEDS)} × {steps:,}, alpha={alpha}", flush=True)
        result = jax.vmap(lambda seed: train(seed, problem, params, steps=steps))(
            jnp.array(SEEDS)
        )
        result.critics.block_until_ready()
        archives[name] = write_snapshot(
            args.output_dir / f"{name}.json.gz", problem, result, alpha, steps
        )
        if name in ("one_hot", "tile_coding", "binary_tiles"):
            original = source["cases"][
                "tile_coding" if name == "binary_tiles" else name
            ]["runs"]
            old_critic = jnp.array([r["critic_weights"] for r in original])
            old_actor = jnp.array([r["actor_weights"] for r in original])
            factor = 2 if name == "binary_tiles" else 1
            regressions[name] = {
                "critic_equal": bool(
                    jnp.allclose(
                        factor * result.critics[:, 1], old_critic, atol=1e-6, rtol=1e-6
                    )
                ),
                "actor_equal": bool(
                    jnp.allclose(
                        factor * result.actors[:, 1], old_actor, atol=1e-6, rtol=1e-6
                    )
                ),
                "max_critic_weight_error": float(
                    jnp.max(jnp.abs(factor * result.critics[:, 1] - old_critic))
                ),
                "max_actor_weight_error": float(
                    jnp.max(jnp.abs(factor * result.actors[:, 1] - old_actor))
                ),
            }
        for checkpoint in range(1, steps // 50_000 + 1):
            case = snapshot_report(data, problem, result, checkpoint)
            case.update(
                {
                    "steps": checkpoint * 50_000,
                    "alpha": alpha,
                    "critic_representation": critic,
                    "actor_representation": actor,
                }
            )
            cases[f"{name}/{checkpoint * 50_000}"] = case
            print(f"  {checkpoint * 50_000:,}: {case['summary']}", flush=True)
    print("Evaluating saved policy/stopping swaps", flush=True)
    swapped = swaps(data, source)
    audits = {}
    for name in ("one_hot", "tile_coding"):
        original = source["cases"][name]["runs"][0]
        audit = sampled_audit(
            data, original["policy_probabilities"], original["stopping"]
        )
        audit["return_passed"] = (
            abs(audit["return"]["mean"] - original["stochastic_start_value"])
            <= 0.001 + 6 * audit["return"]["se"]
        )
        audit["safe_success_passed"] = (
            abs(audit["safe_success"]["mean"] - original["safe_hallway_probability"])
            <= 0.001 + 6 * audit["safe_success"]["se"]
        )
        audits[name] = audit
    revision, dirty = _revision()
    report = {
        "milestone": "10-tile-coding-investigation",
        "seeds": SEEDS,
        "source_report": {"path": str(SOURCE), "sha256": SOURCE_SHA256},
        "code_revision": revision,
        "tracked_worktree_dirty": dirty,
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "jax_enable_x64": jax.config.jax_enable_x64,
        },
        "geometry_audit": geometry,
        "original_runner_and_scaling_regression": regressions,
        "configurations": configurations(),
        "cases": cases,
        "policy_stopping_swaps": swapped,
        "independent_rollout_audits": audits,
        "archives": archives,
        "evaluation_iteration_budgets": [20_000, 200_000],
    }
    evaluated = [
        r for group in (cases, swapped) for case in group.values() for r in case["runs"]
    ]
    report["evaluation_status"] = {
        "total": len(evaluated),
        "retried": sum(bool(r["evaluation_failures"]) for r in evaluated),
        "unresolved": sum(r["safe_hallway_probability"] is None for r in evaluated),
    }
    report["audit_passed"] = (
        geometry["independent_features_equal"]
        and geometry["nonterminal_squared_norms_one"]
        and geometry["terminal_features_zero"]
        and all(r["critic_equal"] and r["actor_equal"] for r in regressions.values())
        and all(
            r["return_passed"] and r["safe_success_passed"] for r in audits.values()
        )
    )
    path = args.output_dir / "investigation.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paths = plot_results(report, args.figures_dir)
    print(
        json.dumps(
            {
                "report": str(path),
                "figures": [str(p) for p in paths],
                "audit_passed": report["audit_passed"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["audit_passed"]:
        raise RuntimeError("Investigation audit failed; results preserved")


if __name__ == "__main__":
    main()
