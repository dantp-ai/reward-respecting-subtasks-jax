# Milestone 2: exact hallway options and reference models

Implements [issue #3](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/3)
on `milestone/02-exact-hallway-options`, branched from merged `master` at `8731f0c`.

## Reproduce

From the repository root:

```sh
uv sync --locked
uv run pytest -q
uv run python -m rrs.experiments.hallway
```

The demonstration writes `artifacts/hallway/baseline.json` and these tracked figures:

- [Start-state route comparison](../figures/milestone_02_hallway_routes.png)
- [Full policies and stopping states](../figures/milestone_02_hallway_policies.png)

Use `--output-dir PATH` for JSON and `--figures-dir PATH` for figures.
The JSON includes the full subtask values, policies, stopping masks, exact models
of both options and all four actions, state and feature ordering, parameters,
package versions, code revision, and tracked-worktree status.

## Result

The spec's reward-respecting start-value target is `gamma**11`, approximately
`0.895338254258716`. The independent environment oracle verifies every starting
state, and both subtask Bellman residuals are zero.

| Quantity from start `(3, 1)` | Reward-respecting | Shortest path |
| --- | ---: | ---: |
| Actions until hallway | 12 | 6 |
| Penalty arrivals | 0 | 4 |
| Subtask value | 0.895338254258716 | −5.851985059900000 |
| Discounted environment reward | 0 | −3.940399 |
| Discounted hallway feature | 0.886384871716129 | 0.941480149401000 |
| Value-iteration sweeps | 16 | 9 |
| Final Bellman residual | 0 | 0 |

The comparison's subtask value counts a cost on all six actions. Its model uses
only the four actual environmental penalties. The reward-respecting model has
zero accumulated reward from the start: the artificial hallway bonus is not
environment reward and must not enter a planning model.

Parameters: `gamma=0.99`, bonus 1, frozen zero main-task weights, zero initial
subtask values, residual and tie tolerances `1e-12`, maximum 10,000 sweeps,
double precision for exact oracles, UP/DOWN/LEFT/RIGHT policy tie order, explicit
seed 0, one deterministic run. JAX model exports use normal float32 precision
and are checked at `1e-6` absolute tolerance.

## Mathematical choices requiring attention in review

The [contract](hallway_options_contract.md) records the equations and their
implementation before adding learning. Two distinctions matter:

1. Equation 2 discounts a stopping bonus by `gamma**(K-1)`, while Equation 15
   discounts successor features by `gamma**K`. Equation 17's printed successor
   learning update appears to omit a factor of gamma. The oracle follows the
   model definition in Equation 15 and verifies one-step action compatibility.
2. Optimizing Equation 2 compares `z` with `gamma * V`; Equation 9's learned
   stopping rule compares `z` with `V`. They agree at every state in this fixed
   experiment, but not for general stopping values. A toy test demonstrates
   the distinction so future extensions cannot silently conflate them.

The exact reward-respecting option stops on arrival at the hallway, the goal,
and eight penalty cells: rows 1–4, columns 3–4. Those eight cells have forced-action
value `−1`, so stopping for zero is better. This follows the formal stopping rule.
The paper's illustrative learned-policy inset emphasizes the hallway and goal;
our figure shows the additional exact stopping states explicitly. We are not
claiming to have reproduced the learned policy or its training curve yet.

Initiating an option still takes at least one action, including initiation in
a stopping state. Already-terminal starts have zero-duration, zero-output models.
The deterministic policy is one tie-broken optimum; it need not match every
arrow in a learned stochastic policy.

## Verification and code map

All 85 tests pass (27 added in this milestone). Checks include:

- One-hot features, terminal zero features, and cancellation in Equation 4.
- Hand-calculated stopping/continuation, equality, and model-discount examples.
- Independent Python rollouts for both subtask values at all 73 states.
- Independent optimal-action backups and shortest-path breadth-first search.
- Model/rollout agreement at all 73 states for six options: 438 cases.
- Primitive actions, environment termination, and nonterminating-cycle rejection.
- Feature-facing JAX predictions, including JIT/VMAP and feature order after the goal.

Deterministic exhaustive rollouts give exact comparisons here; Monte Carlo
uncertainty is unnecessary. The existing PR workflow runs the expanded suite.

| Equation or concept | Implementation |
| --- | --- |
| One-hot `x(s)`, terminal zero | `rrs/representations/tabular.py::one_hot` |
| Eq. 4: stopping-value construction | `rrs/rl/subtasks.py::stopping_value` |
| Eq. 2: exact optimal GVF | `rrs/rl/subtasks.py::solve_subtask` |
| Eqs. 12, 15: exact reward and successor models | `rrs/rl/option_models.py::exact_option_model` |
| Eq. 16: feature-facing linear prediction | `rrs/rl/option_models.py::LinearExpectationModel.predict` |
| Fixed experiment, reporting, figures | `rrs/experiments/hallway.py` |

Conclusion: exact targets are available for the next milestone's off-policy
hallway-option learner. This milestone adds no training loop or learned models.
