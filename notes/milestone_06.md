# Milestone 6: stochastic four-room foundation

Implements [issue #11](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/11)
on `milestone/06-four-room-foundation`, branched from master `025d300`.
The [contract](four_room_contract.md) was committed as `e03c933` before
implementation and baseline experiments. Experiment code revision:
`d2bba7b81a670c3e0e30ed9f00ba5dda41c337db`.

## Reproduce

```sh
uv sync --locked
uv run --locked pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked python -m rrs.experiments.four_room
```

The command enumerates the independent Python environment, solves the main task,
checks the greedy policy with a separate linear solve, and audits sampled returns
from four fixed starts. Outputs:

- [Geometry, optimal primitive policy and values](../figures/milestone_06_four_room_baseline.png)
- [Independent return audit](../figures/milestone_06_four_room_return_audit.png)
- `artifacts/four_room/baseline.json`: layout, full transition distributions,
  values, greedy actions, residual bounds, raw rollout returns and lengths,
  seeds, configuration, versions and acceptance checks.

The two figures are tracked; the JSON report is ignored by git. `--output-dir`
and `--figures-dir` override the default directories. Failed acceptance checks
raise only after saving the report and figures.

## Paper target and environment

This milestone establishes the environment and primitive-action oracle for
[Section 7 / Figure 6](https://arxiv.org/html/2202.03466v4#S7). Four-room option
learning, model learning and planning comparisons remain subsequent work.

The Figure 6 map has 104 traversable cells, including the terminal goal, on a
13-by-13 grid. Start is `(4,1)`, goal is `(9,7)`, and H1–H4 are `(6,2)`, `(3,6)`,
`(7,9)`, `(10,6)`. Twelve penalty cells fill rows 7–10, columns 2–4. Coordinates
include the outer wall and increase downward/rightward. The contract records
the complete ASCII map, including the staggered horizontal dividing walls.

The intended movement occurs with probability 2/3; each other direction,
including the opposite direction, has probability 1/9. Blocked directions keep
their probability mass and leave the position unchanged. Reward is +1 on goal
arrival, -1 on penalty arrival, and zero otherwise. The goal then absorbs with
reward zero. A blocked move in a penalty cell gives -1, following the two-room
arrival-reward interpretation of the paper's per-step gray-region description.
This timing convention was fixed before results were observed.

The JAX environment exposes coordinates as observations. The separate existing
one-hot representation provides 103 nonterminal features and zero goal features.
Hallway labels are metadata for future subtasks; they do not change the reward.

## Baseline results

All **7 experiment acceptance checks passed** with unchanged seeds, budgets,
geometry and tolerances. Exact stochastic value iteration gives:

| Quantity | Result |
| --- | ---: |
| Optimal start value | 0.725118384944 |
| Synchronous Bellman sweeps | 171 |
| Final Bellman residual | 9.10e-13 |
| Value error bound, residual / (1-gamma) | 9.10e-11 |
| Greedy-policy linear-solve error bound | 4.44e-14 |
| Maximum difference between the two value solutions | 5.45e-12 |

Use gamma=0.99, zero initial values, Bellman tolerance 1e-12, and a maximum of
20,000 sweeps. Both solvers use Python doubles. Value iteration takes the maximum
of expected action returns; the policy evaluator solves `(I-gamma*P_pi)V=r_pi`.
Terminal rewards enter the reward term and terminal outcomes never bootstrap.
The goal's value is therefore zero even though arriving at it yields +1.

The greedy policy chooses the first action within 1e-12 of the maximum in
UP, DOWN, LEFT, RIGHT order. The policy plot shows intended actions; movement
remains stochastic. The start value is computed from the stochastic Bellman
equations, not from discounting a single shortest route.

## Independent rollout audit

Each start has 20,000 episodes from a fresh Python `random.Random` stream, using
seeds 6000–6003 in table order. A nine-ticket sampler realizes the directions
without using the enumerated transition table or JAX's categorical sampler.

| Start | Oracle value | Sampled return, mean ± SE | Absolute error | Frozen acceptance limit |
| --- | ---: | ---: | ---: | ---: |
| S `(4,1)` | 0.725118 | 0.728343 ± 0.000849 | 0.003224 | 0.010096 |
| H1 `(6,2)` | 0.378309 | 0.382493 ± 0.007599 | 0.004184 | 0.050595 |
| H3 `(7,9)` | 0.939168 | 0.938600 ± 0.000467 | 0.000568 | 0.007801 |
| Penalty `(9,3)` | -1.720737 | -1.728434 ± 0.014110 | 0.007697 | 0.089660 |

The acceptance limit is `0.005 + 6*SE + gamma**3000/(1-gamma)`. The start-state
sample mean is about 3.8 sample standard errors above the oracle; it passes that
predefined limit. No seeds or thresholds were adjusted. The figure shows ±6 SE
bars and the actual differences from the oracle; the additional 0.005 allowance
and tail bound are recorded in the report and table.

All 80,000 episodes terminated. The longest took 96 actions, below the frozen
3,000-step cap. The maximum omitted-tail bound at that cap is `8.05e-12`; no
sample actually required truncation. Signed returns are retained, including rare
large losses caused by stochastic excursions into the penalty region.

## Tests, reproducibility and scope

All **184 tests pass**, including 26 added for this milestone. The tests compare
all 416 state-action distributions between independent JAX and Python geometry,
check wall probability aggregation and reward timing by hand, validate terminal
absorption and separate features, and exercise eager/JIT/VMAP agreement.

A fixed sampler test checks 18,000 keys for each of 16 state-action pairs, using
seed 6100 and the frozen `0.005 + 6*sqrt(p*(1-p)/18000)` marginal probability
tolerance. It covers corners, a hallway, goal adjacency and penalties, including
coincident blocked outcomes. Other tests cover mixed terminal/continuing
Bellman backups, stochastic action policies, closed cycles, invalid models,
nonconvergence, deterministic special cases, signed rollout discounting,
truncation reporting and reproducible short audits. The existing PR workflow
runs the short suite; full 80,000-episode audits run separately. Pre-commit checks
pass.

Two complete runs at the implementation revision, with a clean tracked worktree,
produced byte-identical JSON and both PNGs. Report SHA-256:
`cca5c1590f4c0e150c24fd3edab1ce38d7ed5fb24a9005ecf519d5c6701ad2ec`.
Versions: Python 3.12.9, JAX/jaxlib 0.7.0 and Matplotlib 3.10.5. Repeatability was
verified in this environment; different RNG libraries or software versions can
produce different sampled trajectories.

The geometry is transcribed from the paper's figure, and the contract explicitly
records the reward-timing interpretation. This establishes a trusted stochastic
primitive baseline. It does not yet reproduce the paper's multi-option advantage
or validate learned four-room options and models.

| Equation or concept | Implementation |
| --- | --- |
| Section 7 movement/reward distribution | `rrs/envs/four_room.py::outcomes` |
| Keyed JAX environment sampling | `rrs/envs/four_room.py::step` |
| Independent coordinate geometry and sampling | `rrs/envs/four_room_reference.py` |
| Expected primitive Bellman backup | `rrs/rl/stochastic_exact.py::action_values` |
| Exact stochastic value iteration | `rrs/rl/stochastic_exact.py::value_iteration` |
| Discounted fixed-policy linear solve | `rrs/rl/stochastic_exact.py::evaluate_policy` |
| Baseline, rollout audit and scientific figures | `rrs/experiments/four_room.py` |
