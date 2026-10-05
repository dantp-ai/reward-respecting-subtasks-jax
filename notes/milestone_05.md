# Milestone 5: planning with actions and options

Implements [issue #9](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/9)
on `milestone/05-option-planning`, branched from master `f81d1d2`.
The [contract](planning_contract.md) was committed as `18358b3` before
implementation and planning experiments. Experiment code revision:
`c1f997f40c2657c8c349e7ca5e092058632c314a`.

## Reproduce

```sh
uv sync --locked
uv run --locked pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked python -m rrs.experiments.planning
```

The command validates the Milestone 4 model checkpoints and frozen policy,
constructs exact references, then runs 16 comparisons with independent policy
evaluation. If Milestone 4 artifacts are absent, it first runs that milestone's
canonical experiment. Existing corrupt or incompatible checkpoints fail
explicitly, even when another checkpoint is missing; they are not silently
replaced through regeneration.

Outputs:

- [Exact-model planning](../figures/milestone_05_exact_planning.png)
- [Planning with models learned for 50,000 transitions](../figures/milestone_05_learned_planning.png)
- [Effect of model-training duration](../figures/milestone_05_model_maturity.png)
- `artifacts/planning/planning.json`: configuration, source hashes, per-run
  planned values, value RMSE, actual returns, selected models, final weights,
  look-ahead counts, standard errors and acceptance checks.

Figures are tracked; JSON artifacts are ignored by git. `--model-dir`,
`--output-dir` and `--figures-dir` override the respective default directories.
The report is written before acceptance failures are raised, preserving evidence.

## Protocol

The target is planning in [Section 5, Equations 18–19](https://arxiv.org/html/2202.03466v4#S5)
and policy evaluation in [Appendix B, Equation 20](https://arxiv.org/html/2202.03466v4#A2).
Use zero initial weights, alpha=1, gamma=0.99 and uniform random sampling of the
72 nonterminal one-hot features. Planning seeds 2000–2099 share their sampled
state prefixes across all configurations and model-training durations.

Each run spends 20,000 look-aheads. Evaluating one state-model pair costs one
look-ahead, so primitives use 5,000 updates and either five-model set uses 4,000.
Record planned start values and full value RMSE every 100 look-aheads. Evaluate
actual policy returns at 0, 2,000, 5,000, 10,000 and 20,000. Those diagnostics
cost a separate 1,460 queries per primitive run or 1,825 per option run, including
queries at the zero-feature goal. They are excluded from the planning-work axis.

The 16 cases are three model sets using exact optimal options, one exact
reference for the frozen stochastic reward-respecting option, and three model
sets at each of four model-training durations (0, 10,000, 20,000, 50,000).
Pair model seeds 1000–1099 with planning seeds 2000–2099 by run index. The
shortest-path option and all primitives have identical policies across sources.

Actual return is evaluated in the independent Python environment. Reselect the
greedy model at every primitive step and draw one action from its policy; do not
commit to the selected option until termination. Keep stochastic action
probabilities for the learned hallway option. A Python-double linear solve
includes reachable closed cycles and checks a Bellman residual bound. No rollout
truncation or negative-value clipping is used. First-maximum ties follow
UP, DOWN, LEFT, RIGHT, optional hallway option.

## Results

All **25 frozen acceptance checks passed**. Budgets, thresholds and seeds were
unchanged. The main-task optimum is `0.99**17 = 0.8429431934`, rather than the
hallway subtask's `0.99**11` target.

| Exact optimal-option models | Look-aheads to 95% of optimum, mean ± SE | Actual return after 2,000 look-aheads, mean ± SE | Final actual return |
| --- | ---: | ---: | ---: |
| Primitives | 4,736 ± 120 | 0 ± 0 | 0.842943 |
| + Shortest path | 4,449 ± 119 | 0.033718 ± 0.016601 | 0.842943 |
| + Reward-respecting | 1,607 ± 64 | 0.783937 ± 0.021616 | 0.842943 |

Every exact-model run reached the 95% target. The reward-respecting option used
33.9% of the primitive budget and 36.1% of the shortest-path budget at that
threshold, comfortably below the frozen 80% limits. These are means of each
run's first recorded crossing, with 100-look-ahead measurement resolution.

Shortest path used 93.9% of the primitive budget: a modest improvement in this
experiment, rather than a slowdown. The main qualitative result is substantially
faster useful propagation with reward-respecting options. This experiment does
not establish that adding a shortest-path option always hurts planning, and the
contract intentionally required no particular ordering of those two baselines.

At the final budget, all four exact cases (including the frozen-policy reference)
had maximum value-vector error below `1.59e-7` over every run and state, and their
actual start returns matched the optimum. Learned models after 50,000 transitions
also produced optimal final actual returns in all three model sets. Their mean
planned start values were approximately `0.842940 ± 0.00000143`.

| Reward-respecting set: model-training transitions | Final planned start, mean ± SE | Final actual return, mean ± SE | Mean absolute planned-start error |
| --- | ---: | ---: | ---: |
| 0 | 0 ± 0 | 0 ± 0 | 0.842943 |
| 10,000 | 0.787904 ± 0.005595 | 0.840598 ± 0.000630 | 0.055039 |
| 20,000 | 0.839139 ± 0.000518 | 0.842775 ± 0.000168 | 0.003805 |
| 50,000 | 0.842940 ± 0.00000143 | 0.842943 ± 0 | 0.00000306 |

At 10,000 model transitions, only 47/100 runs reached the planned-value threshold.
The report leaves the all-run mean crossing time undefined instead of averaging
only successful runs. Nevertheless, these models already induced policies with
near-optimal actual returns after sufficient planning. Accurate value magnitude
and effective greedy action selection are distinct outcomes.

At 50,000 model transitions, the reward-respecting set reached the 95% threshold
in 3,641 ± 112 look-aheads, versus 4,736 for primitives and 4,449 for shortest path.
After just 2,000 look-aheads, its actual return was `0.770797 ± 0.021308`.
The exact model of the same frozen stochastic option needed 3,678 ± 112
look-aheads and achieved `0.771675 ± 0.021327` at 2,000. Compared with the optimal
deterministic option's 1,607-look-ahead crossing, this reference indicates that
the learned option itself explains much of the remaining planning-speed gap.
The small learned-versus-exact crossing difference is not evidence that model
approximation improves planning generally.

## Validation, provenance and limits

Two complete runs at the implementation revision produced byte-identical JSON
and all three PNGs, with a clean tracked worktree. Python 3.12.9,
JAX/jaxlib 0.7.0 and Matplotlib 3.10.5 were used. Report SHA-256:
`9bcd468babbe1bb459d630aa85919af1c1b3c5de27893258b372fd53576d7ae1`.
This establishes repeatability in this environment, not bitwise agreement across
different hardware or dependency versions.

Input model experiment revision: `a6aef107e540a1ff5dacafd9215a21c5762a9d3f`.
Source report SHA-256:
`02114c046b85ff9cc09195fbceff72ccbf9f41e0f49545fd2a553fda4eeba6db`.
The planning report records all four checkpoint hashes and the saved stochastic
policy/stopping coefficients. Input validation checks hashes, gamma, array axes,
feature/model/seed ordering, finite coefficients and discounted successor mass.

All weights and metrics were finite; terminal values stayed zero. The largest
policy evaluation error bound was `3.47e-14`, below `1e-8`. Float32 planning is
checked against Python-double tabular calculations and the independent main-task
oracle. Successors already include duration discounting, so planning adds no
extra gamma or artificial hallway reward.

Initial actual return is zero: first-maximum ties choose UP, producing a safe
closed loop from S. This differs from experiments using random ties or sampled
rollouts. Actual returns are evaluated only at five budgets; straight segments
in the figures connect those measured points. Bands show ±1 standard error.
All learned-model runs share the one option learned by Milestone 3 seed 0;
these error bars do not measure uncertainty across separately learned options.

Good start-state performance does not establish convergence everywhere. The
largest final learned-model value error over states and seeds was `0.007453`,
despite the tiny mean start error. Planning with learned models retains the
finite-data limitations documented in [Milestone 4](milestone_04.md).

## Tests and equation map

All **158 tests pass**, including 25 added in this milestone. Coverage includes
hand-calculated backups, mixed-feature semi-gradients, matrix orientation,
asynchronous update order, terminal zero, discounting, query counts, exact-model
convergence, eager/compiled agreement, shared random prefixes, stochastic and
cyclic policy evaluation, per-step option reselection, checkpoint integrity,
missing threshold crossings, and predicted versus actual returns. The existing
PR workflow runs these short tests; full scientific comparisons remain outside
CI. Pre-commit checks pass.

| Equation or concept | Implementation |
| --- | --- |
| Eq. 18: asynchronous tabular backup | `rrs/rl/planning.py::tabular_plan` |
| Eq. 19: feature-facing semi-gradient planning | `rrs/rl/planning.py::planning_step` |
| Eq. 20: greedy selection and one-action execution | `rrs/rl/plan_evaluation.py::induced_policy` |
| Independent discounted policy return | `rrs/rl/plan_evaluation.py::evaluate_action_policy` |
| Validated Milestone 4 inputs | `rrs/experiments/planning_inputs.py` |
| Fixed-seed planning and model comparisons | `rrs/experiments/planning.py` |
| Metrics, acceptance checks and scientific figures | `rrs/experiments/planning_report.py` |
