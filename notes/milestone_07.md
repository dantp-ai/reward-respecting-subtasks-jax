# Milestone 7: four reward-respecting hallway options

Implements [issue #13](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/13)
on `milestone/07-four-room-options`, branched from master `543bf6b`.
The [contract](four_room_options_contract.md) was committed as `86f4ec7` before
training. Experiment code revision: `e24dc66e5f9a490afd1fc3fe681c087478cbb827`.

All **28 frozen experiment checks passed**, with each option assessed separately.
The full run was repeated with byte-identical reports, snapshots and figures.
This validates the four learned options under our declared one-million-step
budget. It does not establish that all four converge within Figure 7's shorter
200,000-step duration, or reproduce that figure's learning curves numerically.

## Reproduce

```sh
uv sync --locked
uv run --locked pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked python -m rrs.experiments.four_room_options
```

The command trains 30 seeds (7000–7029), each with four independent actor/critic
learners receiving the same uniform-behavior transitions. It saves critics every
2,000 steps and independently evaluates actual stochastic policies at 0, 200,000
and 1,000,000 steps. Outputs:

- [Critic and actual-value RMSE](../figures/milestone_07_learning.png)
- [Estimated and actual start returns](../figures/milestone_07_start_values.png)
- [Four stochastic policies and stopping maps](../figures/milestone_07_options.png)
- `artifacts/four_room_options/learning.json`: full references, per-run critic
  curves, evaluated state values and outcome probabilities, means and standard
  errors, solver bounds, acceptance checks and provenance.
- `artifacts/four_room_options/options_step_0200000.json.gz` and
  `options_step_1000000.json.gz`: all 30 runs' actors, critics, stochastic action
  probabilities and stopping decisions, with axes and state/feature ordering.

The three figures are tracked; JSON and gzip artifacts are ignored by git.
`--output-dir` and `--figures-dir` override the directories. Failed acceptance
checks raise after saving the report and figures. The maps always show seed
7000, chosen before training; arrow lengths represent action probabilities and
red outlines indicate stopping on arrival. Dashed segments between independently
evaluated checkpoints in the curves are visual guides, not additional evaluations.

## Subtasks and independent evaluation

Use the validated [Milestone 6 environment](four_room_contract.md), 103 one-hot
nonterminal features and four hallway bonuses of 1. Main-task weights remain
zero. Actual environment rewards are the cumulants. Gamma is 0.99, both learning
rates are 0.05, both lambdas are zero, and all weights/traces start at zero.
These equations and hallway tasks follow the locally saved paper's
[Sections 2–3 and 7](https://arxiv.org/html/2202.03466v4#S7).

Each shared transition produces four separate importance ratios, TD errors and
updates using the existing Equation 10/UWT implementation. The ratio uses the
requested action's probability divided by 0.25. Behavior resets to S only after
learning a terminal transition; an option's stopping decision never resets it.
There are 1,152–1,262 completed behavior episodes per run.

The independent Python environment supplies the stochastic oracle and evaluator.
The oracle optimizes Equation 2, choosing on arrival between stopping at `z` and
continuing at `gamma*V`. Learning retains Equation 9's comparison `z >= V`, using
the pre-update critic. Tests distinguish these comparisons on a toy problem;
the resulting stopping maps agree at all four exact hallway references here.
Goal arrival keeps its +1 reward, forces stopping, and has zero stopping value.
Every option executes at least one action, including when started at a stopping
state.

Evaluation sums over the learned softmax action probabilities and all stochastic
environment outcomes, using learned stopping. It does not replace policies with
their greedy actions or substitute learned critic values for actual returns.
Sparse Bellman iteration certifies discounted return error via
`residual/(1-gamma) <= 1e-8`. Lower/upper iterations bound undiscounted outcome
probabilities to a gap of `1e-8`; no rollout truncation is used.

The four exact optimizers converge in 120–166 sweeps, with maximum Bellman
residual `9.43e-13`. Independently evaluating their greedy policies agrees with
optimized values within `5.16e-10` over all states. Across the 360 learned-policy
evaluations (30 seeds × 4 options × 3 checkpoints), the maximum return error
bound is `9.99975e-9`, the maximum probability gap is `9.99881e-9`, and the longest
evaluation takes 2,956 iterations. No actual value exceeds its subtask optimum
by more than `4.45e-16`. Targets and goal are stopping states in every snapshot;
all recorded parameters/results are finite and terminal critic values are zero.

## Results at the frozen final budget

Start returns below are means ± one standard error over 30 seeds. The oracle
values refer to each subtask, including its stopping bonus, not just main-task
environment return.

| Option | Exact optimum from S | Estimated return from S | Actual return from S |
| --- | ---: | ---: | ---: |
| H1 | 0.952524 | 0.947274 ± 0.001253 | 0.947411 ± 0.000032 |
| H2 | 0.893704 | 0.880015 ± 0.001785 | 0.881907 ± 0.000089 |
| H3 | 0.782011 | 0.747021 ± 0.002083 | 0.749106 ± 0.000450 |
| H4 | 0.727060 | 0.650128 ± 0.002137 | 0.655052 ± 0.000530 |

Both estimated and actual mean start returns are within the required 0.15 of
their respective optima. RMSE compares all 103 nonterminal states with the
corresponding optimized subtask values, then averages across seeds:

| Option | Critic RMSE, mean ± SE | Actual-value RMSE, mean ± SE | Final critic RMSE / peak mean RMSE |
| --- | ---: | ---: | ---: |
| H1 | 0.093611 ± 0.002962 | 0.051776 ± 0.003376 | 11.50% |
| H2 | 0.086070 ± 0.003062 | 0.038416 ± 0.003771 | 10.31% |
| H3 | 0.085240 ± 0.003027 | 0.038929 ± 0.003781 | 10.60% |
| H4 | 0.101151 ± 0.003107 | 0.056295 ± 0.001642 | 13.13% |

Every RMSE is below 0.40, and each final mean critic RMSE is below 60% of its
peak checkpoint mean. These are project acceptance thresholds fixed before
training, not numerical claims from the paper.

## The 200,000-step checkpoint

Figure 7 spans 200,000 transitions. The contract retained this checkpoint and
declared the longer one-million-step convergence budget before any training.
The shorter run's means are:

| Option | Critic RMSE | Estimated start | Actual start | Actual-value RMSE |
| --- | ---: | ---: | ---: | ---: |
| H1 | 0.291281 | 0.932217 | 0.932861 | 0.129491 |
| H2 | 0.221488 | 0.836581 | 0.841040 | 0.093940 |
| H3 | 0.229639 | 0.421353 | 0.579819 | 0.122691 |
| H4 | 0.603551 | 0.001757 | 0.044280 | 0.439967 |

At this checkpoint H3's actual start gap is 0.202191, and H4's is 0.682780;
both would fail the final start-return gate. H4 also exceeds both final RMSE
limits and learns substantially later than the other options in this experiment.
The final gates pass only at the declared longer budget. No seeds, step sizes,
duration or acceptance thresholds were changed after observing these results.

## Hallway attainment and penalty exposure

These are probabilities, not discounted returns. Target success means stopping
at the hallway; safe target success additionally excludes every negative-reward
arrival. Penalty exposure includes a negative reward on the stopping action.
Learned rows average the final stochastic policies over 30 seeds. Reference rows
use the first greedy action at each optimized-value tie.

| Option | Policy | Target from S | Safe target from S | Any penalty from S |
| --- | --- | ---: | ---: | ---: |
| H1 | Reference | 0.999992 | 0.999992 | 3.73e-10 |
| H1 | Learned | 1.000000 | 1.000000 | 3.19e-10 |
| H2 | Reference | 0.998745 | 0.996638 | 0.003362 |
| H2 | Learned | 0.997855 | 0.996036 | 0.003964 |
| H3 | Reference | 0.998744 | 0.996636 | 0.003364 |
| H3 | Learned | 0.996953 | 0.995122 | 0.004046 |
| H4 | Reference | 0.007799 | 0.007782 | 0.003364 |
| H4 | Learned | 0.077384 | 0.077234 | 0.004604 |

H1's learned target probabilities round to one; they are not exactly one.
Probabilities below `1e-8` are below the certified probability resolution.
For H4, the reward-respecting objective can favor reaching the main goal instead
of the target hallway. Even the optimal reference stops at H4 from S only about
0.78% of the time. Target probability therefore measures behavior, not optimality,
and was never a universal acceptance threshold. Higher target attainment does
not imply higher expected subtask return.

Uniform averages over all 103 nonterminal starting states, including starts in
the penalty region, show the broader behavior:

| Option | Policy | Target | Safe target | Any penalty |
| --- | --- | ---: | ---: | ---: |
| H1 | Reference | 0.301276 | 0.294804 | 0.093227 |
| H1 | Learned | 0.517012 | 0.510979 | 0.103350 |
| H2 | Reference | 0.536501 | 0.529214 | 0.095136 |
| H2 | Learned | 0.575483 | 0.570112 | 0.105310 |
| H3 | Reference | 0.609956 | 0.602578 | 0.095113 |
| H3 | Learned | 0.650280 | 0.644572 | 0.105154 |
| H4 | Reference | 0.125122 | 0.108000 | 0.111874 |
| H4 | Learned | 0.193540 | 0.167306 | 0.107614 |

## Tests, reproducibility and next scope

All **198 tests pass**, including 14 added for stochastic subtask backups,
independent policy evaluation, shared-experience updates and artifacts. Tests
cover terminal/stopping timing, fractional stopping, safe versus unsafe target
attainment, unresolved nontermination, deterministic special cases, requested
versus realized actions, separate importance ratios, terminal learning before
reset, eager/JIT/VMAP agreement and seeded repeatability. Per-option acceptance
tests ensure that another option's result cannot hide a failed gate. The existing
GitHub PR workflow runs this short suite; full experiments run separately.

Two full experiments at the implementation revision, with a clean tracked
worktree, produced byte-identical JSON, both gzip snapshots and all three PNGs.
Report SHA-256:
`c1dbe938c31badf7638424bd5c7f58f17188d63d9d44a525f4b44c4a6a5c7058`.
Snapshot SHA-256 values:

- 200,000: `e8ac21ed30094b5c4cc463eec4b1883996d3bab44481fc2f4446e91b08de45d2`
- 1,000,000: `2e800352b546604d862af4fa73ad9786e243b3b5f503a20a991e10580ff7ec40`

Versions: Python 3.12.9, JAX/jaxlib 0.7.0, Matplotlib 3.10.5. Learning uses
float32; independent references/evaluations use Python doubles. Repeatability
was checked within this environment. A later code revision changes report
provenance even if numerical results remain identical.

Four-room option model learning and planning comparisons remain subsequent work.
This milestone saves the full stochastic policies and stopping rules needed to
freeze options for model learning; it does not yet establish a planning advantage.

| Equation or concept | Implementation |
| --- | --- |
| Stochastic Equation 2 optimal subtask backup | `rrs/rl/stochastic_subtasks.py::solve_subtask` |
| Independent actual returns and outcome probabilities | `rrs/rl/stochastic_subtasks.py::evaluate_option` |
| Equation 4 stopping value | Existing `rrs/rl/subtasks.py::stopping_value` |
| Equations 9–11 and UWT | Existing `rrs/rl/learning.py::actor_critic_update` |
| Four learners sharing one transition | `rrs/experiments/four_room_options.py::experience_step` |
| Fixed-seed experiment | `rrs/experiments/four_room_options.py::main` |
| References, independent checks and snapshots | `rrs/experiments/four_room_option_report.py` |
| Scientific figures | `rrs/experiments/four_room_option_plots.py` |
