# Planning with actions and options

Milestone 5, [issue #9](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/9),
branch `milestone/05-option-planning` from master `f81d1d2`.
Sources: [paper Section 5](https://arxiv.org/html/2202.03466v4#S5), Equations 18–19,
and [Appendix B](https://arxiv.org/html/2202.03466v4#A2), Equation 20.
Models follow the [Milestone 4 contract](model_learning_contract.md).

## Backups, updates and accounting

First implement a Python-double asynchronous tabular oracle (Equation 18).
It receives reward and discounted transition tables plus an explicit sequence
of states. Each update uses the latest values from all preceding updates:

```text
V[s] = max_o (R[s,o] + sum_j N[s,o,j] * V[j])
```

Then implement Equation 19 with the existing feature-facing linear model API:

```text
q[o] = model[o].reward(x) + dot(w, model[o].successor(x))
target = max(q)
delta = target - dot(w, x)
w = w + alpha * delta * x
```

Compute every target from pre-update weights; this is a semi-gradient update.
The successor prediction already contains `gamma**K`. Do not multiply it by
gamma again, normalize its mass, or add an artificial stopping bonus. Zero
terminal features imply zero value and no weight change. The planner receives
features and models; it has no primitive/option special cases.

One evaluated state-option backup costs one look-ahead. A complete update costs
4 for primitives alone and 5 when either hallway option is available. Record
both updates and look-aheads; never compare curves at equal updates while
labeling them equal look-ahead work. Diagnostic evaluations are recorded
separately and do not contribute to the planning-work axis.

## Planning and execution policies

Initialize values/weights at zero and use alpha=1. Sample uniformly with
replacement from the 72 nonterminal states, then construct their feature vectors.
The terminal goal is not sampled. Planning seed streams are separate from model
learning: seeds 2000–2099, each starting with `jax.random.key(seed)`. For each
sample, split into next-key and state-key, then use
`jax.random.randint(state_key, (), 0, 72)`. Generate a common state sequence per
seed and reuse its prefix across methods and model-training checkpoints.

Model order is UP, DOWN, LEFT, RIGHT, then the available hallway option. Greedy
selection uses the first maximum in that order. Record ties through this fixed
rule; do not choose a tie rule based on experimental results.

Report planned start value and the return of the induced policy separately.
Match Appendix B's execution semantics: at each primitive time step, recompute
the greedy option using Equation 20. Execute one action drawn from that option's
policy at the current state, then select again at the next state. There is no
commitment to execute the selected option until its stopping time in this
evaluation. Primitive choices are deterministic; the learned hallway choice
retains its softmax action probabilities. This distinction must have a test.

Evaluate this induced primitive policy through the independent Python environment,
using a Python-double linear solve for discounted return. Include reachable
cycles, which can have finite discounted returns without ever reaching the goal;
do not truncate them or require proper episodic termination. Verify a Bellman
residual bound `residual/(1-gamma) <= 1e-8`. Include all signed returns in reports
and figures, rather than clipping negative performance. Fixed tie selection can
give different initial returns from the paper's sampling/tie conventions.

## Experiment sources

1. Exact optimal-option experiment: the Milestone 2 exact primitive models,
   plus its optimal deterministic reward-respecting or shortest-path option.
   Compare the three model sets at equal look-ahead budgets. The known main-task
   optimum is `0.99**17 = 0.8429431933839268`, distinct from the subtask's
   `gamma**11` target.
2. Frozen-policy reference: exact models of the actual Milestone 3 stochastic
   hallway policy saved by Milestone 4. This isolates model-learning error from
   the difference between a learned option and the optimal deterministic option.
3. Learned-model experiment: all three model sets from Milestone 4 checkpoints
   at 0, 10,000, 20,000 and 50,000 model-training transitions. Pair model seeds
   1000–1099 with planning seeds 2000–2099 by run index. All model sets use the
   same primitive-model checkpoint for a given run and training duration.

Validate checkpoint hashes, schema, feature and model ordering, dimensions,
seeds, gamma, finite coefficients and discounted successor mass. Record source
revision and hashes in the new report. If the ignored Milestone 4 artifacts are
absent, regenerate them using its existing command; fail clearly on corrupted or
incompatible artifacts instead of silently substituting other models.

## Frozen protocol (before the first planning experiment)

- 100 planning seeds, 20,000 look-aheads per run and method. Record planned
  values and value RMSE every 100 look-aheads, including initialization.
- Independently evaluate actual induced-policy returns at look-ahead counts
  0, 2,000, 5,000, 10,000 and 20,000. Report means and standard errors over runs.
- For exact optimal-option models and the exact frozen-policy reference, every
  run's final value vector must match the main-task oracle within `1e-5` maximum
  absolute error, and its actual final start return within `1e-5` of optimal.
- In the exact optimal-option experiment, the reward-respecting method's mean
  first checkpoint reaching 95% of optimal start value must use at most 80% of
  the look-aheads required by each alternative. All exact runs must reach that
  target within budget. Report the shortest-path/primitive ratio without
  requiring a particular ranking between those two alternatives.
- For each model set at 50,000 model-training steps, the final mean planned
  start value and actual return must each be within 0.05 of the true optimum.
- For the learned hallway model set, the 50,000-step checkpoint's final mean
  planned-start absolute error must be smaller than at 10,000 model steps, and
  actual mean return must be no worse by more than 0.01. Record intermediate
  checkpoints even if progress is not strictly monotone.
- All parameters and metrics must be finite; terminal values must remain zero.
  These are project thresholds, not numerical results attributed to the paper.

Preserve any failed result. Diagnose equations, successor discounting, indexing,
sampling, query counts, model provenance and policy evaluation before changing
a protocol. Never tune these budgets, seeds or thresholds after seeing results.

## Tests and artifacts

Test hand-computed backups, semi-gradients on nonbinary features, terminal zero,
matrix orientation, one-step action equivalence, multi-step option discounting,
asynchronous update order and exact look-ahead counts. Verify the Python oracle
against the known main-task solution, then eager/JIT/scan agreement within
float32 tolerance `1e-6`. Add independent execution-policy tests for loops,
stochastic choices and reselecting options after every primitive action.

Keep short seeded planning and checkpoint-integrity checks in the existing PR
workflow. Save scientific figures as `figures/milestone_05_*.png`, the ignored
JSON report under `artifacts/planning/`, and results/limitations in
`notes/milestone_05.md`. Record final weights, per-run curves, actual returns,
query counts, source provenance, seeds, parameters, versions and acceptance
checks. Confirm reproducibility. Leave README.md uncommitted and pause for
review before pushing.
