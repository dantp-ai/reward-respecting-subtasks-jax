# Four-room planning with actions and learned options

Milestone 9, [issue #17](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/17),
branch `milestone/09-four-room-planning` from master `535741b`.
Sources: `notes/planning_contract.md`, specs `06_planning.md`,
`07_four_room_reproduction.md` and `10_experiment_and_testing_protocol.md`, and
paper Equations 18–20, Section 7 and Appendix B. Build on the validated
[four-room models](four_room_models_contract.md). No paper baselines requiring
external implementations are in scope.

## Methods and model sources

Compare two model sets: four primitive actions, and those same four actions plus
the four H1–H4 reward-respecting options. Every model is a feature-facing linear
expectation model over the 103 nonterminal one-hot features. A query evaluates
one model at one state. A complete action-only update therefore costs four
look-aheads; an update with options costs eight.

Use three exact references:

1. Exact stochastic primitive models from the independent four-room environment.
2. Exact action models plus exact stochastic models of each optimized Equation 2
   hallway policy and stopping rule.
3. Exact action models plus exact stochastic models of the frozen Milestone 7
   seed-7000 policies actually used to train the Milestone 8 models.

Then repeat both action-only and action-plus-option comparisons at all six
Milestone 8 checkpoints: 0, 10,000, 20,000, 30,000, 40,000 and 200,000 model
transitions. Planning runs pair seeds 3000–3029 with model-training seeds
8000–8029 by run index. Exact references are common across runs. This separates
the value of option quality from error in its learned model, while preserving
the same action-model checkpoint and planning-state prefix for each pair.

Validate the ignored Milestone 8 report, each checkpoint SHA-256, all array axes,
seeds, feature/state order, gamma, finiteness, terminal behavior, nonnegative
successor mass and the link to the frozen policy export. If the default Milestone
8 artifacts are absent, regenerate them with the unchanged canonical command.
Fail on corrupt existing or explicitly supplied inputs; do not silently replace
them with new policies or models.

## Planner and actual policy evaluation

Use the existing asynchronous Equation 18 update in Python-double references
and the existing feature-facing Equation 19 semi-gradient planner. Initialize
values and weights at zero, alpha=1, gamma=0.99. Sample nonterminal states
uniformly with replacement. For each seed, split the planning PRNG key into
next and state keys per update and sample with `randint(state_key, (), 0, 103)`.
Use one common state sequence and its prefixes for every method and maturity
checkpoint. Select the first maximum in model order: UP, DOWN, LEFT, RIGHT,
H1, H2, H3, H4. Keep terminal features and values zero.

Freeze a budget of 1,600,000 look-aheads for every case. Save planned start value
and full state-value RMSE every 1,600 queries. This is 400 planning updates for
the four-action case and 200 for the eight-model case. Save the complete per-run
weight vectors at each checkpoint.

At 0, 200,000, 400,000, 600,000 and 1,600,000 look-aheads, independently evaluate
the policy induced by the current planning weights in the stochastic four-room
environment. At every primitive step, reselect the maximizing model, draw one
primitive action from that model's policy, then reselect in the next state.
Primitive policies are deterministic; the frozen hallway policies retain their
stochastic action probabilities. Do not commit to executing an option to its
termination during this Appendix B policy-return evaluation. A stochastic
Python-double linear solve includes reachable cycles and has residual/(1-gamma)
<=1e-8. Do not truncate rollouts or clip signed returns. Report planned values
and actual returns separately; option-model backups can overestimate the value
of this reselected policy because their targets assume option continuation.
Policy queries are diagnostics and are excluded from the planning-work axis.

## Frozen experiment

This protocol is committed before running planning comparisons:

- 30 planning seeds 3000–3029 and paired model seeds 8000–8029.
- 1,600,000 look-aheads per case; 1,600-query metric checkpoints.
- Actual policy returns at look-ahead budgets 0, 200,000, 400,000, 600,000 and
  1,600,000.
- Measure RMSE against the exact stochastic four-room primitive task over all
  103 nonterminal states. Report per-run curves and means ± standard errors.
- For exact primitive-only planning, every final state value and independently
  evaluated start return must be within `1e-5` of the exact main-task solution.
  Every value, weight and reported return must be finite; terminal values remain
  zero, query totals equal the frozen budget, and all return-solver bounds pass.
- Preserve planned-value errors, actual returns and first look-ahead reaching
  95% of the main-task optimal start value for every case and run. Report how
  often each case reaches that value and its mean first-hit budget among runs
  that reach it. Missing hits remain explicit; never drop them from summaries.

The speed comparison is a scientific result, not a pass/fail condition: report
the measured first-hit budgets for exact optimized options, exact frozen options,
the exact primitive baseline and every learned-model checkpoint. The milestone
measures whether options improve planning and actual execution; it does not
promise that all learned models will improve either quantity. Preserve negative
returns, missing thresholds and model-induced overestimation. No seeds, budgets,
thresholds, model checkpoints, policies or tie rules may change after results
are observed.

## Tests and artifacts

Retain all two-room planning tests. Add hand-calculated four-room sparse-stochastic
policy-return tests, including mixed continuing/terminal outcomes, nonterminal
cycles, terminal rewards and stochastic option actions. Test four-vs-eight
look-ahead counts, common state prefixes, asynchronous update/eager/JIT agreement,
tie order, checkpoint integrity, source pairing and reproducible short runs.
The existing PR workflow runs short tests; full planning runs remain outside CI.

Save results in `notes/milestone_09.md`, figures as
`figures/milestone_09_*.png`, and ignored JSON/checkpoint artifacts under
`artifacts/four_room_planning/`. Reports record per-run values/returns, look-ahead
and diagnostic query counts, paired seeds, per-checkpoint source hashes,
equation/configuration details, code revision and software versions. Repeat the
full experiment and require byte-identical reports, checkpoints and figures in
this environment. Keep README.md uncommitted. Commit locally and pause for review
before pushing.
