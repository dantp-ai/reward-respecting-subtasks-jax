# Milestone 12: Planning with tile-coded action and option models

Issue [#23](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/23).
Branch `milestone/12-tile-coded-planning`, from master `f49f888` (merged PR #22).
Sources: local specs `06_planning.md`, `08_representation_extensions.md`,
`10_experiment_and_testing_protocol.md`, and the Milestone 5/11 contracts.
Commit this protocol before any new planning experiments.

## Question and comparisons

Does the early option-model accuracy advantage of tiles improve actual task
returns per planning look-ahead, despite less accurate primitive models?
Keep Milestone 11's frozen policies and representations unchanged. Compare four
sets, ordered UP, DOWN, LEFT, RIGHT, optionally followed by shortest-path,
learned one-hot hallway, or learned tile-coded hallway option. All configurations
share their representation's primitive models at each model-training checkpoint.

For each of the two representations, evaluate exact models of those same frozen
policies plus learned checkpoints at 0, 10k, 20k and 50k model transitions.
This gives 40 cases: two representations, five sources, four option sets.
The exact models isolate representation-dependent planning updates from model
approximation. They do not replace learned options with optimal options.

## Inputs and exact controls

Validate Milestone 11's source report, frozen policies and all eight compressed
checkpoints: hashes, schema, model/observation/feature order, seeds 2000–2099,
gamma and learning parameters, checkpoint steps, dimensions, finiteness and
terminal features. When the source report is available, validate every present
declared file before reporting missing inputs. An absent report makes its
artifact manifest unavailable and fails explicitly for a partial directory.
Regenerate the unchanged Milestone 11 experiment automatically only when its
source directory has no artifacts; partial or corrupt inputs fail explicitly.
Record source report and artifact hashes and revisions in the new report.

Recompute exact reward and native successor tables with the independent Python
environment/oracle and compare against saved references at 1e-8. Realize these
tables as feature-facing linear models using X's double-precision pseudoinverse
D: reward coefficients R D.T and successor matrices N.T D.T, with X containing
nonterminal source rows. Check reconstruction within 1e-10 before casting and
1e-5 in float32. This fitting is only for exact-model controls; learned model
coefficients are loaded unchanged. Full row rank permits exact realization.

Tile-model coefficients are not transition probabilities. Negative coefficients
or a sum different from gamma are not grounds for rejection. Do not clip,
normalize, project or decode the learned models for planning. The common-state
decoder from Milestone 11 was a measurement device, not the planning interface.

## Backups, budgets and execution

Use the existing Equation 19 feature-facing semi-gradient update unchanged:

```text
q[o] = R_o(x) + w @ N_o(x)
w = w + alpha * (max(q) - w @ x) * x
```

Successors already contain gamma**K: no extra gamma or subtask stopping bonus.
Zero initialization, alpha=1 for both unit-norm feature maps, gamma=0.99. The
planner receives feature vectors and generic models, never option-specific
logic. Changing sampled-state features changes the update's generalization.

Use 100 planning seeds **12000–12099**, paired by index with model seeds
2000–2099. Sample uniformly from the 72 nonterminal observations with the
existing `sample_states` key schedule. Reuse each seed's common prefix across
all cases. Use **20,000 look-aheads**, with metrics every **100**, including zero.
One state-model backup costs one query: four queries per primitive-only update,
five with an option. Record update and query counts separately.

Measure state-space value RMSE and planned start value at every checkpoint.
Independently evaluate actual main-task returns at look-aheads
**0, 2k, 5k, 10k, 20k** using Appendix B semantics: reselect the greedy model
at every primitive step, then draw one action from its frozen policy. Preserve
first-maximum ties and action order. This is not committed option execution.
Use the existing Python-double discounted policy solve on the true environment,
including reachable cycles; require residual/(1-gamma) <=1e-8. Diagnostic queries
are counted separately and excluded from the planning-budget axis.

## Reporting and acceptance

Report every seed and all signed returns, model-training durations, final value
errors, and paired tile-minus-one-hot differences. Report first reaching 95% of
optimal planned value and actual return separately, with actual-return hits
limited to the declared evaluation grid. Missing hits stay explicit: do not
average only successful runs. The main-task optimum comes from independent
value iteration (start value 0.99**17), not from the subtask bonus.

Implementation checks: mixed-feature manual backups and successor orientation;
tabular runner regression and independent exact baseline; policy reselection;
terminal zero; query accounting; reproducible JIT/VMAP; input validation and
corruption/missing-input tests; independent return bounds. Run the full suite.

Record numerical/performance checks separately, without requiring tiles to win:
- Every planning trajectory and backup used to select a policy is finite.
- For each exact case, final maximum state-value error and actual return error
  should each be <=1e-5 for every run. Failure of tiled planning is a result,
  not a reason to change alpha or budget afterward.
- For each learned-50k case, final mean actual return and mean planned start
  should be within 0.05 of optimal.
- For each learned hallway option and representation, final actual mean return
  using 50k models should be no worse than with 10k models by more than 0.01.

Retain instability, negative returns, overestimation and failures. Never evaluate
a policy selected from nonfinite backups as if it were valid. Retain such seeds
with explicit failure messages and unavailable metrics; any aggregate requiring
a missing metric is unavailable. Save nonfinite parameter values explicitly.
Do not tune settings or silently omit failures. No learned-model stability or
contraction is assumed merely because the tile feature matrix has full rank.

## Artifacts and review

Save per-seed curves, evaluation-grid weights and model selections, query counts,
provenance and checks under `artifacts/tile_coded_planning/`. Save figures as
`figures/milestone_12_*.png` and findings in `notes/milestone_12.md`. Repeat the
complete experiment at one clean implementation revision; verify byte-identical
reports, saved parameters and figures. Keep long runs outside PR CI. Leave
README.md uncommitted and stop for review before pushing implementation changes.
