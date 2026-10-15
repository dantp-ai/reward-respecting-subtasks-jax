# Milestone 12: Planning with tile-coded models

Issue [#23](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/23),
branch `milestone/12-tile-coded-planning`, from merged master `f49f888` (PR #22).
The [contract](tile_coded_planning_contract.md) was committed as `7904aff`
before experiments. Completed experiment and independent audit revision:
`85bab9b18e4f335c2aa419155c378d054dd9a35b`.

## Result

**Tile-coded planning is unstable under the frozen settings, even with exact
models.** Its value estimates grow enormously while actual task returns become
strongly negative. One-hot planning reaches the optimum, 0.842943, with exact
models and the final learned models. The implementation and experiment are
complete, but the declared performance targets are not all met: 58 of 76
numerical/performance checks pass and 18 fail, all for tiles. All 265 tests pass.

An independent double-precision replay in state space reproduces the growth.
Model-learning error alone therefore cannot explain the failure. No learning
rate, representation, seed, budget or acceptance threshold was changed after
seeing the results. Failures remain in the report and figures.

## Frozen comparison

There are 40 cases: two representations, five model sources (exact and learned
at 0, 10k, 20k and 50k transitions), and four model sets. Each set contains the
four primitive actions, optionally augmented with the shortest-path hallway
option, learned one-hot hallway option, or learned tile hallway option. The
same Milestone 11 frozen policies are modeled in both representations.

Each case uses 100 planning seeds 12000–12099, paired with model seeds
2000–2099. Uniform sampled states share each seed's prefix across cases.
Zero-initialized planning uses the unchanged Equation 19 update, alpha=1,
gamma=0.99 and unit-norm features: 72 one-hot features or 98 tile features.
Learned models are loaded unchanged; their coefficients are neither clipped
nor interpreted as transition probabilities. Exact controls realize independent
oracle predictions as linear models in each representation.

The budget is 20,000 model look-aheads: 5,000 primitive-only updates or 4,000
updates with an option. Value metrics are recorded every 100 look-aheads.
Actual returns are independently solved at 0, 2k, 5k, 10k and 20k look-aheads,
giving 4,000 planning trajectories and 20,000 return evaluations. Execution
reselects the greedy model at every primitive step, then samples one action
from its frozen policy. It does not commit to an option. Diagnostic queries
are recorded separately and excluded from the planning budget.

## Exact-model results

Final start values and actual returns, mean ± standard error across 100 seeds:

| Representation | Added option | Planned value | Actual return |
| --- | --- | ---: | ---: |
| One-hot | None | 0.842943 ± 0 | 0.842943 ± 0 |
| One-hot | Shortest path | 0.842943 ± 0 | 0.842943 ± 0 |
| One-hot | Learned one-hot | 0.842943 ± 0 | 0.842943 ± 0 |
| One-hot | Learned tiles | 0.842943 ± 0 | 0.842943 ± 0 |
| Tiles | None | 4.8924e10 ± 1.5387e10 | -83.1186 ± 3.1206 |
| Tiles | Shortest path | 9.8009e8 ± 5.3751e8 | -77.7410 ± 3.7136 |
| Tiles | Learned one-hot | 2.6451e8 ± 7.0242e7 | -74.1479 ± 3.8855 |
| Tiles | Learned tiles | 2.8717e8 ± 7.1233e7 | -72.7565 ± 3.8587 |

One-hot benefits from the learned hallway options early: at 2,000 look-aheads,
mean actual return is 0 with primitives alone, 0.781587 with the learned
one-hot option and 0.780549 with the learned tile option. Mean first reaching
95% of optimal planned value is 4,515, 4,243, 1,857 and 1,864 look-aheads for
the four sets respectively. All 100 runs in every one-hot exact case reach
95% of optimal actual return on the evaluation grid by the final budget.

Tile estimates cross the planned-value threshold even earlier (1,084, 1,135,
738 and 748 look-aheads), but this is overestimation, not successful planning.
**Zero of 100 runs in each tile exact case reaches the actual-return threshold
at any evaluated budget.** Missing hits remain explicit rather than being
averaged over a successful subset.

## Learned-model maturity

Mean actual return after 20,000 planning look-aheads:

| Representation | Model transitions | Primitives | + Shortest path | + Learned one-hot | + Learned tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-hot | 0 | 0 | 0 | 0 | 0 |
| One-hot | 10,000 | 0.841111 | 0.841111 | 0.841111 | 0.841111 |
| One-hot | 20,000 | 0.842943 | 0.842943 | 0.842943 | 0.842943 |
| One-hot | 50,000 | 0.842943 | 0.842943 | 0.842943 | 0.842943 |
| Tiles | 0 | 0 | 0 | 0 | 0 |
| Tiles | 10,000 | -77.565106 | -70.987463 | -65.543663 | -65.548203 |
| Tiles | 20,000 | -79.045201 | -75.307528 | -72.670354 | -73.113012 |
| Tiles | 50,000 | -84.572678 | -69.918755 | -71.164312 | -75.144732 |

At 50k model transitions, one-hot planned start values are approximately
0.842941. Tile planned values are 3.5848e11, 5.5837e9, 2.6357e9 and 2.2427e9
for the four sets. Better model accuracy does not ensure better planning with
these updates. In particular, the earlier tile option-model accuracy advantage
from Milestone 11 does not produce useful planning in this comparison.

The complete report includes every seed, standard errors, paired tile-minus-
one-hot differences, state-value RMSE, selected models and evaluation weights.
Uncertainty here is conditional on Milestone 11's single frozen policy pair;
it does not measure variation across independently trained option pairs.

## Independent numerical audit

After observing the failure, a descriptive audit replays the predetermined first
planning seed, 12000, with exact primitive models in pure NumPy double precision.
It operates directly on the 72 state values and independent exact state-space
reward/successor tables, without fitted feature-model coefficients or JAX
planning. For feature table X and sampled state s, it computes:

```text
delta = max_o(R[o,s] + P[o,s] @ V) - V[s]
V = V + (X @ X.T)[:,s] * delta
```

This is the state-value form of the frozen alpha=1 feature update. P already
contains discounting. With one-hot features, the update changes only the
sampled state's value. With shared tiles, it also changes other states.

| Look-aheads | One-hot double start value | Tile double start value |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 2,000 | 0 | 8.863305 |
| 5,000 | 0.842943 | 267.264088 |
| 10,000 | 0.842943 | 104,395.233023 |
| 20,000 | 0.842943 | 6,483,352,545.117965 |

The saved JAX tile start value at the final budget is 6,483,146,624. The maximum
state discrepancy divided by the double replay's maximum absolute state value
is 3.23e-5. One-hot maximum absolute discrepancy is 1.58e-7. Thus substantial
growth persists without float32 planning or pseudoinverse model construction.
This supports instability in the shared-feature update under these settings;
it does not establish that all tile-coded planners or step sizes fail.
The audit is a post-run diagnostic, not an additional predeclared acceptance
test or a parameter search.

## Validation and reproducibility

- **265 tests pass**, including mixed-feature manual backups, successor
  orientation, old one-hot runner regression, query accounting, JIT/VMAP,
  policy reselection, missing/nonfinite metrics, input corruption, exact
  reconstruction and the independent state-space recurrence. Pre-commit passes.
- All trajectories and policy-selection backups are finite, and terminal
  predictions remain zero. Finiteness alone does not imply useful planning.
- All 20,000 actual-return evaluations resolve. The largest independent
  residual-based error bound is 3.77e-12, below the required 1e-8. Signed
  returns range from approximately -100 to 0.842943; none are clipped.
- Exact tile-model reconstruction error is 8.66e-15 in double precision and
  4.77e-7 after float32 conversion, within the frozen limits. Both feature
  matrices have row rank 72. Independently recomputed native references agree
  with projected references within 1.62e-12 (successors) and 8.54e-13 (rewards).
- Of 76 numerical/performance checks, **18 fail**: eight exact tile value/return
  checks, eight final learned tile value/return checks, and two tile hallway
  model-maturity checks. All one-hot checks and all three evaluation checks pass.
- Two complete runs at clean revision `85bab9b` produce byte-identical outputs:
  40 case reports, 40 weight archives, the combined report, the independent
  audit and three figures (85 files). The initial run at `597d698` has identical
  case reports/weights; only the combined report's revision metadata differs.
- A subsequent display-only change sets the signed logarithmic axis before
  plotting, so autoscaling avoids unused negative decades. Final figures were
  regenerated twice from the unchanged report and are byte-identical. There
  were no changes to experimental settings or numerical results.

## Reproduce and inspect

```sh
uv sync --locked
uv run --locked --offline --no-cache pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked --offline --no-cache python -m rrs.experiments.tile_coded_planning
uv run --locked --offline --no-cache python -m rrs.experiments.tile_coded_planning_audit
```

An empty Milestone 11 artifact directory triggers its unchanged experiment.
A partial or corrupt directory fails explicitly. Source report and checkpoint
hashes, software versions, seeds, revision and tracked-worktree status are
recorded in `artifacts/tile_coded_planning/planning.json`. Local artifacts are
ignored by Git; the notes and figures are committed.

- [Exact-model planning](../figures/milestone_12_exact_models.png)
- [Final learned-model planning](../figures/milestone_12_learned_models.png)
- [Model maturity and actual returns](../figures/milestone_12_model_maturity.png)

Environment: Python 3.12.9, JAX/jaxlib 0.7.0 on CPU, NumPy 2.3.2 and Matplotlib
3.10.5. Planning uses float32; decoded metrics, independent policy evaluation,
exact reference construction and the state-space audit use double precision.

SHA-256 values:

- M11 source report: `77e790db302e81886fd3e0726b2f1d64a148fbe3b24c20c99417a7aa24c8d5ae`
- Combined planning report: `0a81e1465dc7b7140ab834caaf466341a753194bd9668a43afc501c7b62c3e41`
- State-space audit: `36ae960be6129c2e3fa1dd04facb0c4c13481d4485690ebd317463d4815e6464`
- Exact figure: `6db75dc581295c22b3d8e9e9873b5e158dc9e713793b4a4c1533a9d2599e85e9`
- Learned figure: `2d82efd8304950098a404796743153d667e39b1f32b7bdd909f327fa2a6fed70`
- Maturity figure: `a01ecb81aae798aa9a2c2da086d0589dec52e9f8c85c017f2b1e6d429d1d0812`

## Next investigation

Investigate planning stability before extending the representation further.
Freeze a new protocol that separates step-size effects from shared-feature
update effects, using exact primitive models first and preserving this failed
alpha=1 run as the baseline. Assess actual return alongside value estimates.
This milestone has not tested whether smaller step sizes or a different
planning update restore stable, useful tile-coded planning.
