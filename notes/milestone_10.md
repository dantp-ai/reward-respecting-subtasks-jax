# Milestone 10: Tile-coded hallway-option learning

Issue [#19](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/19),
branch `milestone/10-tile-coded-options`, from master `a0d0af6` (merged PR #18).
The [contract](tile_coding_contract.md) was proposed in commit `0e6094b` and
accepted before the experiment. Implementation revision:
`198c37253fff547284cf65181a7525ae6cbdf45a`.

## Reproduce

```sh
uv sync --locked
uv run --locked --offline --no-cache pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked --offline --no-cache python -m rrs.experiments.tile_coded_options
```

The command trains and evaluates both representations, writes
`artifacts/tile_coded_options/learning.json` and two compressed parameter
archives, and saves these figures:

- [Learning curves and actual option returns](../figures/milestone_10_option_learning.png)
- [Seed-0 policies, stopping maps and paired safe attainment](../figures/milestone_10_option_policies.png)

Optional `--output-dir` and `--figures-dir` arguments select the destinations.
Artifacts preserve all critic checkpoints, final actor/critic weights and traces,
environment positions, keys, episode counts, feature geometry, feature ordering,
state-feature tables, per-run metrics and independent evaluation bounds.
Evaluation failures remain explicit; an unresolved seed makes the corresponding
aggregate unavailable rather than being excluded from the mean.

## Experiment and results

This is the first part of Stage 8's representation extension, using the existing
two-room task and Milestone 3 learning equations. It tests prediction-feature
sharing while holding the hallway stopping signal fixed. It is not a paper
figure reproduction or a comparison of learned option models and planning yet.

Both methods use seeds 0–99, 50,000 uniform-behavior transitions, metrics every
500 transitions, gamma 0.99, actor/critic step sizes 0.1, zero trace lambdas and
zero initial weights. The same PRNG schedule pairs the experience. Hallway
attainment at `(3, 7)` supplies the same binary bonus one, with zero main-task
weights. The signal is separate from the actor/critic input, and termination
retains the original destination-state semantics.

One-hot has 72 features. Tile coding has **98 features**, four tilings of width
`(2, 2)` and offsets `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`. Four active
features of 0.5 give squared norm one. There are 72 distinct nonterminal feature
vectors and 221 unordered pairs of distinct states sharing at least one feature.
The representation generalizes through shared features; it is not a smaller
state encoding. Terminal and invalid observations have zero features.

| Final metric | One-hot, mean ± SE | Tile coding, mean ± SE | Paired tile-minus-one-hot, mean ± SE |
| --- | ---: | ---: | ---: |
| Critic RMSE over 72 states | 0.238545 ± 0.005171 | 0.305880 ± 0.002297 | +0.067335 ± 0.005532 |
| Predicted subtask value from S | 0.813672 ± 0.000961 | 0.247488 ± 0.010409 | -0.566184 ± 0.010198 |
| Actual stochastic option return from S | 0.820896 ± 0.000275 | 0.192093 ± 0.009200 | -0.628803 ± 0.009072 |
| Safe hallway attainment probability | 0.951849 ± 0.000172 | 0.538755 ± 0.017013 | -0.413094 ± 0.016972 |

The exact subtask start value is 0.895338254. Actual option returns include the
subtask stopping bonus and are distinct from main-task planning returns.
Each standard error in the final column is computed from the 100 within-seed
differences, accounting for pairing rather than treating the two groups as
independent.

The one-hot result reproduces the recorded Milestone 3 metrics. Tile coding
reduces intermediate value RMSE during much of training, but finishes with worse
value accuracy and much worse actual option behavior under the fixed settings.
Every one of the 100 paired runs has lower safe attainment with tile coding.
Its individual safe probabilities range from 0.003640 to 0.875895. The fixed
seed-0 plot also shows an extra stopping state at `(2, 1)`, a non-penalty cell,
relative to the one-hot policy. That illustration does not establish the cause
of degradation for every seed.

## Interpretation and follow-up diagnostic

The subsequent [controlled investigation](tile_coding_investigation.md) tests
training budget, step sizes, actor/critic sharing, stopping maps and penalty
boundaries. It finds recovery to 99.269% safe attainment at 200k transitions,
with lower discounted return than one-hot, and localizes much of the early
difficulty to policy generalization. These follow-up results supplement the
frozen 50k comparison recorded here.

This experiment supports a narrow finding: the prescribed feature sharing is
harmful to option quality at this learning rate and finite data budget. It does
not show that tile coding is generally worse, or that it cannot eventually
learn the task. Unit feature norms preserve the scale of an individual update
at its source state, but they do not remove updates' effects on neighboring
states. No tilings, step sizes, seeds, budgets or acceptance thresholds were
changed after seeing these results.

After the frozen experiment, an additional capacity diagnostic found that the
72-by-98 nonterminal tile-feature matrix has row rank 72. A least-squares fit to
the exact subtask values has RMSE `9.44e-16` in NumPy double precision. Thus the
observed critic error is not forced by an inability to represent that value
function. This diagnostic does not identify which learning dynamics caused the
poor policy, and the fitted weights were never used for training or evaluation.
It was not a predeclared acceptance check. Reproduce it with:

```python
import json
import numpy as np
from rrs.experiments.hallway import build_baselines

with open("artifacts/tile_coded_options/learning.json") as stream:
    report = json.load(stream)
data = build_baselines()
indices = list(data.nonterminal_indices)
x = np.array(report["cases"]["tile_coding"]["representation"]["state_features"])[indices]
y = np.array(data.solutions["reward_respecting"].values)[indices]
w, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
print(rank, np.sqrt(np.mean((x @ w - y) ** 2)))
```

Later representation milestones can use the saved policies to study model
learning and planning. These results should also inform a separately specified
learning-dynamics diagnostic; extending the budget or changing step sizes would
be a new experiment, not a replacement for this result.

## Validation and repeatability

All **235 tests pass**, including 21 added for this milestone. They cover
independent tile-coordinate examples, negative tile indices, exclusion masks,
normalization and JAX transformations; tabular trajectory/weight/trace
regression; target invariance; hand-calculated shared-feature and terminal
updates; seeded training; state-space reporting; paired standard errors; failed
evaluation retention; and deterministic parameter serialization. Pre-commit
checks pass on all changed files. The existing PR workflow runs the short suite.

All 12 report acceptance checks passed. Both methods' weights and metrics were
finite and terminal predictions stayed zero. All 200 policies were independently
evaluated with value-error bounds and safe-success probability interval widths
at most 1e-8. The one-hot control passed every original Milestone 3 threshold.
As frozen in the contract, an improvement over one-hot was not required for the
tile experiment to be a valid result.

Two full runs at the clean implementation revision produced byte-identical JSON,
both gzip archives and both PNGs. SHA-256 values:

- Report: `2a0e7c460275b468c61f7383bead4e7a4fa4fafb114297e912f19640293856f7`
- One-hot parameters: `4ea541cecf7ecd0360041e89907fb126be9998952ffd1c44d9cfcba3f1c2fc74`
- Tile parameters: `030eb12d9ec19e81700f9655aea722b755ab2a7d6860eb75b75daabf2d19a119`
- Learning figure: `c1444c7fb264dc146ef4a85a3d7e340507130c8c6be01dee6396f6fe19cbb90d`
- Policy figure: `490783d1ec5c0dd62a4b94c32543832b62d247bdd1ddf0235589f24c34b395d9`

Environment: Python 3.12.9, JAX/jaxlib 0.7.0, Matplotlib 3.10.5, NumPy 2.3.2.
Reports and parameter archives record code revision and tracked worktree state;
running at a later revision changes their provenance and hence their hashes.

| Concept | Implementation |
| --- | --- |
| Geometry-only normalized feature map | `rrs/representations/tile_coding.py` |
| Fixed hallway signal and paired experience | `rrs/experiments/tile_coded_options.py` |
| Unchanged Equations 5, 9–11 and UWT | `rrs/rl/learning.py` |
| State-space metrics and paired comparisons | `rrs/experiments/tile_coded_option_report.py` |
| Independent option evaluation | `rrs/rl/policy_evaluation.py` |
| Milestone figures | `rrs/experiments/tile_coded_option_plots.py` |
