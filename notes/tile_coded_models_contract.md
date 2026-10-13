# Milestone 11: Tile-coded action and option models

Branch `milestone/11-tile-coded-models`, from merged master `d5be2bd` (PR #20).
Issue [#21](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/21).
Sources: local specs `05_option_models.md`, `08_representation_extensions.md`,
`10_experiment_and_testing_protocol.md`, and the Milestone 4 and 10 contracts.
This protocol is fixed before new option or model training. Planning utility
belongs to Milestone 12; this milestone measures model accuracy.

## Fixed options and representations

Train two hallway options using the unchanged Milestone 10 learner: one-hot and
the original geometry-only normalized tiles, seed **100**, **200,000** transitions,
actor/critic alpha 0.1, gamma 0.99, both lambdas zero, zero initialization and the
same behavior key schedule. Seed 100 was not used in the investigation. This
predetermined pair is not selected by performance and is not a new population
comparison of option quality. Preserve their actual softmax probabilities and
arrival stopping maps, plus weights and independently evaluated quality.

Freeze both policies throughout model learning. Model seven options in this
order: UP, DOWN, LEFT, RIGHT, existing exact shortest-path hallway option,
learned one-hot hallway option, learned tile-coded hallway option. Each fixed
policy and stopping map is identical in the two model-representation cases.
Policy lookup uses observations in the experiment runner, independently of the
features supplied to the model learner.

Compare only the existing 72-dimensional one-hot map and the 98-dimensional
Milestone 10 tile map (four active 0.5 features, width 2, four offsets). No
penalty labels, extra features, trace changes, or representation tuning.
Terminal features are zero. Save feature order and full observation tables.

## Learning and evaluation semantics

Reuse the unchanged feature-facing model update from Milestone 4:

```text
R(x) = reward_weights @ x
N(x) = successor_weights @ x
delta_R = r + gamma * (1-beta_next) * R(x_next) - R(x)
delta_N = gamma * [beta_next*x_next + (1-beta_next)*N(x_next)] - N(x)
rho = pi(action|observation) / 0.25
```

Use actual environment rewards, without the hallway subtask bonus. Stopping is
checked on arrival after a forced first action, even from stopping sources.
Terminal transitions keep immediate reward and zero the successor target.
Use existing UWT with alpha_r=alpha_p=0.1, gamma=0.99 and lambda=0. Model
weights/traces start at zero. Each behavior transition updates all seven models.

Use **100 model-training seeds 2000–2099**, paired across representations and
independent of the frozen options. Train **50,000** transitions with metrics
every **500**, including initialization. Preserve the established four-key split,
uniform behavior, and reset to S only after learning a goal transition.
Save model checkpoints at 0, 10k, 20k and 50k for later planning.

Evaluate predictions over all 72 nonterminal source states, never by comparing
feature weights directly. Report per-model learning curves and per-seed errors:

- Reward RMSE over source states.
- Native successor RMSE: sqrt(mean_source(sum_feature(error**2))). Do not divide
  by feature count. Different output feature geometries limit direct comparison.
- Common-state successor RMSE: decode predicted successor features to the same
  72 discounted terminal-state indicators before comparing against the one-hot
  exact model. With nonterminal feature matrix X (72 by F), use its double-
  precision Moore–Penrose inverse D (F by 72), checking X D = I at 1e-10.
  Decode row predictions as N D. Both fixed representations have full row rank.
  This is a measurement transform, not a learned transition probability matrix;
  do not clip or renormalize outputs or use D in training. Record its conditioning.

Report mean ± SE and paired tile-minus-one-hot differences for reward and
common-state successor errors. Keep native errors labeled separately.
The oracle and decoding transform never enter the learning update.

## Independent references and acceptance

Use the existing Python-double stochastic option oracle and independent Python
environment for both feature maps and every fixed option. Require every
reward/successor coefficient error bound <=1e-10 with at most 20,000 iterations.
Verify tile successors equal one-hot successors times X, with tolerance 1e-8.
Audit both learned hallway options using 10,000 independent sampled executions
from each of S, `(1,13)`, H and `(3,3)`, seeds 21260–21263 for one-hot policy and
21264–21267 for tiled policy. Use the same rollouts to accumulate both feature
maps, including per-feature second moments. Check reward and every successor
component against the exact models within `0.01 + 6*sample_SE`. Raise on a
10,000-action overrun; never truncate samples silently.

Implementation acceptance: hand-calculated mixed-feature updates; tabular
trajectory/parameter regression against the existing runner at atol=rtol=1e-6;
correct policy/feature separation, matrix orientation and terminal semantics;
finite parameters/metrics; zero terminal predictions; independent reference and
rollout checks; reproducible short JIT/VMAP tests and the full existing suite.

Record scientific performance checks separately from implementation checks:
each model's final mean native reward/successor errors should at least halve
initial errors. Record the existing tabular thresholds (primitive 0.01, shortest
path 0.10, learned options 0.20) for the one-hot case as diagnostics. Different
frozen policies mean these are not an exact Milestone 4 result reproduction.
No tile-win requirement. Preserve failed performance checks and explain them;
never tune seeds, settings or thresholds afterward or omit failed runs.

## Artifacts and workflow

Save full policy tables, provenance, parameters, decoder, exact references,
per-seed histories, evaluation bounds and hashed model checkpoints under
`artifacts/tile_coded_models/`. Save figures under `figures/milestone_11_*.png`
and results in `notes/milestone_11.md`. Repeat the complete experiment at one
clean implementation revision and verify byte-identical artifacts and figures.
Keep the long experiment outside PR CI; short tests use the existing workflow.
Keep README.md uncommitted. Stop for review before pushing.
