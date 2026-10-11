# Milestone 10: Tile-coded hallway-option learning

Proposed implementation contract, prepared before any extension training.
Issue [#19](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/19).
Branch `milestone/10-tile-coded-options`, from merged master `a0d0af6` (PR #18).
Sources: local specs `08_representation_extensions.md`, `01_architecture.md`,
`04_reward_respecting_subtasks_and_options.md` and
`10_experiment_and_testing_protocol.md`. The specs exist locally under ignored
`specs/`; this tracked contract makes the proposed scope reviewable.

## Scope and question

Stage 8 starts with tile coding. This first milestone compares one-hot and
tile-coded actor/critic inputs on the existing deterministic two-room hallway
subtask. Does feature sharing change subtask-value accuracy, actual option
return, and safe hallway attainment under the same experience budget?

Use the Milestone 3 environment and option-learning protocol as the paired
baseline. The four-room reproduction remains a separate validated experiment.
The later Stage 8 milestones will extend option models and planning to the new
representation, then study controlled aliasing and traces. This milestone
does not claim to complete Stage 8's full model/planning comparison.

## Representation and target semantics

Expose a small functional interface, conceptually
`features = representation(params, observation)`. Reuse the current one-hot
implementation and feature-facing actor/critic. No framework or inheritance
hierarchy is needed. Actor, critic, and their traces use the selected feature
dimension; observation IDs must not enter the learning equations.

The proposed tile map uses four coordinate tilings, tile width `(2, 2)`, and
offsets `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`, in that order. For each tiling,
compute `floor((observation - offset) / width)` independently per coordinate.
Within each tiling, sort the distinct tile-coordinate pairs occupied by legal
nonterminal observations lexicographically; concatenate those four blocks.
Build this fixed index from the environment geometry alone, without values,
policies, rewards, or training results. Do not hash or merge different blocks.

Every legal nonterminal observation activates one feature per tiling, each of
value `1/sqrt(4) = 0.5`. Thus its squared feature norm is one, as for the one-hot
baseline. There is no bias feature. The goal and observations outside the legal
nonterminal set map to zeros. Freeze feature ordering before training and export
the tile parameters, tile coordinates, state-feature table, and feature count.
Measure distinct feature vectors and overlaps; do not assume compression or
claim that tile coding necessarily produces fewer features than one-hot.

Keep the target separate from the prediction representation:

```text
h(observation) = 1 at hallway (3, 7), otherwise 0
main-task value estimate = 0
stopping value z(observation) = h(observation)
```

The hallway signal stays binary and has bonus one for both representations.
It is not a tile index, a tile activation, or an extra actor/critic input.
This is the same fixed stopping-value function as Milestone 3, where zero
main-task weights reduce Equation 4 to the hallway indicator. Retaining that
function isolates the prediction representation; it is not an experiment in
attaining arbitrary tile features or replacing Equation 4 for nonzero main
weights. Terminal arrival overrides the stopping value to zero.

## Proposed paired experiment

- Representations: existing one-hot and the single tile map above.
- Seeds 0–99 for both; 50,000 behavior transitions per seed; metrics every 500
  transitions, including zero initialization.
- Uniform requested-action probabilities 0.25, gamma 0.99, actor and critic
  step sizes 0.1, both trace lambdas zero, zero initial weights and traces.
- Preserve the Milestone 3 key schedule: `key(seed)`, then
  `split(next, action, environment, reset)` on every transition.
- Pair identical behavior trajectories by seed. Update on goal arrival before
  resetting to S; option stopping never resets the behavior environment.
- Preserve Equations 5, 9, 10, 11 and UWT. In particular, learned stopping uses
  `z >= predicted_value`, not the exact oracle's discounted comparison.

Evaluate predictions as `feature_matrix @ critic_weights` over the same 72
nonterminal states, rather than interpreting weights as state values. Report
critic RMSE against the unchanged exact subtask oracle and predicted start
value at every checkpoint. At the final checkpoint, independently evaluate the
actual softmax policy and learned stopping map in the existing Python reference
environment, with value-error bound and safe-success probability interval width
at most 1e-8. Retain each run's actual subtask return, safe hallway probability,
stopping map, and weights. Also report means and standard errors of paired
tile-minus-one-hot differences; keep predicted and actual values distinct.

## Acceptance and verification

1. Independent coordinate examples verify tile boundaries, negative tile
   coordinates caused by offsets, distinct tiling blocks, ordering, overlap,
   normalization, terminal masking, and invalid-observation masking. Eager,
   JIT, and VMAP feature outputs agree exactly for integer grid observations.
2. The one-hot path reproduces the existing Milestone 3 runner on short seeded
   trajectories: identical environment states, keys, actions and reset timing;
   all weights/traces and predicted values agree within `atol=rtol=1e-6`.
   Its full experiment must pass the unchanged Milestone 3 acceptance checks.
3. The hallway signal and terminal overrides agree for both representations at
   every legal state. A hand-calculated overlapping-feature transition checks
   the unchanged actor/critic update, including a terminal reward transition.
4. All weights, predictions and reported metrics are finite, terminal predicted
   values are zero, and each independent evaluation meets its error bounds.
   Preserve and diagnose failures; do not silently omit seeds or change the
   tile map, normalization, learning rates, budgets or thresholds afterward.
5. Tile coding need not beat the tabular control to complete this research
   milestone. Report degradation, variance, and termination changes explicitly.
6. Repeat the full paired experiment at one clean implementation revision:
   reports, saved parameters and figures must be byte-identical in this software
   environment. Record feature metadata, seeds, parameters and package versions.

Short mathematical and integration tests run in the existing PR workflow; the
full paired experiment stays outside CI. Save figures under
`figures/milestone_10_*.png`, ignored artifacts under
`artifacts/tile_coded_options/`, and the results in `notes/milestone_10.md`.
Commit this contract before training. Keep README.md uncommitted and pause for
the user's review before pushing implementation changes.
