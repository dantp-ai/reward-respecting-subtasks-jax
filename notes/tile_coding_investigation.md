# Milestone 10 investigation: Why did tile coding underperform?

Requested after the frozen [Milestone 10 result](milestone_10.md).
This is a separate diagnostic study on the same milestone branch. Preserve the
original experiment, its parameters and artifacts. The original result does not
establish that tile coding is generally inferior, nor does tile coding guarantee
improvement on every discrete task.

## Diagnostic protocol, specified before the new runs

Use all 100 original seeds, paired by index. Do not select favorable seeds or
replace the original 50,000-step results. Keep the environment, behavior stream,
hallway signal, gamma, trace lambdas and initialization unchanged. Report every
declared comparison, including negative results. These are exploratory
diagnostics prompted by the existing result, not independent confirmation of a
configuration chosen without seeing data.

1. **Implementation/scaling audit.** Independently enumerate tile coordinates;
   check the terminal mask, feature normalization, training/evaluation features
   and exact value representability. Run binary tile activations with both
   step sizes 0.025 for 50,000 steps. Compare its predictions and preferences
   against the original activations of 0.5 and step sizes 0.1. With four tiles,
   these parameterizations should give the same functional updates. Verify a
   hand-calculated update and short-run actor/critic equivalence first.
2. **Evaluate the evaluator.** Independently simulate 10,000 option executions
   from S for predetermined seed 0 of each original representation, using the
   Python environment and RNG seed 10000. Force the first action, apply stopping
   on arrival, preserve terminal rewards, and track penalty encounters. Check
   return and safe-success estimates against the exact evaluator using
   `0.001 + 6 * sample_standard_error`. A 10,000-action overrun is an error,
   not a truncated sample.
3. **Stopping versus policy at evaluation.** Evaluate all four combinations of
   original one-hot/tile action policies and stopping maps for every seed.
   This isolates effects of final stopping maps while holding policies fixed;
   it does not undo effects the critic had while learning the actor.
4. **Actor versus critic during learning.** Train one-hot actor + tiled critic,
   and tiled actor + one-hot critic, for 50,000 transitions at step sizes 0.1.
   Use the same update equations with separate actor and critic feature vectors.
   Validate the diagnostic runner against the unchanged original runner when
   their representations coincide. This tests which component's generalization
   accounts for most of the gap under the fixed budget.
5. **Learning budget.** Train both original representations to 200,000 steps,
   retaining full actor/critic snapshots at 50k, 100k, 150k and 200k. Compare all
   those budgets, not just the final one, with both step sizes 0.1.
6. **Step-size sensitivity.** For both representations, use joint actor/critic
   step sizes 0.025, 0.05 and 0.2 for 50,000 steps. The original 0.1 cases form
   the fourth point. Report the full paired grid without promoting its winner
   as a new validated default.
7. **Reward-boundary sharing.** At step sizes 0.1 and 50,000 steps, train a tile
   representation whose feature blocks are split into penalty and non-penalty
   cells. Each state retains four active 0.5 features; only cross-boundary
   sharing is removed. This intervention uses extra environment information,
   so an improvement would diagnose the chosen generalization, not establish
   universal superiority of tile coding.

The diagnostic learner continues to consume feature vectors. A finite
observation/feature table belongs only to the experiment runner, not to any
learning equation. Compare complete short trajectories, weights and predictions
with the original implementations at `atol=rtol=1e-6`; preserve larger
discrepancies for diagnosis. The binary scaling equivalence has the same
tolerance. Exact evaluator error bounds remain at most 1e-8; no silent dropping
of failed runs is permitted.

Save all per-seed results, configuration metadata, source hashes, feature tables
and final learned parameters to `artifacts/tile_coding_investigation/`, and any
figures to `figures/milestone_10_investigation_*.png`. Add findings to this note,
including causal limits and any protocol deviations. Never use an exact fitted
value function or oracle stopping rule to train the experimental learners.

## Evaluation-budget amendment

The first diagnostic run at revision `e5f44cc` encountered an evaluation failure
for the tiled alpha-0.2 comparison: the probability interval remained `5.68e-6`
wide after the existing 20,000-iteration limit. Its training parameters were
saved before evaluation. For the rerun, retain every initial failure and allow
one retry with 200,000 evaluation iterations, keeping the `1e-8` accuracy
threshold unchanged. If a retry fails, retain that seed and make the affected
aggregate unavailable. No training settings or acceptance tolerances change.

## Reference

Sutton and Barto, *Reinforcement Learning: An Introduction*, Section 9.5.4,
[tile coding and step-size scaling](https://web.stanford.edu/class/psych209/Readings/SuttonBartoRL.pdf#page=228):
each tiling contributes an active binary feature; step-size scaling controls
the update at the sampled state, and shared tiles propagate smaller updates
to other states. The geometry and offsets determine that generalization.
