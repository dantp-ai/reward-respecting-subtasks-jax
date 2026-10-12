# Milestone 10 investigation: Why did tile coding underperform?

Requested after the frozen [Milestone 10 result](milestone_10.md).
This is a separate diagnostic study on the same milestone branch. Preserve the
original experiment, its parameters and artifacts. The original result does not
establish that tile coding is generally inferior, nor does tile coding guarantee
improvement on every discrete task.

**Finding:** the original 50k result reflects harmful early policy generalization
and premature stopping under the chosen coordinate tiles. At 200k transitions,
ordinary tiles reach 99.269% safe attainment versus 99.036% for one-hot, although
their discounted return remains lower. Separating penalty/safe feature blocks
reaches 96.709% at 50k, but uses additional environment information. The audit
found no encoding, normalization or evaluation discrepancy. See results below;
the original experiment remains unchanged.

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

## Results and interpretation

All comparisons use the same 100 paired seeds. Safe attainment means stopping
at the hallway without encountering any negative reward; it is not merely
entering the hallway. Returns include the discounted subtask stopping bonus.
Errors shown below are one standard error across seeds.

### Full declared training comparisons

| Configuration | Steps | Joint alpha | Safe attainment %, mean ± SE | Actual return, mean ± SE | Critic RMSE, mean ± SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_hot | 50,000 | 0.1 | 95.184856 ± 0.017220 | 0.820896 ± 0.000275 | 0.238545 ± 0.005171 |
| one_hot | 100,000 | 0.1 | 97.880228 ± 0.005386 | 0.859418 ± 0.000098 | 0.037863 ± 0.000904 |
| one_hot | 150,000 | 0.1 | 98.665765 ± 0.002949 | 0.871573 ± 0.000054 | 0.022050 ± 0.000099 |
| one_hot | 200,000 | 0.1 | 99.036208 ± 0.001896 | 0.877600 ± 0.000033 | 0.018029 ± 0.000088 |
| tile_coding | 50,000 | 0.1 | 53.875481 ± 1.701297 | 0.192093 ± 0.009200 | 0.305880 ± 0.002297 |
| tile_coding | 100,000 | 0.1 | 97.367798 ± 0.112184 | 0.790784 ± 0.005771 | 0.096469 ± 0.005352 |
| tile_coding | 150,000 | 0.1 | 98.895885 ± 0.006096 | 0.853535 ± 0.000269 | 0.026689 ± 0.000272 |
| tile_coding | 200,000 | 0.1 | 99.268793 ± 0.002300 | 0.867291 ± 0.000102 | 0.019956 ± 0.000168 |
| binary_tiles | 50,000 | 0.025 | 53.875481 ± 1.701297 | 0.192093 ± 0.009200 | 0.305880 ± 0.002297 |
| tiled_critic_only | 50,000 | 0.1 | 95.004668 ± 0.018187 | 0.815559 ± 0.000303 | 0.097556 ± 0.002050 |
| tiled_actor_only | 50,000 | 0.1 | 45.145659 ± 1.971679 | 0.211126 ± 0.010854 | 0.330289 ± 0.002278 |
| split_penalty_tiles | 50,000 | 0.1 | 96.709232 ± 0.012191 | 0.824232 ± 0.000282 | 0.052337 ± 0.000549 |
| one_hot_alpha_0.025 | 50,000 | 0.025 | 0.000000 ± 0.000000 | -0.021914 ± 0.000122 | 0.747556 ± 0.001208 |
| one_hot_alpha_0.05 | 50,000 | 0.05 | 87.503901 ± 0.217929 | 0.713955 ± 0.002638 | 0.512591 ± 0.002703 |
| one_hot_alpha_0.2 | 50,000 | 0.2 | 97.860773 ± 0.007551 | 0.858596 ± 0.000141 | 0.042173 ± 0.001577 |
| tile_coding_alpha_0.025 | 50,000 | 0.025 | 0.000000 ± 0.000000 | -0.141059 ± 0.011506 | 0.606233 ± 0.002458 |
| tile_coding_alpha_0.05 | 50,000 | 0.05 | 3.193825 ± 0.914009 | -0.006355 ± 0.001484 | 0.423121 ± 0.002128 |
| tile_coding_alpha_0.2 | 50,000 | 0.2 | 95.140664 ± 0.998258 | 0.720121 ± 0.014287 | 0.144055 ± 0.007357 |

The binary-tile alpha 0.025 case is a reparameterization of normalized tiles
at alpha 0.1; it is distinct from normalized tiles trained at alpha 0.025.

### Final stopping-map swaps

| Fixed action policy | Stopping map | Safe attainment %, mean ± SE | Actual return, mean ± SE |
| --- | --- | ---: | ---: |
| one_hot | one_hot | 95.184856 ± 0.017220 | 0.820896 ± 0.000275 |
| one_hot | tile_coding | 90.946886 ± 0.381852 | 0.785132 ± 0.003270 |
| tile_coding | one_hot | 72.676837 ± 0.711992 | 0.194463 ± 0.010008 |
| tile_coding | tile_coding | 53.875481 ± 1.701297 | 0.192093 ± 0.009200 |

### What the interventions establish

1. **Actor sharing is a major contributor during learning.** Keeping the actor
   one-hot while tiling only the critic preserves 95.005% safe attainment at
   50k; tiling only the actor yields 45.146%. Critic-only tiling also improves
   value RMSE from 0.238545 to 0.097556. Thus useful value generalization and
   harmful policy generalization coexist in this experiment. The components
   interact through the TD error and importance ratio, so this is not an
   additive decomposition of causes.
2. **Final stopping errors matter, but do not explain everything.** Giving the
   original tiled policy the one-hot stopping map improves attainment from
   53.875% to 72.677%, still below 95.185% for the one-hot policy. Its actual
   return barely changes (0.192093 to 0.194463). Swapping final maps does not
   remove the critic's influence on the actor during training.
3. **Cross-boundary sharing is harmful at this budget.** Of 221 overlapping
   state pairs, 38 cross the penalty/non-penalty boundary. Blocking this
   sharing gives 96.709% safe attainment and return 0.824232 at 50k. This
   diagnostic uses the penalty label, doubles the nominal feature count to
   196, and changes both actor and critic sharing. It does not prove which
   individual shared update causes the problem or establish superiority of
   generic coordinate tiles. Both original and split feature matrices have
   enough rank to represent arbitrary values on the 72 nonterminal states.
4. **The original budget catches tiles before recovery.** At 150k and 200k,
   tile coding slightly exceeds one-hot on safe attainment. At 200k the paired
   difference is +0.232585 ± 0.002359 percentage points. Its return is still
   lower by 0.010309 ± 0.000084 and its critic RMSE is slightly higher. There
   is no across-metric winner. The 50k result alone is insufficient to judge
   the eventual behavior of the representation.
5. **Learning rate matters, but simply lowering it does not repair this
   finite-budget comparison.** The complete grid is above. At 50k, alpha 0.2
   brings tiles to 95.141%, while one-hot at that same alpha reaches 97.861%.
   Smaller rates delay attainment for both methods. These exploratory rates
   are not newly validated defaults.

### Observed behavior in saved policies (post-hoc inspection)

Averaged across all 100 original runs, from S the tiled action policy chooses
left into the wall with probability 0.97138 and down with probability 0.01511.
The one-hot policy chooses down with probability 0.82900. Sixty tiled runs stop
at safe cell `(2, 1)`, versus zero one-hot runs. These observations illustrate
the policy and stopping problems; they were inspected after the declared runs.
At 200k, the tiled policy chooses down from S with probability 0.85052 and left
with probability 0.14302; the penalty-split variant already chooses down with
probability 0.87850 at 50k. Slow progress is consistent with the lower
discounted return, although these start-state probabilities do not decompose
the complete return difference.

For an example of the sharing mechanism, safe `(3, 1)` and penalty `(3, 2)`
share two of their four active tiles: their feature inner product is 0.5.
A linear update at one therefore changes the other's value or action logits
as well. Unit normalization fixes the source-state update scale; it cannot
make these cross-state effects disappear. The interventions support harmful
boundary-related policy generalization as an explanation, without proving a
unique causal trajectory for all seeds.

### Audit and reproducibility

- Independent enumeration reproduces every tile feature exactly; nonterminal
  squared norms are one and terminal features are zero. The 72-by-98 feature
  matrix has rank 72, condition number 19.6144, and fits the exact value with
  RMSE 9.44e-16. The fitted parameters are never used by the learner.
- At 50k, the diagnostic runner reproduces every original actor and critic
  parameter exactly for both representations. Binary tiles with alpha 0.025
  reproduce normalized tiles with alpha 0.1 exactly after rescaling weights.
- Independent seed-0 simulations use 10,000 executions per representation.
  One-hot sampled safe attainment is 0.9563 ± 0.002044 versus exact 0.952653;
  tiles give 0.4671 ± 0.004989 versus exact 0.468226. Sampled returns are
  0.825171 ± 0.002038 versus exact 0.821810, and 0.202470 ± 0.003574 versus
  exact 0.203179. All four checks pass the declared tolerance. This sampled
  audit covers seed 0; the exact evaluator covers every seed.
- All 2,200 fixed-policy evaluations resolve at value/probability bounds
  below 1e-8. Only normalized tiles at alpha 0.2, seed 62, need the documented
  retry: 30,497 iterations. Its initial failure remains in the report.
- **242 tests pass**, including seven new diagnostic tests. Pre-commit checks
  pass. All 12 parameter archives are byte-identical between the initial run
  and the rerun with evaluation failure handling. The original report, both
  original parameter archives and both original figures retain their hashes.

Protocol revision: `d206143`. Initial implementation: `e5f44cc`. Completed
experiment revision: `86557ec29b20fb096a82e5e6197215056fa59c0f` (clean tracked worktree).
Python 3.12.9, JAX 0.7.0 on CPU with float32, NumPy 2.3.2. The only protocol
amendment is the evaluation-iteration retry documented above. The figure
formatter was subsequently adjusted to suppress overlapping minor tick labels;
no training or evaluation results changed.

```sh
MPLCONFIGDIR=.cache/matplotlib uv run --locked --offline --no-cache python -m rrs.experiments.tile_coding_diagnostics
uv run --locked --offline --no-cache pytest -q
```

Report: `artifacts/tile_coding_investigation/investigation.json`.
SHA-256: `f4e2ffe0cfbad1637e3a20270f3db414a92401fbdb07b87c23aa0b4df1d35324`.
The report records the hashes of all 12 compressed parameter archives, complete
per-seed results and evaluation failures. Archives include feature tables and
actor/critic snapshots, allowing independent reconstruction of each policy.

- [Budget and step-size comparisons](../figures/milestone_10_investigation_sensitivity.png)
- [Actor/critic and stopping-map comparisons](../figures/milestone_10_investigation_components.png)

### Implication for the next experiment

Keep the original 50k result and report it alongside learning-budget sensitivity.
Choose training budgets and separate actor/critic representations explicitly
before subsequent model/planning comparisons. Test any preferred configuration
on new seeds before treating this exploratory study as confirmation. Keep the
penalty-split representation labeled as an environment-informed diagnostic.
This study does not establish a universal advantage for either encoding.
