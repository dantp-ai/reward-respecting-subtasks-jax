# Four-room action and option model learning

Milestone 8, [issue #15](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/15),
branch `milestone/08-four-room-models` from master `adf7e7b`.
Sources: the locally saved paper's Equations 12–17, Section 7 and Appendix A;
specs `05_option_models.md`, `07_four_room_reproduction.md` and
`10_experiment_and_testing_protocol.md`. Build on the
[model-learning semantics](model_learning_contract.md) and
[validated four-room options](milestone_07.md). Planning is a separate milestone.

## Frozen policies and model targets

Use Milestone 7's final 1,000,000-step policies for predetermined seed 7000, with
their actual softmax probabilities and Equation 9 stopping rules. Do not select
by performance, greedify, retrain during model learning, or change stopping maps.
All model-training runs share these four options, so uncertainty describes model
training conditional on that fixed option set, not option-training variation.

Load the canonical `artifacts/four_room_options/options_step_1000000.json.gz`,
whose SHA-256 is
`2e800352b546604d862af4fa73ad9786e243b3b5f503a20a991e10580ff7ec40`.
Verify provenance, ordering and content before selecting seed 7000. If the
default input is missing, regenerate it using the unchanged full Milestone 7
command and verify the hash. A supplied missing path or corrupt existing input
must fail explicitly, never trigger replacement. The hash pins this environment's
validated artifact; a different software environment may require a separately
reviewed input rather than silently substituting a different policy.

Eight models in order UP, DOWN, LEFT, RIGHT, H1, H2, H3, H4. Primitive policies
are deterministic and stop on every arrival. Use the existing feature-facing
linear expectation model and TD/UWT update, with 103 nonterminal one-hot features:

```text
R(s) = E[sum(t=0..K-1, gamma**t * environment_reward[t])]
N(s) = E[gamma**K * x(final)]
rho = pi(requested_action | x) / 0.25
delta_r = r + gamma*(1-beta(next))*R(next) - R(x)
delta_n = gamma*(beta(next)*x(next) + (1-beta(next))*N(next)) - N(x)
```

The artificial hallway bonus is never model reward. Goal arrival retains +1 and
has zero successor contribution. Stopping is checked after an action, even when
the source is a stopping state. The successor discount is `gamma**K`, following
Equation 15 and the previously documented Equation 17 ambiguity.

Each uniform-behavior transition updates all eight models with separate ratios
and stopping decisions. Use the validated JAX environment; only environment
termination resets to S, after learning the terminal transition. Randomness:
`key(seed)`, then `split(next, action, environment, reset)` and uniform requested
action via `randint(action, (), 0, 4)` on every transition.

## Independent stochastic references and audits

Extend the existing Python-double model oracle to sum over both action
probabilities and environmental outcomes from the independent Python environment.
Retain deterministic support and regression checks. Explicitly reject a reachable
class that never stops. Normalize only policy-row roundoff within `1e-6`.
Use Bellman residual/(1-gamma) <= `1e-10`, at most 20,000 sweeps.

Test mixed terminal/nonterminal outcomes, wall aggregation, fractional stopping,
zero-probability branches, first-action semantics and matrix orientation by hand.
All four primitive reference models must equal direct expected one-step reward
and discounted next-feature sums within `1e-10`. For each frozen hallway option,
`R + N @ z / gamma` must match the independent Milestone 7 subtask evaluator
within `2e-8` over all 104 states. This identity accounts for the subtask bonus's
`gamma**(K-1)` discount while model successors use `gamma**K`.

Audit each of the four hallway reference models with independent Python rollouts
using the nine-ticket environment sampler, not the enumerated transition table.
Starts, in order: S `(4,1)`, H1 `(6,2)`, H2 `(3,6)`, H3 `(7,9)`, H4 `(10,6)`,
penalty `(9,3)`. Use 10,000 samples per option/start pair, seeds 8100–8123 in
option-major order, 10,000-action cap. A horizon overrun raises; no silent
truncation. Compare environment reward and every discounted successor coordinate
with the exact reference using `0.01 + 6*sample_standard_error`. Retain audit
sample moments, errors, tolerances and longest durations.

## Frozen model-training experiment

This protocol is committed before any Milestone 8 training results are observed.

- 30 model-training seeds 8000–8029, independent of the frozen option stream.
- 200,000 transitions per run, matching the span of Appendix A's model-learning
  plot; metric checkpoints every 1,000 transitions, including zero initialization.
- gamma=0.99, alpha_r=alpha_p=0.1, lambda=0; float32 learning, zero initial model
  weights and traces. Reuse the existing update equations unchanged.
- Save complete models at 0, 10,000, 20,000, 30,000, 40,000 and 200,000 transitions
  for later comparisons of model maturity and planning. The early checkpoints
  follow the four marked intervals in Appendix A's plot.
- Reward RMSE is `sqrt(mean_source((R_learned-R_exact)**2))` over all 103
  nonterminal sources. Successor RMSE is
  `sqrt(mean_source(sum_output((N_learned-N_exact)**2)))`; do not divide by the
  feature count. Report each model separately, with means and SE across seeds.

Final acceptance criteria apply to each model's mean errors:

| Models | Reward RMSE limit | Successor RMSE limit | Each final error / initial error |
| --- | ---: | ---: | ---: |
| Each primitive action | 0.15 | 0.40 | <= 0.65 |
| Each frozen hallway option | 0.25 | 0.25 | <= 0.50 |

These are project thresholds, not numerical claims from the paper. Constant
step sizes in a stochastic environment leave sampling error even in primitive
models; the deterministic Milestone 4 primitive limit of 0.01 is inappropriate
here. No cross-model average can satisfy a failed individual criterion.

All recorded parameters and metrics must be finite, with zero terminal
predictions. Successor weights must remain >= -1e-7 and their sum per source
must be <= gamma+1e-6. All reference identities and 24 rollout audits must pass.
Repeat the full experiment: reports, model checkpoints, frozen-option exports
and figures must be byte-identical in this environment. Do not change seeds,
budgets, thresholds or step sizes after observing results; preserve and diagnose
failures through the established equation/environment/discount/indexing checks.

## Tests and artifacts

Test eight separate updates from one shared transition, requested-action ratios,
arrival stopping, zero artificial bonus, terminal learning before reset, no reset
at option stopping, eager/JIT/VMAP agreement and short seeded reproducibility.
Retain the existing two-room model regression tests. Short tests run in the
existing PR workflow; the 30-run experiment and 240,000 rollouts remain separate.

Save `figures/milestone_08_*.png`, results in `notes/milestone_08.md`, and ignored
JSON/gzip artifacts under `artifacts/four_room_models/`. Checkpoints include
model/feature order, array axes, gamma, seeds, configuration, step count and a
hash linking the frozen-option definition. Export that definition alongside
the models so later planning cannot silently use different policies. Record
source hashes, software versions, code revision, and per-model checks.
Keep README.md untouched and uncommitted. Commit locally and pause for review
before pushing.
