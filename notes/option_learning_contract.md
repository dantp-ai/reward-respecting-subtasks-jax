# Off-policy hallway option learning contract

Milestone 3, branch `milestone/03-hallway-option-learning` from master `561c331`.
Source: [paper Section 3, Equations 3–11 and UWT](https://arxiv.org/html/2202.03466v4#S3).
The environment, features and fixed subtask are specified in
[Milestone 2](hallway_options_contract.md). Model learning and planning are separate.

## One transition

The learner consumes current/next feature vectors, action, cumulant, next stopping
value, behavior action probability and terminal flag. It never consumes state IDs.
Critic weights have shape `(features,)`; actor weights have shape
`(actions, features)`. The latter is the paper's flattened 288 state-action
features, written as a 4 by 72 matrix. All weights and traces start at zero.

Compute every quantity below from the same **pre-update** parameters:

```text
v = dot(critic, x); v_next = dot(critic, x_next)                 # Eq. 3
pi = softmax(actor @ x)                                       # Eq. 11
rho = pi[action] / behavior_probability
beta = terminal or (z_next >= v_next)                         # Eq. 9
delta = cumulant + beta*z_next + gamma*(1-beta)*v_next - v     # Eq. 5
critic_gradient = x
actor_gradient = outer(one_hot(action) - pi, x)
```

Terminal arrival overrides `beta=1` and `z_next=0`; preserve its immediate reward.
The terminal representation is zero. Equation 9 deliberately compares `z_next`
with `v_next`, not `gamma*v_next`; this differs from the optimal-stopping oracle
in general. Stop on equality. Do not hard-code the learned stopping states.

Apply UWT independently to actor and critic using the same delta, rho and beta:

```text
trace = rho * (trace + gradient)
weights = weights + alpha * delta * trace
trace = gamma * lambda * (1-beta) * trace
```

Input behavior probabilities must be positive. Keep this precondition explicit;
do not clip importance ratios. The policy gradient holds the sampled action fixed.

## Experience and reproducibility

Run uniform random behavior (`mu=0.25`) in the validated JAX environment, starting
at S. Reset to S only after environment termination at G, after applying the last
update. Option stopping does **not** reset behavior experience. Every initiated
option takes at least one action, including initiation at a stopping state.

Each run uses `jax.random.key(seed)`, seeds 0 through 99. Each transition splits
the carried key into next-key, action-key, environment-key and reset-key, in that
order. Choose behavior actions with `jax.random.randint(action_key, (), 0, 4)`.
Use float32 learning, gamma 0.99, critic/actor alpha 0.1, both lambdas zero,
frozen main-task weights zero and hallway bonus one. Learning never receives the
optimal values or model tables.

## Frozen experiment protocol (before the first training run)

- 100 independent runs, 50,000 environment transitions each, checkpoints every
  500 transitions, including the initial zero parameters. No seed selection.
- RMSE uses all 72 nonterminal critic values against the Milestone 2 oracle.
  Plot mean and standard error across runs, plus estimated start value and its
  exact target `0.99**11 = 0.8953382542587163`.
- Final mean RMSE must be at most 0.40 and at most 75% of the largest checkpoint
  mean RMSE (allowing the initial transient visible in the paper).
- Final mean estimated start value must be within 0.15 of the exact target.
- Evaluate each final **stochastic** policy with its learned Equation 9 stopping
  decisions. Mean probability of reaching H without any penalty arrival must
  be at least 0.80. Report the actual expected subtask return separately from the
  critic estimate; greedy arrows alone are not an evaluation.
- All parameters and reported metrics must be finite. These are project
  acceptance thresholds, not numerical results claimed by the paper.

Evaluate frozen policies by independent Python-double Bellman iteration using
the Python reference environment. Compute expected Equation 2 return and safe
hallway probability. A penalty arrival makes safe attainment fail even if the
option subsequently reaches H. Terminal G or stopping elsewhere also fails.
Source stopping never suppresses the first action. Discounted values use an
absolute error bound of `1e-8`; safe success probabilities are bounded by lower
and upper iterates until their gap is at most `1e-8` (maximum 20,000 sweeps).
Fail explicitly on nonconvergence. This evaluates all stochastic paths without
sampling error or silently truncated rollouts. Report mean and standard error.

If a frozen criterion fails, preserve the result and investigate equations,
transition timing, state/feature ordering, ratios and resets before considering
a different protocol. Do not tune the thresholds, seeds or budget after seeing
results. Distinguish an implementation check from a reproduction claim.

## Checks and artifacts

Hand-calculated tests cover TD error, stable softmax, analytic log-policy
gradient, importance ratios, UWT ordering, nonzero traces, stopping equality,
the Equation 9/oracle distinction, and terminal masking. Compare a short
transition sequence against an independent scalar implementation before using
JIT/scan/vmap. Verify compiled and eager results within float32 tolerance `1e-6`.

A short seeded CI run checks reproducibility, finite/nontrivial learning and
reset semantics, without asserting convergence from a tiny training budget.
Frozen-policy evaluator tests include stochastic branching, self-loops,
penalties, stopping on arrival, terminal transitions and nontermination.

Save figures as `figures/milestone_03_*.png`. Save an ignored JSON report under
`artifacts/option_learning/`, including seeds, all hyperparameters, checkpoint
metrics, final weights, stochastic evaluation, acceptance checks, package
versions and code revision. Record scientific results in `notes/milestone_03.md`.
