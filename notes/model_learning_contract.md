# Action and option model learning contract

Milestone 4, [issue #7](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/7),
branch `milestone/04-option-model-learning` from master `b1e4266`.
Source: [paper Section 4, Equations 12–17](https://arxiv.org/html/2202.03466v4#S4).
Build on the [exact models](hallway_options_contract.md) and
[option learner](option_learning_contract.md). Planning comparisons belong to
Milestone 5; this milestone establishes model accuracy, not planning utility.

## Targets and one-transition update

For a fixed option (policy pi and stopping probability beta), predict:

```text
R(x) = dot(reward_weights, x)                    # Eq. 16
N(x) = successor_weights @ x                    # Eq. 16
R(s) = E[sum(t=0..K-1, gamma**t * reward[t])]     # Eq. 12
N(s) = E[gamma**K * x(final)]                    # Eq. 15
```

Use actual environment rewards, never the artificial hallway stopping bonus
or the shortest-path subtask's -1 cumulant. A nonterminal source always executes
at least one action, even if it is a stopping state. Terminal features are zero.

From pre-update parameters, for behavior transition `(x, a, r, x_next, done)`:

```text
rho = pi(a|x) / mu(a|x)
beta = 1 if done else beta(x_next)
delta_r = r + gamma*(1-beta)*R(x_next) - R(x)
delta_n = gamma*(beta*x_next + (1-beta)*N(x_next)) - N(x)
```

On environment termination explicitly zero the successor target, keeping the
immediate reward. Apply existing UWT separately to reward weights and each row
of the successor matrix, with gradient `x`, the same rho/beta, separate reward
and successor step sizes, and the configured lambda. All deltas and stopping
decisions precede any update. Each successor row has its own trace; matrix
columns correspond to input features, rows to predicted successor features.
The learner consumes features and supplied option probabilities, never state IDs.

Equation 17 as printed omits a gamma on the stopping-feature target when inserted
into Equation 5. Follow Equation 15 and the established model interface above.
For a primitive action this must yield reward `r` and successor `gamma*x_next`.
Test one- and two-step examples to prevent shifting to `gamma**(K-1)`.

## Frozen options and behavior

There are six models, ordered UP, DOWN, LEFT, RIGHT, reward-respecting, shortest
path. Primitives use deterministic policies and beta=1. The shortest-path
comparison is the existing deterministic Milestone 2 option, including its
UP/DOWN/LEFT/RIGHT tie order. Its model still learns environment reward.

Regenerate the Milestone 3 option with seed 0 and its unchanged 50,000-step
protocol, then freeze its actual softmax policy and Equation 9 stopping mask.
Do not select a seed by performance or convert it to a greedy policy. Save its
weights, probabilities and stopping mask. All model-training runs use this same
fixed option, isolating model-training variation from option-training variation.

Every uniform random behavior transition updates all six models with their own
importance ratios. Start at S, learn the goal reward before resetting to S, and
never reset behavior when an option stops. Seeds use `jax.random.key(seed)`;
split into next-key, action-key, environment-key, reset-key on every transition,
then sample with `jax.random.randint(action_key, (), 0, 4)`. Model weights/traces
start at zero. gamma=0.99, alpha_r=alpha_p=0.1, lambda=0, float32 learning.

## Independent references

Extend the oracle to fixed stochastic policies and stopping probabilities by
Python-double Bellman iteration over the independent Python environment:

```text
R(s) = sum_a pi(a|s) * [r + gamma*(1-beta(next))*R(next)]
N(s) = sum_a pi(a|s) * gamma * [beta(next)*x(next)
                              + (1-beta(next))*N(next)]
```

Terminal transitions retain reward and have zero successor contribution.
Already-terminal source rows are zero. Validate that every state can reach
stopping/termination with positive probability; reject improper options. Stop
when the maximum Bellman residual divided by `(1-gamma)` is at most `1e-10`,
with at most 20,000 sweeps. Check deterministic cases against the existing exact
oracle and hand-calculated stochastic cases. Normalize only float32 policy
roundoff within `1e-6`; reject invalid policy rows and stopping probabilities.

Independently audit the frozen hallway model using Python reference rollouts
from S, `(1,13)`, H and `(3,3)`, 10,000 samples per start, seeds 20260–20263.
Compare reward and each successor feature with the oracle using tolerance
`0.01 + 6 * sample_standard_error`; these tolerances are fixed before sampling.
Raise if any rollout exceeds 10,000 actions; do not silently truncate returns.

## Frozen experiment protocol (before model training)

Run 100 model-training seeds 1000–1099 for 50,000 transitions each, with metric
checkpoints every 500 transitions, including zero initialization. These streams
are separate from the frozen option's training stream. Oracle tables are used
only for measurement outside the learning update.

For each model and seed, compute errors over all 72 nonterminal states:

```text
reward_RMSE = sqrt(mean_s (R_learned(s)-R_exact(s))**2)
successor_RMSE = sqrt(mean_s sum_j (N_learned(s)[j]-N_exact(s)[j])**2)
```

Do not divide successor error by the feature count and dilute vector error.
Report means and standard errors across runs. Frozen acceptance criteria:

- Final mean reward and successor RMSE must each be at most half their initial
  value, separately for every model.
- Both final errors must be at most 0.01 for each primitive action, 0.20 for the
  learned reward-respecting option, and 0.10 for the shortest-path comparison.
- Every parameter/metric must be finite. Terminal predictions must be zero.
- Independent oracle, Monte Carlo audit and mathematical/integration tests pass.

These are project thresholds, not numerical claims from the paper. Preserve
failures and inspect semantics before considering another protocol; do not tune
these thresholds, seeds or budget after observing results.

## Checks, outputs and review

Before JIT/VMAP/scan, test hand-calculated primitive, continuation, stopping,
terminal, zero-rho, fractional-beta and nonzero-trace updates, including mixed
features and successor matrix orientation. Check a tiny learned deterministic
option against its analytic model. Compare eager and compiled execution within
float32 tolerance `1e-6`. Keep a short reproducible training test in existing CI.

Save scientific figures as `figures/milestone_04_*.png`. Under ignored
`artifacts/model_learning/`, save JSON metrics and compressed model checkpoints
at steps 0, 10,000, 20,000 and 50,000 for later planning experiments. Record
feature/model order, policy source, seeds, hyperparameters, package versions,
code revision and acceptance results. Confirm reproducibility, document results
in `notes/milestone_04.md`, and stop for review before pushing. README.md stays
uncommitted.
