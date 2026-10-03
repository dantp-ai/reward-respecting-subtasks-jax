# Milestone 3: off-policy hallway option learning

Implements [issue #5](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/5)
on `milestone/03-hallway-option-learning`, branched from merged master `561c331`.
The [contract](option_learning_contract.md) was committed as `2c4505b` before
training. Experiment implementation revision: `54d7b8f95b1c4ae94afea1a19c4ba54e5e2041fe`.

## Reproduce

```sh
uv sync --locked
uv run pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run python -m rrs.experiments.option_learning
```

The command runs the frozen experiment, evaluates all final stochastic policies,
and writes `artifacts/option_learning/learning.json` plus these tracked figures:

- [Value learning curves](../figures/milestone_03_option_learning.png)
- [Stochastic policy and safe hallway probability](../figures/milestone_03_stochastic_policy.png)

Use `--output-dir PATH` for JSON and `--figures-dir PATH` for figures. The default
keeps all figures under `figures/`. The report contains checkpoint metrics for
every seed, final actor/critic weights, policy probabilities, stopping positions,
hyperparameters, evaluation error bounds, acceptance results and code revision.
It preserves the report and exits unsuccessfully if a frozen criterion fails.

## Experiment and result

Paper target: reproduce the qualitative Figure 2 behavior from
[Section 3](https://arxiv.org/html/2202.03466v4#S3): initial error increase followed
by declining RMSE, and start-state value approaching `gamma**11 ≈ 0.895338`.

Seeds: 0–99, each with 50,000 behavior transitions and checkpoints every 500.
Hyperparameters: uniform behavior, gamma 0.99, actor/critic step sizes 0.1,
both trace lambdas zero, zero initial weights/traces, frozen main-task weights
zero, hallway bonus one. There was no parameter, seed or budget search.

| Final quantity | Mean ± standard error | Frozen criterion |
| --- | ---: | --- |
| Critic RMSE over 72 nonterminal states | 0.238545 ± 0.005171 | ≤ 0.40 |
| RMSE as fraction of peak mean RMSE | 0.221796 | ≤ 0.75 |
| Estimated start value | 0.813672 ± 0.000961 | Within 0.15 of 0.895338 |
| Actual stochastic-policy start value | 0.820896 ± 0.000275 | Report separately |
| Safe hallway probability from S | 0.951849 ± 0.000172 | Mean ≥ 0.80 |

All frozen criteria passed, including finite parameters and metrics. Mean RMSE
started at 0.906070, peaked at 1.075515 after 1,000 transitions, then fell to
0.238545. The actual policy's value remains below the optimum after this finite
budget; these results do not claim full convergence or exact numerical agreement
with the paper's curve. The paper's original random streams are unavailable here,
and our checkpoint spacing is 500 transitions rather than every transition.

Safe hallway probabilities ranged from 0.945971 to 0.956768 across the 100 runs.
They measure the complete stochastic policies, including occasional bad actions;
they are not success rates for greedy arrows. The evaluator enumerates transitions
through the independent Python environment and uses Bellman iteration with
absolute value-error bound and success-probability interval width at most `1e-8`.
Every evaluation converged within 233 sweeps. Reported standard errors describe
variation across training seeds, not Monte Carlo evaluation noise.

Two full runs at the implementation revision produced byte-identical JSON and
both PNGs. Both runs had a clean tracked worktree. Python 3.12.9 and JAX/jaxlib
0.7.0 were used; the JSON records package versions and revision automatically.

## Semantics to review

The critic and actor share a TD error, importance ratio and destination stopping
decision computed before either weight update. Each UWT call accumulates its
trace with importance sampling, updates weights, then decays the trace.

The learned stopping function uses Equation 9 (`z >= V`), not the exact oracle's
general optimal-stopping comparison (`z >= gamma*V`). Environment termination
forces stopping and zero stopping value while retaining the terminal reward.
Only environment termination resets behavior experience. Option stopping alone
leaves the behavior trajectory running, as required for off-policy learning.

Known paper ambiguity: the illustrative text/inset emphasizes hallway and goal
stopping. Our Equation 9 implementation also learns stopping in eight penalty
cells (rows 1–4, columns 3–4), as predicted by Milestone 2's exact oracle. Seed 0
shows these explicitly in the figure. Stopping for zero there is preferable to
the negative value of taking another action. Initiation still requires one action
even when the source is a stopping state. No stopping set was hard-coded.

## Verification and equation map

All 110 tests pass, including 25 added in this milestone. Tests cover numerical
updates and gradients, trace order, an independent scalar update sequence,
eager/compiled agreement, explicit key splitting, terminal update/reset order,
option stopping without reset, short reproducible training, and independent
stochastic evaluation with loops, penalties, terminal states and nonconvergence.
The existing PR workflow runs this suite; long training stays outside CI.
All pre-commit checks pass.

| Equation or concept | Implementation |
| --- | --- |
| Eq. 3: linear value | `rrs/rl/learning.py::linear_value` |
| Eq. 4: stopping value | `rrs/rl/subtasks.py::stopping_value` |
| Eq. 5: TD error | `rrs/rl/learning.py::td_error` |
| Eq. 9: stopping decision | `rrs/rl/learning.py::stopping_rule` |
| Eq. 10 and UWT: actor/critic update | `rrs/rl/learning.py::actor_critic_update` |
| Eq. 11: linear softmax | `rrs/rl/learning.py::softmax_policy` |
| Eq. 2: frozen stochastic policy evaluation | `rrs/rl/policy_evaluation.py::evaluate_policy` |
| Experience and seeded training | `rrs/experiments/option_learning.py` |
| Independent evaluation, report and figures | `rrs/experiments/learning_report.py` |

The option learner now has independent numerical and empirical checks. Learning
its reward and successor-feature models is the next milestone; this change adds
no model-learning updates or planner.
