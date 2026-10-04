# Milestone 4: learning action and option models

Implements [issue #7](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/7)
on `milestone/04-option-model-learning`, branched from master `b1e4266`.
The [contract](model_learning_contract.md) was committed as `1b0b7b5` before
implementation and training. Experiment code revision:
`a6aef107e540a1ff5dacafd9215a21c5762a9d3f`.

## Reproduce

```sh
uv sync --locked
uv run pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run python -m rrs.experiments.model_learning
```

The command regenerates and freezes the Milestone 3 option, builds independent
exact model references, audits them with sampled rollouts, then trains six models
from each behavior stream. It writes:

- [Model error curves](../figures/milestone_04_model_learning.png)
- [Final hallway-model predictions](../figures/milestone_04_hallway_model.png)
- `artifacts/model_learning/learning.json`: configuration, frozen policy weights,
  references, per-seed metrics, standard errors, audits and acceptance results.
- `artifacts/model_learning/models_step_{00000,10000,20000,50000}.json.gz`:
  model weights for every run, with feature order, model order, array axes, seeds,
  step count and gamma. These checkpoints support the next planning milestone.

Artifacts are ignored by git; the two figures are tracked. `--output-dir` and
`--figures-dir` override the default paths. Compressed checkpoints are ordinary
JSON readable with `json.loads(gzip.decompress(path.read_bytes()))`. Their reward
arrays use `(run, option, input_feature)` order; successor arrays use
`(run, option, output_feature, input_feature)`, matching `LinearExpectationModel`.

## Protocol and results

Paper target: the model-accuracy portion of [Figure 3 and Section 4](https://arxiv.org/html/2202.03466v4#S4).
Model error should decrease with off-policy experience. Planning utility and the
Figure 3 planning inset remain for Milestone 5, as agreed for this milestone.

Freeze the actual stochastic policy and Equation 9 stopping rule learned by
Milestone 3 seed 0 after 50,000 transitions. All 100 model-training runs share
this option. Also learn four primitive models and the deterministic shortest-path
comparison from Milestone 2. No policy seed was selected by performance.

Model-training seeds are 1000–1099, each with 50,000 transitions and metrics every
500. Use uniform behavior, zero initial model weights/traces, gamma 0.99,
alpha_r=alpha_p=0.1 and lambda=0. These are separate random streams from option
training. No hyperparameters, thresholds, seeds or budgets changed after training.

| Model | Final reward RMSE, mean ± SE | Final successor RMSE, mean ± SE |
| --- | ---: | ---: |
| UP | 3.44e-8 ± 0 | 1.50e-4 ± 5.99e-5 |
| DOWN | 5.44e-7 ± 2.08e-7 | 1.76e-4 ± 5.02e-5 |
| LEFT | 3.43e-5 ± 8.40e-6 | 3.12e-5 ± 1.03e-5 |
| RIGHT | 7.66e-6 ± 4.34e-6 | 1.75e-4 ± 9.49e-5 |
| Reward-respecting | 0.060469 ± 0.001655 | 0.050277 ± 0.001485 |
| Shortest path | 0.015613 ± 0.002365 | 3.36e-7 ± 1.62e-7 |

All 27 frozen checks passed. Every model's final mean errors were below half
their initial values and below the absolute limits: 0.01 for primitives, 0.20
for reward-respecting, and 0.10 for shortest path. All parameters and metrics
were finite; all terminal predictions were zero.

The hallway reward error fell from 0.483145 to 0.060469 (12.5% of its initial
value); successor error fell from 0.798976 to 0.050277 (6.3%). Successor RMSE
averages squared vector norms across source states, without dividing by the
number of output features. The error plot uses a logarithmic vertical scale
to show both the stochastic option errors and small primitive-model errors.
Shading is one standard error across model-training seeds.

There is still finite-budget underestimation in the far-right room. For example,
the hallway model's mean reward prediction at `(5,13)` is about 0.195 below its
exact value. Passing the aggregate thresholds does not imply convergence at
every state. Uncertainty across independently learned option policies is also
outside this experiment: all model runs intentionally use the same frozen option.

## Independent validation and interpretation

The stochastic oracle evaluates fixed policy/stopping probabilities using
Python-double Bellman iteration over the independent Python environment.
It rejects options with a reachable class that never stops. All deterministic
cases agree with the existing exact models. The largest final oracle error bound
was `9.203e-11`, below the fixed `1e-10` tolerance. RMSE measurement and the JSON's
reference weight arrays use float32 exports; oracle error bounds apply before
that export, whose rounding is below the learning comparison tolerances.

An additional 40,000 independent Python rollouts audited the frozen hallway
model from S, `(1,13)`, H and `(3,3)`, using 10,000 samples per start and seeds
20260–20263. Reward and every successor coordinate passed the preregistered
`0.01 + 6*sample_standard_error` tolerance. The longest rollout took 70 actions,
well below the 10,000-action limit; none was truncated.

The reference model belongs to the actual learned stochastic option. At S its
expected environment reward is approximately `-0.045290`, and its discounted
final hallway feature is `0.858429`. Occasional stochastic mistakes explain why
these differ from the optimal deterministic option's zero reward and
`gamma**12` successor. The artificial hallway bonus is never model reward.

Known paper discrepancy: Equation 17's printed stopping-feature target would
produce a `gamma**(K-1)` successor when inserted into Equation 5. This milestone
follows Equation 15 and the established planning interface, using `gamma**K`.
Primitive models therefore predict immediate reward and `gamma*next_features`.
Terminal arrival retains the environment reward and has zero successor target.
Stopping is checked on arrival; only environment termination resets behavior.

Two complete runs at the implementation revision produced byte-identical reports,
both PNGs and all four compressed checkpoints, with a clean tracked worktree.
Python 3.12.9 and JAX/jaxlib 0.7.0 were used; the report records package versions
and revision. These checks establish repeatability in this environment, not
bitwise agreement across different JAX versions or hardware.

## Tests and equation map

All 133 tests pass, including 23 added in this milestone. They cover hand-derived
TD/UWT updates, mixed features, matrix orientation, terminal masking, fractional
stopping, zero importance ratios, nonzero traces, a learned two-step model,
analytic stochastic oracles, improper-option rejection, independent rollouts,
eager/compiled agreement, reset timing, reproducible short training, vector-error
metrics and checkpoint serialization. The existing PR workflow runs the suite;
the 100-run experiment remains outside CI. All pre-commit checks pass.

| Equation or concept | Implementation |
| --- | --- |
| Eq. 16: feature-facing prediction | `rrs/rl/option_models.py::LinearExpectationModel` |
| Eqs. 12/15/17 and UWT: model update | `rrs/rl/model_learning.py::model_update` |
| Eqs. 12/15: stochastic exact model | `rrs/rl/stochastic_models.py::stochastic_option_model` |
| Frozen policies and uniform-behavior training | `rrs/experiments/model_learning.py` |
| Independent Monte Carlo audit | `rrs/experiments/model_audit.py` |
| Metrics, saved models and scientific figures | `rrs/experiments/model_report.py` |
