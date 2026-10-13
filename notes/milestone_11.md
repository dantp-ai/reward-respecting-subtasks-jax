# Milestone 11: Tile-coded action and option models

Issue [#21](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/21),
branch `milestone/11-tile-coded-models`, from merged master `d5be2bd` (PR #20).
The [contract](tile_coded_models_contract.md) was committed as `09534d7` before
any new option or model training. Implementation revision:
`411f74f2c05342eda3903981a0e3b9cfbe5e3bd7`.

## Result

Tile-coded models learn the two frozen hallway options faster early in training,
but one-hot models finish with lower errors at the prescribed 50,000-transition
budget. Primitive-action models also finish more accurately with one-hot.
All implementation checks and all 42 declared performance diagnostics pass.
This is a model-accuracy result; it does not establish planning utility.

## Frozen experiment

Both model representations learn the same seven options: four primitive actions,
existing exact shortest-path hallway option, and two learned hallway options.
The latter use the unchanged Milestone 10 actor-critic with one-hot or original
geometry-only tiles, fresh predetermined seed 100, and 200,000 transitions.
Their policies and stopping maps are frozen before model learning; neither was
selected by performance. No penalty-aware features or post-result tuning are used.

| Frozen option | Safe hallway attainment from S | Actual subtask return from S |
| --- | ---: | ---: |
| one_hot_option | 99.031313% | 0.877581 |
| tile_option | 99.272251% | 0.867784 |

Each model case uses 100 paired seeds 2000–2099, 50,000 uniform-behavior
transitions, gamma 0.99, alpha_r=alpha_p=0.1, lambda=0 and zero initial weights.
Metrics are recorded every 500 transitions. Models use actual environment
rewards, without the artificial hallway bonus, and gamma**K successor features.
Only goal termination resets behavior, after learning the terminal transition.
All options execute an action before testing arrival stopping.

The cases use 72 one-hot features or the unchanged 98 normalized tile features.
Model policies are looked up by observation independently of model features.
The core TD/UWT model update is unchanged. Changing model representation cannot
change which fixed policy or stopping decisions it models.

All errors below evaluate predictions over the 72 nonterminal source states.
Native successor error uses a vector norm, without dividing by feature count.
For a common output space, decode successor predictions to 72 discounted
final-state indicators using the pseudoinverse of the nonterminal feature table.
This transform is used only for measurement. It does not clip, normalize, or
feed oracle values into model learning. Both feature matrices have row rank 72;
condition numbers are 1 and 19.6144, with decoder identity residuals 0 and
1.50e-15. The common metric can amplify tile-feature errors and is reported
alongside the native metric, not treated as a transition-probability model.

## Final accuracy: mean ± standard error

| Fixed option | One-hot reward RMSE | Tile reward RMSE | One-hot common successor RMSE | Tile common successor RMSE |
| --- | ---: | ---: | ---: | ---: |
| up | 3.44128e-08 ± 0 | 0.00553166 ± 8.4e-05 | 0.000134842 ± 3.31e-05 | 0.145974 ± 0.00135 |
| down | 8.86769e-07 ± 2.53e-07 | 0.0284254 ± 0.000431 | 0.000573483 ± 0.000204 | 0.145304 ± 0.0012 |
| left | 3.53351e-05 ± 6.34e-06 | 0.0172859 ± 0.000367 | 4.0717e-05 ± 1.25e-05 | 0.146833 ± 0.00124 |
| right | 4.99878e-06 ± 2.63e-06 | 0.0113495 ± 0.000206 | 0.00020047 ± 5.62e-05 | 0.136973 ± 0.0012 |
| shortest_path | 0.0190348 ± 0.00272 | 0.0381543 ± 0.000967 | 1.64625e-07 ± 2.91e-09 | 0.0388117 ± 0.000737 |
| one_hot_option | 0.0332849 ± 0.00101 | 0.0378137 ± 0.00117 | 0.0297505 ± 0.000765 | 0.035574 ± 0.00111 |
| tile_option | 0.029211 ± 0.000763 | 0.0360207 ± 0.0012 | 0.0269888 ± 0.000655 | 0.0339274 ± 0.00114 |

One-hot native successor RMSE equals its common-state RMSE. Tile-native errors
are shown separately because their output coordinates differ:

| Fixed option | Tile native successor RMSE, mean ± SE |
| --- | ---: |
| up | 0.0366618 ± 0.00054 |
| down | 0.0441989 ± 0.00068 |
| left | 0.0372926 ± 0.000462 |
| right | 0.0314241 ± 0.000455 |
| shortest_path | 0.0388117 ± 0.000737 |
| one_hot_option | 0.0355747 ± 0.00111 |
| tile_option | 0.033928 ± 0.00114 |

The following differences are computed within each of the 100 paired seeds.
Positive means tiles have higher error; these SEs account for paired experience.

| Fixed option | Tile minus one-hot reward RMSE | Tile minus one-hot common successor RMSE |
| --- | ---: | ---: |
| up | +0.00553163 ± 8.4e-05 | +0.145839 ± 0.00133 |
| down | +0.0284245 ± 0.000431 | +0.144731 ± 0.00114 |
| left | +0.0172505 ± 0.000363 | +0.146792 ± 0.00124 |
| right | +0.0113445 ± 0.000205 | +0.136772 ± 0.00117 |
| shortest_path | +0.0191195 ± 0.00244 | +0.0388115 ± 0.000737 |
| one_hot_option | +0.00452876 ± 0.00117 | +0.00582347 ± 0.000932 |
| tile_option | +0.00680969 ± 0.00105 | +0.00693866 ± 0.000859 |

## Learning speed and interpretation

At 20,000 model transitions, tiles have substantially lower errors for both
learned hallway options. The final checkpoint reverses this ordering:

| Fixed option | Model steps | One-hot reward RMSE | Tile reward RMSE | One-hot common successor RMSE | Tile common successor RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| one_hot_option | 10,000 | 0.233621 | 0.124450 | 0.284890 | 0.090274 |
| one_hot_option | 20,000 | 0.140181 | 0.058227 | 0.147264 | 0.045708 |
| one_hot_option | 50,000 | 0.033285 | 0.037814 | 0.029750 | 0.035574 |
| tile_option | 10,000 | 0.220146 | 0.116020 | 0.303444 | 0.077815 |
| tile_option | 20,000 | 0.124654 | 0.052814 | 0.139596 | 0.041006 |
| tile_option | 50,000 | 0.029211 | 0.036021 | 0.026989 | 0.033927 |

The complete curves, rather than only an early or final checkpoint, are needed
for assessing the representation. Generalization helps these fixed option
models early while the tabular models eventually achieve greater accuracy at
this budget and constant learning rate. The primitive successor gap is larger
in common state units than native tile units. Since the tiles have full row
rank, this is not forced by an inability to express arbitrary state tables.
This study does not uniquely explain the learning dynamics or establish
asymptotic error floors; it contains no learning-rate or extended-budget sweep.

Each model-representation comparison holds its target option fixed. The SEs
measure model-training variation conditional on this one frozen policy pair,
not variation across independently trained options. The one-hot thresholds
from Milestone 4 are diagnostics here: its frozen policy and model seeds differ.
No claim of universal tile-coding superiority or inferiority follows.

## Validation

- **255 tests pass**, including 13 new cases covering tabular trajectory,
  parameter and key regression; nonzero traces; mixed-feature updates;
  policy/representation separation; actual hallway and terminal rewards;
  state-space metrics and matrix orientation; reference projection and decoder
  rank; JIT/VMAP; independent feature rollouts; and short full model cases.
- Pre-commit checks pass. The existing GitHub PR workflow runs the short suite;
  the long seeded experiment remains outside CI.
- All learned parameters and metrics are finite and terminal predictions are
  zero. All 14 option/representation reference models meet coefficient error
  bounds <=1e-10; the largest bound is 9.90e-11.
- Independently computed tile references agree with projected one-hot
  references to 1.62e-12 for successors and 8.54e-13 for rewards.
- Audits covering 80,000 independent rollouts pass the frozen `0.01 + 6*SE`
  tolerances, including both feature maps and all four starts for each learned
  option. The longest sampled execution is 37 actions; no truncation occurs.
- All 42 performance diagnostics pass: every native reward/successor mean
  error at least halves, and every one-hot model meets its declared absolute
  error limit. Tile wins are not required for scientific validity.
- A post-run NumPy-double recomputation from final saved parameters agrees
  with all reported per-seed errors within 8.90e-8. Final keys, environment
  positions and episode counts are identical across the two representation
  cases, confirming paired behavior experience.
- Two complete runs at the clean implementation revision produce byte-identical
  outputs: the frozen policies, both per-case metric reports, the combined
  report, all eight model archives, and all three figures (15 files total).
  There were no protocol deviations or unresolved experiment failures.

## Reproduce and inspect

```sh
uv sync --locked
uv run --locked --offline --no-cache pytest -q
MPLCONFIGDIR=.cache/matplotlib uv run --locked --offline --no-cache python -m rrs.experiments.tile_coded_models
```

Artifacts live under `artifacts/tile_coded_models/`: the frozen policy pair,
exact references, per-seed metric histories, representation tables and decoders,
and eight model archives at steps 0, 10k, 20k and 50k. Each archive records
seeds, axis ordering, model parameters and traces, environment state, keys,
episode counts and a hash of its frozen-policy source. The report records
archive hashes, provenance and all individual checks.

- [Reward-model learning](../figures/milestone_11_reward_models.png)
- [Common-state successor-model learning](../figures/milestone_11_successor_models.png)
- [Native successor-feature learning](../figures/milestone_11_native_successor_models.png)

Environment: Python 3.12.9, JAX/jaxlib 0.7.0 on CPU, NumPy 2.3.2,
Matplotlib 3.10.5. Learning and reported metric arithmetic use float32;
exact references, decoder construction and independent audits use double
precision. The independent metric recomputation checks the effect of rounding.

SHA-256 values (checkpoint hashes are recorded inside the report):

- Combined report: `77e790db302e81886fd3e0726b2f1d64a148fbe3b24c20c99417a7aa24c8d5ae`
- Frozen options: `815184ba84d6c160fd2326077e8d36ce13a7fcc56d8b70a2d896aa765de52ed6`
- Reward figure: `ffc6e522821b219aa164219c336572ff635da27dc19f8102a8830ab2dec02ae4`
- Common successor figure: `7e8b824513a368f49e080e3c372883ed6fde5a06d735a0f0a40d00237b288bf1`
- Native successor figure: `3c8dc2a1e992e36d1a6fedade5173bb8ab9101bd909825a65ffd3f022976ad55`

## Next milestone

Milestone 12 should evaluate planning with the saved model checkpoints, using
exact-model controls and independently measured task returns. Earlier accurate
option models may help planning sooner, while primitive-model errors may offset
any benefit. That question remains open until the planning comparison runs.
