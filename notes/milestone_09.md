# Milestone 9: Four-room planning with reward-respecting options

Issue [#17](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/17),
branch `milestone/09-four-room-planning`, based on master `535741b`. The frozen
protocol is in [four_room_planning_contract.md](four_room_planning_contract.md).

## Result

The exact primitive planner reached the known start-state optimum (0.725118385)
within the 1e-5 value and return tolerances. The exact optimal hallway option
models and exact models of the frozen Milestone 7 options also reached the same
planned value and independently evaluated return. Their mean first 95%-of-optimum
look-ahead budgets were 1,760 for optimized options, 8,853 for frozen options,
and 19,787 for primitives; all 30 runs in each exact case reached the threshold.

| Milestone 8 model transitions | Model set | Planned start value | Actual return | Runs reaching 95% |
| ---: | --- | ---: | ---: | ---: |
| 0 | Actions | 0.000000 | -0.005322 | 0/30 |
| 0 | Actions + options | 0.000000 | -0.005322 | 0/30 |
| 10,000 | Actions | 0.512231 | 0.502737 | 1/30 |
| 10,000 | Actions + options | 0.512850 | 0.496314 | 1/30 |
| 20,000 | Actions | 0.714137 | 0.545696 | 21/30 |
| 20,000 | Actions + options | 0.718131 | 0.546124 | 21/30 |
| 30,000 | Actions | 0.737893 | 0.570375 | 26/30 |
| 30,000 | Actions + options | 0.749264 | 0.589188 | 29/30 |
| 40,000 | Actions | 0.735352 | 0.556848 | 25/30 |
| 40,000 | Actions + options | 0.754528 | 0.557213 | 28/30 |
| 200,000 | Actions | 0.752272 | 0.613296 | 28/30 |
| 200,000 | Actions + options | 0.765603 | 0.617990 | 30/30 |

Option-backed planning was faster to reach the planned-value threshold at the
later learned checkpoints, but planned values exceeded actual reselected-policy
returns. At 200,000 model transitions, the option-backed plan finished at
0.765603 while the independently evaluated return was 0.617990. This is the
measured effect of planning with option continuation and evaluating by
reselecting a model after every primitive action; the protocol reports both
quantities separately.

All six acceptance checks passed: values remained finite, terminal values were
zero, every case used 1,600,000 look-aheads, policy-evaluation error bounds were
at most 1e-8, and the exact primitive final value vector and return met the
1e-5 oracle tolerances. The full experiment was repeated; the report, 15 weight
traces and three figures were byte-identical across both runs.

Figures: [exact references](../figures/milestone_09_exact_references.png),
[learned models](../figures/milestone_09_learned_models.png), and
[model maturity](../figures/milestone_09_model_maturity.png). Full per-run data
and compressed weights are preserved under the ignored
`artifacts/four_room_planning/` directory.

Validation: `uv run --locked --offline --no-cache pytest -q` — 214 passed.
