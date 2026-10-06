# Stochastic four-room foundation

Milestone 6, [issue #11](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/11),
branch `milestone/06-four-room-foundation` from master `025d300`.
Source: the locally saved paper, [Section 7 and Figure 6](https://arxiv.org/html/2202.03466v4#S7).
Scope: environment and exact primitive-action baseline. Multiple option learning,
model learning and planning comparisons follow in later milestones.

## Geometry and rewards

Coordinates are `(row, column)`, zero based from the top left, including the
outer wall. Transcription of Figure 6 (digits label hallway features):

```text
#############
#.....#.....#
#.....#.....#
#.....2.....#
#S....#.....#
#.....#.....#
##1####.....#
#.PPP.###3###
#.PPP.#.....#
#.PPP.#G....#
#.PPP.4.....#
#.....#.....#
#############
```

There are 104 traversable cells, including goal `(9,7)`, hence 103 nonterminal
one-hot features. Start is `(4,1)`. Hallways H1–H4 are `(6,2)`, `(3,6)`, `(7,9)`,
`(10,6)`, respectively. Twelve penalty cells occupy rows 7–10, columns 2–4.
Hallway labels are metadata, not extra environment state or rewards.

Actions UP, DOWN, LEFT, RIGHT have indices 0–3. The realized movement direction
equals the requested action with probability 2/3; each of the other three
directions has probability 1/9. A wall blocks the realized direction, leaving
the position unchanged. Aggregate the probabilities of coincident outcomes;
never redraw after hitting a wall or redistribute its probability to legal moves.

Reward is evaluated at the resulting position: +1 on first arrival at the goal,
-1 at a penalty cell, zero elsewhere. This includes remaining in a penalty cell
after a blocked movement, following the two-room arrival-reward convention. The
paper's per-step gray-region wording is interpreted consistently with that
convention. The goal is absorbing: all later steps return goal, reward 0 and
terminated=True. Only the experiment runner resets; step never resets or truncates.
Discount is gamma=0.99.

## Interfaces and independent implementations

The JAX environment follows the existing functional API:

```text
reset(key, params) -> State(position)
observe(state) -> coordinate array
step(key, state, action, params) -> (next_state, observation, reward, terminated)
outcomes(state, action, params) -> probabilities, positions, rewards, terminated
```

Params contain walls, penalties, start, goal and a 4x4 movement-probability table.
The outcomes API enumerates the four absolute realized directions before merging
identical results. Step samples one of these with `jax.random.categorical` on the
log probabilities using the supplied key. Inputs must be legal positions, action
indices 0–3 and a valid row-stochastic probability table. Eager, JIT and VMAP must
agree for identical keys. Legal-position enumeration is a host-side helper.

Use the existing separate coordinate-to-one-hot representation; omit the goal
from its feature list. Environment code must not construct learning features.

The independent Python reference duplicates geometry with explicit coordinate
conditions, without importing the JAX environment or sharing its movement helper.
Its enumerated distributions merge coincident `(position, reward, terminal)`
outcomes. Sampling uses a caller-owned `random.Random`, drawing an integer 0–8:
0–5 select the intended direction and 6–8 select the other directions in action
order. This independently tests the probability rule; seeds need not produce the
same trajectories across the Python and JAX generators.

## Exact stochastic baseline

Represent a finite MDP by sparse outcomes `(probability, next_state, reward,
terminated)` for each state and action. Validate dimensions, state indices,
finite rewards, nonnegative finite probabilities and row mass within 1e-12 of 1.
Keep reward and terminal flags on each outcome, including terminal arrival reward.
Use Python double precision, independently of JAX's default float32:

```text
Q(s,a) = sum_outcomes p * (r + gamma * (0 if terminated else V(next_state)))
V_new(s) = max_a Q(s,a)
```

Use synchronous value iteration from zero. Recompute and report the Bellman
residual at the returned vector; its value-error bound is residual/(1-gamma).
Collect greedy ties within 1e-12 and use the first maximizing action for rollouts.
Also solve `(I-gamma*P_pi)V=r_pi` for that policy, retaining zero terminal rows
and including discounted closed cycles. Compare values from both methods.
Retain the existing deterministic oracles and their behavior.

## Frozen checks and experiment (before implementation/results)

- Compare JAX and independent reference distributions at all 104*4 state-action
  pairs, including terminal. Aggregated probabilities agree within 1e-7;
  positions, rewards and terminal flags agree exactly. Hand checks cover
  corners, hallway walls, entering/leaving penalties and goal arrivals.
- Check JAX step sampling at `(1,1)`, H1 `(6,2)`, `(9,8)` and penalty `(9,3)`,
  all four requested actions, using 18,000 independent keys from seed 6100 for
  each pair. For every possible outcome, empirical probability must be within
  `0.005 + 6*sqrt(p*(1-p)/18000)` of the independent reference. No other outcome
  may occur. Reusing keys across pairs is permitted; these are marginal tests.
- Test eager/JIT/VMAP agreement, repeatable keys, terminal zero features, and
  an independent hand-computed stochastic Bellman example with mixed terminal
  and continuing outcomes. Include stochastic self-loops and deterministic
  models as special cases, plus invalid distributions and nonconvergence.
- Baseline value iteration: gamma 0.99, residual tolerance 1e-12, maximum
  20,000 iterations. Require residual/(1-gamma) <= 1e-10, finite values and zero
  goal value. Independent policy evaluation must have error bound <= 1e-10
  and agree with value iteration at every state within 1e-9.
- Audit greedy-policy returns with independent Python rollouts from S `(4,1)`,
  H1 `(6,2)`, H3 `(7,9)` and penalty `(9,3)`: 20,000 episodes per start, seeds
  6000–6003 respectively. Each start uses one fresh `random.Random(seed)` stream.
  Maximum length is 3,000 steps. Record every signed discounted return and
  episode length, means and standard errors, and truncation counts. Bound any
  omitted tail by `gamma**3000/(1-gamma)`; require no truncated audit episode.
  Each mean must match the oracle within `0.005 + 6*SE + tail_bound`.
- Run the full baseline command twice. JSON and scientific figures must be
  byte-identical in the same software environment. Record versions, seeds,
  configuration, exact values and policies, residuals and code revision.

Freeze all tolerances before observing results. Preserve and diagnose failures
without tuning the geometry, seeds or thresholds to make them pass. In particular,
the optimal stochastic value is an output of the oracle, not an assumed route
discount or a copied paper curve value.

## Artifacts and workflow

Save figures under `figures/milestone_06_*.png`, an ignored JSON report under
`artifacts/four_room/`, and reproducibility/results notes in `notes/milestone_06.md`.
Include a labeled geometry/policy/value figure and an oracle-versus-rollout audit
figure. Short tests run in the existing every-PR GitHub Actions workflow; the
80,000-episode scientific audit runs locally. Commit locally and pause for review
before pushing. Leave README.md untouched and uncommitted.
