# Exact hallway options and model contract

Milestone 2, [issue #3](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/3).
Uses the validated [two-room environment](two_room_contract.md).
Sources: paper [Sections 2–4](https://arxiv.org/html/2202.03466v4#S2),
especially Equations 2, 4, 5, 9, 12, 15, and 17.

## Features and fixed subtask inputs

The representation layer maps coordinates to 72 one-hot features in row-major
order, omitting the terminal goal. The goal maps to the zero vector. State
enumeration in the exact environment model still has 73 rows; feature indices
and model-row indices are therefore separate, especially after the goal row.
The hallway `(3, 7)` is one of the 72 nonterminal features.

For the illustrative reward-respecting subtask, freeze main-task weights at
zero and use hallway bonus weight 1. These are the initial main-task estimates,
not the optimal main-task values computed in Milestone 1. Equation 4 is:

```text
z(x) = dot(w, x) - w[h] * x[h] + bonus * x[h]
```

Thus `z(hallway)=1` and all other stopping values are zero. The cumulant is the
actual transition reward. Stopping is allowed at every nonterminal state.

The comparison uses cumulant `-1` per action, zero stopping values, and permits
stopping only at the hallway or terminal goal, as specified in Section 2.
Both options' environment models always use actual environment rewards, even
though the shortest-path option is optimized using a different cumulant.

## Execution and subtask Bellman equation

An option initiated in a nonterminal state takes at least one action. Its
stopping function is checked at the destination after that action; there is no
zero-duration option. This includes initiation at the hallway. Initiation at
the already-terminal goal returns no actions, zero reward, and zero features.
Environment termination overrides option continuation.

For a destination `s'`, Equation 2 permits either taking the stopping value now
or continuing with a discounted future return:

```text
arrival(s') = max(z(s'), gamma * V(s'))  if stopping is allowed
           = gamma * V(s')             otherwise
Q(s,a) = cumulant(s,a) + arrival(next(s,a))
```

For a terminal transition, `arrival=0`. The terminal source row has value zero.
Use synchronous `V_new(s)=max_a Q(s,a)` from zero, Python double precision,
`gamma=0.99`, and final Bellman residual tolerance `1e-12` (maximum 10,000 sweeps).
Report every action within `1e-12` of the maximum. Choose the first such action
in UP/DOWN/LEFT/RIGHT order for a deterministic reference policy.
Stop on equality: `can_stop(s) and z(s) >= gamma * V(s)`.

Equation 9 instead compares `z(s)` with the *undiscounted* learned `V(s)`.
This oracle optimizes the return actually defined in Equation 2. The two rules
agree for the fixed illustrative subtask (zero stopping values except a unit
hallway bonus), and this agreement is checked over every state. They need not
agree for arbitrary nonzero main-task weights. Do not silently reuse this exact
optimal-stopping rule as the later Equation 9 learning rule.

Stopping is not hard-coded to just the hallway and goal for the reward-respecting
option. Some penalty-region states can have a nonpositive forced-action value
and therefore prefer stopping for zero. Record these states explicitly rather
than changing the mathematical definition to match a picture.

The stopping reward at action count `K` is discounted by `gamma**(K-1)` in
Equation 2. From the start, the safe hallway route has 12 actions, zero environment
reward, and subtask value `gamma**11`. The shortest route has 6 actions and
passes through four penalty cells. Its *subtask* value is
`-sum(gamma**t for t in range(6))`; its *environment* reward model is
`-sum(gamma**t for t in range(4))`. These quantities must not be confused.

## Exact option models

For each fixed deterministic policy and stopping function, compute:

```text
R_o(s) = sum(gamma**t * reward[t] for t in range(K))
N_o(s) = gamma**K * x(final_state)
```

These are Equations 12 and 15. Use memoized one-step recurrences on the exact
transition table, checking for nonterminating cycles rather than truncating:

```text
R_o(s) = r(s,a) + gamma * (1 - beta(s')) * R_o(s')
N_o(s) = gamma * (beta(s') * x(s') + (1 - beta(s')) * N_o(s')
```

On environment termination, keep the immediate reward and return zero successor
features. The model excludes the artificial stopping bonus. A primitive action
is a constant policy with stopping always true, so its model must be immediate
environment reward and `gamma * next_features`.

Discount discrepancy: inserting Equation 17's printed successor target into
Equation 5 gives `gamma**(K-1) * x(final_state)`, whereas Equation 15 and the
planning interface require `gamma**K`. This milestone follows Equation 15.
The missing factor in the printed learning update is recorded for model learning;
the exact oracle does not redefine its target to hide the discrepancy.

The exact tables retain double precision. They can be exported as linear JAX
models with reward weights and successor columns in *feature* order, per
Equation 16. Prediction accepts a feature vector, not an environment state ID.
The normal JAX float32 export is checked within `1e-6`; exact oracle and rollout
comparisons use `1e-12`.

## Acceptance and independent checks

- Test one-hot/terminal encoding, Equation 4 on hand-calculated feature vectors,
  and a toy case distinguishing stopping now from discounted continuation.
- Reward-respecting start value equals `0.99**11` within `1e-12`; all final
  Bellman residuals are at most `1e-12`.
- The safe option reaches the hallway in 12 actions with no penalties; the
  comparison takes 6 actions and incurs four penalties.
- Check both policies, stopping decisions, and model predictions over every
  legal starting state using rollouts through the independent Python environment.
- Check all four primitive models, terminal masking, and nontermination errors.
- For these deterministic policies, exhaustive exact rollouts replace noisy
  Monte Carlo estimates; no statistical tolerance or random sampling is needed.
- Save route and policy comparisons under `figures/milestone_02_*.png`, with a
  reproducible JSON report and short result note. Learning remains out of scope.
