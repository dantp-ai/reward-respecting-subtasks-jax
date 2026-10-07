# Four reward-respecting hallway options

Milestone 7, [issue #13](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/13),
branch `milestone/07-four-room-options` from master `543bf6b`.
Sources: the locally saved paper's Sections 2–3 and 7, Equations 2–5, 9–11, UWT,
and Figures 6–7. Environment: [Milestone 6 contract](four_room_contract.md).
Model learning and planning comparisons remain subsequent milestones.

## Subtasks and stochastic references

Use the four hallway features in H1–H4 order: `(6,2)`, `(3,6)`, `(7,9)`, `(10,6)`.
Keep 103 row-major nonterminal one-hot features separate from 104 model states.
Freeze main-task weights at zero; replace each target weight by bonus 1 using
Equation 4. Each option receives the actual environment reward as its cumulant.
Stopping is allowed everywhere, checked only after at least one action. Goal
arrival retains reward +1 and forces stopping with stopping value zero.

For sparse stochastic outcomes `(p, next_state, reward, terminal)`, the exact
forced-action subtask oracle solves Equation 2 in Python double precision:

```text
arrival(s') = max(z(s'), gamma*V(s')) if stopping allowed, else gamma*V(s')
Q(s,a) = sum_outcomes p * (reward + (0 if terminal else arrival(s')))
V_new(s) = max_a Q(s,a); terminal source values = 0
```

Start values at zero, gamma=0.99, final Bellman residual <=1e-12, maximum 20,000
sweeps. Keep greedy ties within 1e-12, selecting the first action in UP/DOWN/LEFT/
RIGHT order for reference-policy evaluation. Oracle stopping compares `z>=gamma*V`.
Learning must retain Equation 9's `z>=V`, using pre-update critic values and
stopping on equality. Test the rules' distinction on a toy example and verify
their agreement for these particular zero-main-weight, unit-bonus references.

Evaluate fixed policies using the independent Python environment model, summing
over both action probabilities and stochastic environment outcomes. Normalize
only float32 policy-row roundoff (row sums must be within 1e-6 of one).

```text
V_pi(s) = E[r + beta(s')*z(s') + gamma*(1-beta(s'))*V_pi(s')]
```

Environment termination overrides beta and z. Source stopping never suppresses
the first action. In addition to expected subtask returns, compute probabilities
of stopping at the target, stopping there without any penalty arrival, and any
penalty arrival before option termination. Include penalty on the final action.
Terminal arrival or stopping elsewhere fails target attainment. Reaching G can
still be optimal for the subtask: target attainment is a diagnostic, not a
universal success threshold.

Use independent sparse Bellman iteration. Certify return error via
residual/(1-gamma) <=1e-8. Bound undiscounted probabilities with lower/upper
iterations from 0/1 until the maximum gap <=1e-8; maximum 20,000 sweeps. Reject
unresolved nontermination explicitly instead of truncating or changing a policy.
Report full state vectors, errors and iteration counts. Reference policies must
match their optimized subtask values within 2e-8 over all states.

## Learning from shared experience

Reuse the existing feature-facing `actor_critic_update`, without altering its
equations. Four independent actor/critic/trace sets receive each identical
behavior transition; each computes its own policy ratio, stopping decision and
TD error. Behavior is uniform over four actions, starts at S, and resets only
after learning the terminal transition. Option stopping does not reset behavior.

Parameters: gamma=0.99; critic and actor step sizes both 0.05 (Section 7);
both lambdas zero; all weights/traces initially zero. No oracle values or model
tables enter training. Use float32 JAX. Each run owns `jax.random.key(seed)`;
split each transition into next-key, action-key, environment-key, reset-key.
Sample the requested action uniformly with randint; pass environment-key to the
validated stochastic environment. The importance ratio is pi(requested action)/
0.25, not a ratio for the realized movement direction.

## Frozen experiment and acceptance

This protocol is committed before any training results are observed:

- 30 independent training seeds 7000–7029. Every run trains all four options from
  its shared trajectory. Option learners are independent across runs.
- 1,000,000 transitions per run; record critics every 2,000 transitions, including
  zero initialization. Preserve actor/critic snapshots at 200,000 and 1,000,000.
  Figure 7 spans 200,000 transitions and still shows H3 lagging. Preserve that
  paper-duration comparison and use the predeclared longer budget to assess all
  four options under the same convergence criteria. Do not describe the longer
  budget as the paper's protocol.
- Independently evaluate actual stochastic policies and learned stopping at
  initialization, 200,000 and 1,000,000 transitions. Report each option separately,
  with means and standard errors over seeds. Never select policies by performance.
- Critic RMSE uses all 103 nonterminal states against each exact subtask oracle.
  For **each** option, final mean critic RMSE <=0.40 and <=60% of its peak
  checkpoint mean RMSE. Final mean estimated start value must be within 0.15 of
  the exact optimum for that subtask.
- For **each** option, final mean actual start return must be within 0.15 of its
  exact subtask optimum; final mean actual-value RMSE over all nonterminal states
  must be <=0.40. Report 200,000-step results even if they fail these final gates.
- Every evaluated actual policy value must stay below the optimal subtask value
  within 2e-8. Hallway and goal must be stopping states for every trained option
  at both saved snapshots. All parameters, curves and evaluation metrics finite;
  terminal critic values zero. No cross-option average can satisfy a failed gate.
- Report target/safe-target/penalty probabilities from S and their uniform average
  over nonterminal starts, alongside reference probabilities. Do not require zero
  penalty exposure in a stochastic environment.
- Repeat the full experiment. Reports, compressed policy snapshots and PNGs must
  be byte-identical in this software environment. Record seeds, parameters, axes,
  feature/state order, versions, code revision and snapshot hashes.

These are project acceptance thresholds, not numerical claims from the paper.
Preserve failures and diagnose equations, stopping, stochastic outcomes, indices,
ratios, shared transitions and reset timing. Do not tune seeds, thresholds or
training duration after seeing the results.

## Tests and artifacts

Test stochastic subtask backups, reward/terminal/stopping timing, fractional
stopping, target versus safe attainment, penalty exposure, cycles and deterministic
special cases. Test independent learner updates on the same transition, distinct
ratios/targets, no reset at option stopping, terminal learning before reset,
eager/scan/JIT/VMAP agreement and short seeded reproducibility in CI.

Save figures as `figures/milestone_07_*.png`, ignored JSON reports and deterministic
gzip policy snapshots under `artifacts/four_room_options/`, and results in
`notes/milestone_07.md`. Plot learning and actual returns, plus all four stochastic
policies and stopping maps for fixed seed 7000. Keep README.md untouched and
uncommitted. Commit locally and pause for user review before pushing.
