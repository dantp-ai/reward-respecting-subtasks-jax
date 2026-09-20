# Two-room environment contract

Milestone 1, [issue #1](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/1).
Source: Sutton et al., [Section 1 and Figures 1–2](https://arxiv.org/html/2202.03466v4#S1.F1).

## Geometry

Coordinates are `(row, column)`, zero-based from the top left, including the
outer walls. The map is transcribed from the paper's figures:

```text
    012345678901234
 0  ###############
 1  #.PPPP.#......#
 2  #.PPPP.#......#
 3  #SPPPP.H......#
 4  #.PPPP.#......#
 5  #.PPPP.#......#
 6  #......#..G...#
 7  ###############
```

`#` is a wall, `P` a penalty cell, and `.` an ordinary cell. Start is `(3, 1)`,
hallway is `(3, 7)`, and goal is `(6, 10)`. Each room has 36 cells; including
the hallway and excluding the goal gives the paper's 72 nonterminal states.
The penalty rectangle has 20 cells: rows 1–5, columns 2–5.

## Transition and reward semantics

- Action IDs: `UP=0`, `DOWN=1`, `LEFT=2`, `RIGHT=3`.
- Moves are deterministic and cardinal. A blocked move leaves the position unchanged.
- Reward is computed from the resulting position: `+1` for entering the goal,
  `-1` for ending in a penalty cell, and `0` otherwise. A blocked move while
  in a penalty cell still costs `-1`; leaving a penalty cell for a safe cell costs `0`.
- Goal entry ends the episode. Subsequent steps stay at the goal, return `0`,
  and remain terminal. This absorbing post-terminal API behavior is a project
  convention; the paper stops episodes on goal entry. Reset is explicit.
- The environment does not apply discounting. The baseline uses `gamma=0.99`.
- Legal states and action IDs are preconditions of the JAX step function.

## Interfaces and ownership

The small flat `rrs` package keeps the initial checkout runnable without a build
backend. `envs`, `rl`, and `experiments` separate transitions, exact baselines,
and result generation. Later feature construction belongs outside the environment.

`default_params()` returns JAX arrays for walls, penalties, start, and goal.
`State(position)` is a JAX-compatible named tuple. `reset(key, params)` returns
a state. `observe(state)` returns a coordinate array; it currently equals the
position but is distinct from a learning feature vector.
`step(key, state, action, params)` returns
`(next_state, observation, reward, terminated)`.
Keys remain explicit but are unused by this deterministic environment.
There is no automatic reset, time limit, or learning feature construction.

The Python oracle uses tuples, bounds checks, and explicit conditionals. It
does not import the JAX environment or reuse its map, movement table, or reward logic.

The exact baseline enumerates all 73 traversable positions in row-major order,
including an absorbing goal row. Its deterministic model stores next-state
indices, rewards, and terminal flags for every action. Tabular indices belong
to this exact oracle; they do not prescribe future learning interfaces.

## Baseline equation and acceptance

Use synchronous value iteration initialized at zero:

```text
Q(s, a) = r(s, a) + gamma * (1 - done(s, a)) * V(next(s, a))
V_new(s) = max_a Q(s, a)
```

This is the primitive-action Bellman optimality backup for the discounted
return in paper Equation 1. Terminal transitions do not bootstrap; the goal
row has value zero. Report the final Bellman residual and all maximizing actions
(absolute tie tolerance `1e-12`), not just the last iterate's change.

Acceptance is fixed before implementation:

- All 288 nonterminal state/action transitions match the independent oracle.
- Explicit cases cover every border and dividing-wall segment, hallway passage,
  blocked movement, penalty entry/stay/exit, reward timing, and terminal behavior.
- After eager semantics pass, JIT and VMAP agree on all 292 transitions including
  the terminal state, with exact comparisons for discrete outputs and rewards.
- The returned optimal route reaches the goal in 18 actions without penalty.
  Its first 17 rewards are zero and its final reward is one, so its return is
  `gamma**17 = 0.8429431933839268`. Start-value absolute error must be at most
  `1e-12`, using Python double precision in the reference solver.
- Final Bellman residual must be at most `1e-12`. Independently computed
  shortest distances through safe cells must confirm the 18-action route.
- Tests, a route plot, and a JSON baseline report are reproducible from a
  locked environment. The report records parameters, seed, package versions,
  code revision, and whether tracked files have uncommitted changes.

Subtask solutions, learned options, and option models belong to later milestones.
