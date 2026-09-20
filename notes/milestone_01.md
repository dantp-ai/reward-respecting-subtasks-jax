# Milestone 1: verified two-room environment and exact baseline

Implements [issue #1](https://github.com/dantp-ai/reward-respecting-subtasks-jax/issues/1)
on `milestone/01-two-room-foundation`, branched from `master`.

## Reproduce

Run from the repository root:

```sh
uv sync --locked
uv run pytest -q
uv run python -m rrs.experiments.two_room
```

The last command writes `artifacts/two_room/baseline.json` and
`artifacts/two_room/optimal_route.png`. Generated files are ignored by Git.
Use `--output-dir PATH` to choose another destination. The default seed is 0;
`--seed N` changes the explicit PRNG key, but this environment is deterministic.

## Result

Paper target: [Section 1, Figures 1–2](https://arxiv.org/html/2202.03466v4#S1.F1),
an optimal main-task start value of `0.99**17`, with a route around the penalty region.

Implementation result:

- 72 nonterminal states, one absorbing terminal goal, four actions.
- All 288 nonterminal transitions agree with the independent Python oracle.
- All 292 transitions, including terminal self-loops, pass JIT, VMAP, and combined checks.
- Optimal route: 18 actions, zero penalties, final reward `+1`.
- Start value: `0.842943193383927`; expected `0.842943193383927`.
- Value iteration: 22 synchronous sweeps, final Bellman residual `0.0`.
- Independent breadth-first search confirms the safe route length.
- 58 tests pass, including hand-calculated Bellman examples and terminal masking.

Parameters: `gamma=0.99`, residual tolerance `1e-12`, maximum 10,000 sweeps,
zero initial values, Python double precision for the exact solver, seed 0,
one deterministic run. Action ties use absolute tolerance `1e-12`;
the displayed route selects the first maximizing action in UP/DOWN/LEFT/RIGHT order.

Expected differences: multiple shortest safe routes exist in the right room;
the displayed route is one deterministic choice. This milestone measures the
exact main-task optimum and does not reproduce a learned planning curve.

Known ambiguity: the figures specify the map visually. Its coordinate
transcription, destination-based reward timing, and the project's absorbing
post-terminal convention are explicit in [the contract](two_room_contract.md).
No numerical discrepancy remains against this milestone's paper target.

Commit: each generated JSON report records the actual code revision and whether
tracked files were modified when it ran, along with Python and package versions.

Conclusion: environment and main-task baseline acceptance checks pass. The next
milestone can establish exact hallway-subtask solutions before option learning.

## Review guide

1. `notes/two_room_contract.md`: map, API, Bellman equation, fixed tolerances.
2. `rrs/envs/reference.py` and `rrs/envs/two_room.py`: independent transition implementations.
3. `rrs/rl/exact.py`: synchronous value iteration and convergence diagnostics.
4. `tests/`: semantic, exhaustive, transformation, and baseline oracle checks.
5. `rrs/experiments/two_room.py`: exact model enumeration, route plot, and JSON report.

The commit hooks use Ruff for linting, import ordering, complexity, and formatting.
The previous separate import sorter conflicted with Ruff, and Flake8 3.9.2 crashed
on Python 3.12. General file checks remain in pre-commit. README.md is excluded
from the milestone commits.
