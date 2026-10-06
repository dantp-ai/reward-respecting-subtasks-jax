"""Independent Python four-room dynamics; no shared JAX geometry or helpers."""

from typing import NamedTuple


class Outcome(NamedTuple):
    probability: float
    position: tuple[int, int]
    reward: float
    terminated: bool


def _traversable(row, column):
    if not (1 <= row <= 11 and 1 <= column <= 11):
        return False
    if column == 6:
        return row in (3, 10)
    if row == 6 and column < 6:
        return column == 2
    if row == 7 and column > 6:
        return column == 9
    return True


def legal_positions():
    return tuple((r, c) for r in range(13) for c in range(13) if _traversable(r, c))


def _validate(position, action):
    if not _traversable(*position):
        raise ValueError("Position must be traversable")
    if action not in (0, 1, 2, 3):
        raise ValueError("Action must be 0, 1, 2, or 3")


def _move(position, direction):
    if position == (9, 7):
        return position, 0.0, True
    row, column = position
    if direction == 0:
        row -= 1
    elif direction == 1:
        row += 1
    elif direction == 2:
        column -= 1
    else:
        column += 1
    if not _traversable(row, column):
        row, column = position
    result = (row, column)
    if result == (9, 7):
        return result, 1.0, True
    reward = -1.0 if 7 <= row <= 10 and 2 <= column <= 4 else 0.0
    return result, reward, False


def distribution(position, action) -> tuple[Outcome, ...]:
    """Merge probabilities of all directions producing the same outcome."""
    _validate(position, action)
    merged = {}
    for direction in range(4):
        result = _move(position, direction)
        p = 2 / 3 if direction == action else 1 / 9
        merged[result] = merged.get(result, 0.0) + p
    return tuple(Outcome(p, *result) for result, p in sorted(merged.items()))


def step(position, action, rng):
    """Draw from an independent nine-ticket sampler using a caller-owned RNG."""
    _validate(position, action)
    ticket = rng.randrange(9)
    direction = (
        action if ticket < 6 else tuple(a for a in range(4) if a != action)[ticket - 6]
    )
    return _move(position, direction)
