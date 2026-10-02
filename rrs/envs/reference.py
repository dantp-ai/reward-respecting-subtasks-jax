"""Independent Python oracle for the fixed paper map.

Only the coordinate contract is shared with the JAX implementation. This module
intentionally duplicates geometry and semantics using explicit conditionals.
"""


def _traversable(row: int, column: int) -> bool:
    if not 1 <= row <= 6:
        return False
    if 1 <= column <= 6 or 8 <= column <= 13:
        return True
    return row == 3 and column == 7


def legal_positions() -> tuple[tuple[int, int], ...]:
    return tuple(
        (row, column)
        for row in range(8)
        for column in range(15)
        if _traversable(row, column)
    )


def step(position: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
    """Return (next position, reward, terminal) for one legal state and action."""
    row, column = position
    if not _traversable(row, column):
        raise ValueError("Position must be a traversable cell")
    if action not in (0, 1, 2, 3):
        raise ValueError("Action must be 0, 1, 2, or 3")
    if position == (6, 10):
        return position, 0.0, True

    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        column -= 1
    else:
        column += 1
    if not _traversable(row, column):
        row, column = position

    next_position = (row, column)
    if next_position == (6, 10):
        return next_position, 1.0, True
    if 1 <= row <= 5 and 2 <= column <= 5:
        return next_position, -1.0, False
    return next_position, 0.0, False
