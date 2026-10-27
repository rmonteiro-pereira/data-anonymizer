"""Values that must differ from row to row, and text built from a template.

Unique integer keys and unique template strings are drawn without repeats. A template
that has fewer possible values than rows is an error, not a silent repeat.
"""

import numpy as np

from data_anonymizer.errors import SchemaError
from data_anonymizer.synthesis.schema import ColumnSchema

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RADIX = {"#": 10, "?": len(_LETTERS)}

# A template with at most this many values, or at most 20 per row, is drawn by position in
# the list of all its values. A larger one has so few repeats that redrawing them is cheap.
_ENUMERABLE_SPACE = 10_000_000


def unique_integer_capacity(column: ColumnSchema) -> int | None:
    """Return how many distinct integers the column's range holds, or None when unbounded."""
    if column.maximum is None:
        return None
    lower, upper = _integer_range(column, rows=0)
    return max(upper - lower + 1, 0)


def unique_integers(column: ColumnSchema, rows: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `rows` distinct integers inside the column's range."""
    lower, upper = _integer_range(column, rows)
    available = upper - lower + 1
    if available < rows:
        raise SchemaError(
            f"{column.name}: {rows} unique integers do not fit between {lower} and {upper}"
        )
    return _distinct_integers(lower, upper, rows, rng)


def _distinct_integers(lower: int, upper: int, rows: int, rng: np.random.Generator) -> np.ndarray:
    available = upper - lower + 1
    if available <= rows * 4:
        return rng.permutation(np.arange(lower, upper + 1))[:rows]
    # Drawing from a huge range and dropping repeats avoids building the whole range.
    drawn: set[int] = set()
    while len(drawn) < rows:
        batch = rng.integers(lower, upper + 1, size=rows - len(drawn))
        drawn.update(int(value) for value in batch)
    return rng.permutation(np.fromiter(drawn, dtype="int64"))


def _integer_range(column: ColumnSchema, rows: int) -> tuple[int, int]:
    lower = 1 if column.minimum is None else int(np.ceil(column.minimum))
    upper = lower + rows * 10 if column.maximum is None else int(np.floor(column.maximum))
    return lower, upper


def template_values(column: ColumnSchema, rows: int, rng: np.random.Generator) -> list[str]:
    """Fill the column's template once per row, with all rows distinct when `unique` is set.

    In a template, `#` becomes a digit and `?` an uppercase letter.
    """
    template = column.template
    assert template is not None
    if not column.unique:
        return [fill_template(template, rng) for _ in range(rows)]
    space = _template_space(template)
    if rows > space:
        raise SchemaError(
            f"{column.name}: {rows} unique values do not fit the template {template!r}, "
            f"which has only {space} possible values"
        )
    if space <= max(_ENUMERABLE_SPACE, 20 * rows):
        positions = _distinct_integers(0, space - 1, rows, rng)
        return [_template_at(template, int(position)) for position in positions]
    return _redraw_until_distinct(template, rows, rng)


def fill_template(template: str, rng: np.random.Generator) -> str:
    """Replace each `#` by a random digit and each `?` by a random uppercase letter."""
    characters = []
    for character in template:
        if character == "#":
            characters.append(str(rng.integers(0, 10)))
        elif character == "?":
            characters.append(_LETTERS[int(rng.integers(0, len(_LETTERS)))])
        else:
            characters.append(character)
    return "".join(characters)


def _template_space(template: str) -> int:
    space = 1
    for character in template:
        space *= _RADIX.get(character, 1)
    return space


def _template_at(template: str, position: int) -> str:
    """Return the template value at a position in the list of all its values."""
    characters = list(template)
    for index in range(len(characters) - 1, -1, -1):
        radix = _RADIX.get(characters[index])
        if radix is None:
            continue
        position, digit = divmod(position, radix)
        characters[index] = str(digit) if radix == 10 else _LETTERS[digit]
    return "".join(characters)


def _redraw_until_distinct(template: str, rows: int, rng: np.random.Generator) -> list[str]:
    values: dict[str, None] = {}
    while len(values) < rows:
        for _ in range(rows - len(values)):
            values[fill_template(template, rng)] = None
    return list(values)
