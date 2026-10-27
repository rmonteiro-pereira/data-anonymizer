"""Apply a text transformation in Python to the values of a table column.

Python code cannot run inside a SQL engine, so the values are collected, transformed
and joined back as a lookup. Gaps stay gaps.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import narwhals as nw

from oblitus.table import Table

TextTransform = Callable[[str], str]


def canonical_value(value: object) -> str:
    """Return one value as text, writing whole-number floats without `.0`.

    An integer ID column turns into floats as soon as it has a gap, so `123.0` and
    `123` must get the same pseudonym.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def lookup_distinct(table: Table, column: str, transform: TextTransform) -> nw.Expr:
    """Transform each distinct value of a table column once, and join the results back.

    Gaps stay gaps. Use it for deterministic transformations: slow ones such as
    Argon2 cost one call per distinct value, and the work runs on any engine.
    """
    mapping = {value: transform(canonical_value(value)) for value in table.distinct_values(column)}
    return table.lookup(column, mapping, nw.String)


def lookup_per_row(table: Table, column: str, transform: TextTransform) -> nw.Expr:
    """Transform every present value on its own, so equal values can get different results.

    For randomized transformations such as a fresh salt or nonce per cell. The lookup is
    keyed on a column that tells every row apart, which means it collects one value per
    row into Python. Prefer `lookup_distinct` whenever the transformation is deterministic.
    """
    row_key = table.row_identity
    pairs = _collected(table.frame.select(nw.col(row_key), nw.col(column)).drop_nulls())
    mapping = {
        key: transform(canonical_value(value))
        for key, value in zip(pairs.get_column(row_key), pairs.get_column(column), strict=True)
    }
    return table.lookup(row_key, mapping, nw.String)


def present_counts(table: Table, column: str) -> Counter[Any]:
    """How many rows hold each distinct present value of the column."""
    grouped = table.frame.select(nw.col(column)).drop_nulls().group_by(column).agg(nw.len())
    return Counter(dict(_collected(grouped).iter_rows()))


def distinct_entity_counts(table: Table, column: str, by: str) -> Counter[Any]:
    """How many distinct entities (values of `by`) have each present value of the column.

    The entity is read from its original values, so an earlier action that replaced it does
    not change the count.
    """
    entity = table.original_column(by)
    pairs = table.frame.select(nw.col(column), nw.col(entity)).drop_nulls().unique()
    grouped = pairs.group_by(column).agg(nw.len())
    return Counter(dict(_collected(grouped).iter_rows()))


def distinct_entities_among(table: Table, column: str, by: str, values: list[Any]) -> int:
    """How many distinct entities (values of `by`) hold any of these values of the column."""
    entity = table.original_column(by)
    holders = table.frame.filter(nw.col(column).is_in(values)).select(nw.col(entity).n_unique())
    return int(_collected(holders).item())


def _collected(frame: nw.DataFrame[Any] | nw.LazyFrame[Any]) -> nw.DataFrame[Any]:
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    return frame
