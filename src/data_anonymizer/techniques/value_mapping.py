"""Apply a text transformation to the present values of a column, leaving gaps as gaps."""

from collections.abc import Callable

import narwhals as nw
import pandas as pd

from data_anonymizer.table import Table

TextTransform = Callable[[str], str]


def canonical_text(values: pd.Series) -> pd.Series:
    """Return the present values as text, writing whole-number floats without `.0`.

    An integer ID column turns into floats as soon as it has a gap, so `123` would
    become `123.0` and get a different pseudonym than in a table without gaps.
    """
    present = values.dropna()
    if pd.api.types.is_float_dtype(present) and (present % 1 == 0).all():
        return present.astype("int64").astype(str)
    return present.astype(str)


def map_present(values: pd.Series, transform: TextTransform) -> pd.Series:
    """Transform every present cell on its own."""
    result = values.astype(object).copy()
    result[values.notna()] = canonical_text(values).map(transform)
    return result


def map_distinct(values: pd.Series, transform: TextTransform) -> pd.Series:
    """Transform each distinct value once and reuse the result for its repeats.

    Use it for deterministic transformations; slow ones such as Argon2 then cost
    one call per distinct value instead of one per row.
    """
    present = canonical_text(values)
    mapping = {original: transform(original) for original in present.unique()}
    result = values.astype(object).copy()
    result[values.notna()] = present.map(mapping)
    return result


def canonical_value(value: object) -> str:
    """Return one value as text, writing whole-number floats without `.0`.

    The table version of `canonical_text`: `123.0` and `123` get the same pseudonym.
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
