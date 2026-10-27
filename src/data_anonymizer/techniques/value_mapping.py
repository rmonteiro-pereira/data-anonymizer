"""Apply a text transformation to the present values of a column, leaving gaps as gaps."""

from collections.abc import Callable

import pandas as pd

TextTransform = Callable[[str], str]


def map_present(values: pd.Series, transform: TextTransform) -> pd.Series:
    """Transform every present cell on its own."""
    result = values.astype(object).copy()
    present = values.notna()
    result[present] = values[present].astype(str).map(transform)
    return result


def map_distinct(values: pd.Series, transform: TextTransform) -> pd.Series:
    """Transform each distinct value once and reuse the result for its repeats.

    Use it for deterministic transformations; slow ones such as Argon2 then cost
    one call per distinct value instead of one per row.
    """
    present = values.dropna().astype(str)
    mapping = {original: transform(original) for original in present.unique()}
    result = values.astype(object).copy()
    result[values.notna()] = present.map(mapping)
    return result
