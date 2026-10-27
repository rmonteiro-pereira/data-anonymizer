"""Write new values into a table for the rows that match given keys, on the table's engine.

Steps such as Mondrian generalization, joint microaggregation and l-diversity enforcement
decide in Python, on a small summary of the table, what each group of rows should hold.
This module sends those decisions back: a lookup table is built on the same engine and
joined to the data, so a Polars or DuckDB table never has to be rebuilt row by row.

A key is the values of one or more columns, and a missing value is a key like any other.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import narwhals as nw

from oblitus.engines import Engine, WorkingFrame, collect, lookup_frame

# Separates the parts of a joined key; it does not occur in ordinary text.
_PART_SEPARATOR = "\x1f"
_MISSING_PART = "\x00missing"

_JOIN_KEY = "__da_rewrite_key"
_ROW_ORDER = "__da_rewrite_order"
_KEY_PART = "__da_rewrite_part_"
_NEW_VALUE = "__da_rewrite_new_"


def distinct_rows(frame: WorkingFrame, columns: Sequence[str]) -> "nw.DataFrame[Any]":
    """One row per distinct combination of `columns`, with its row count in `count`.

    Rows come back in a fixed order, so the same data gives the same answer on every engine.
    """
    counted = frame.group_by(list(columns)).agg(nw.len().alias("count"))
    return collect(counted).sort(list(columns), nulls_last=True)


def overwrite_by_key(
    frame: WorkingFrame,
    key_columns: Sequence[str],
    keys: Sequence[tuple[Any, ...]],
    replacements: Mapping[str, tuple[Sequence[Any], Any]],
) -> WorkingFrame:
    """Replace columns for the rows whose key is listed.

    `keys` holds one tuple per group, in the order of `key_columns`. `replacements` maps a
    column to its new values, one per key, and the Narwhals type those values have. Every row
    of `frame` must match a key, because the new column holds only what the lookup gives it.
    In-memory tables keep their row order.
    """
    schema = frame.collect_schema()
    part_names = [f"{_KEY_PART}{position}" for position in range(len(key_columns))]
    new_names = {column: f"{_NEW_VALUE}{position}" for position, column in enumerate(replacements)}
    columns: dict[str, Sequence[Any]] = {}
    dtypes: dict[str, Any] = {}
    for position, part_name in enumerate(part_names):
        columns[part_name] = [key[position] for key in keys]
        dtypes[part_name] = schema[key_columns[position]]
    for column, (values, dtype) in replacements.items():
        columns[new_names[column]] = values
        dtypes[new_names[column]] = dtype

    engine = Engine.DUCKDB if isinstance(frame, nw.LazyFrame) else Engine.POLARS
    lookup = lookup_frame(frame, engine, columns, dtypes)
    lookup = lookup.with_columns(_joined_key(part_names).alias(_JOIN_KEY)).drop(*part_names)
    keyed = frame.with_columns(_joined_key(key_columns).alias(_JOIN_KEY))
    joined = _left_join(keyed, lookup)
    replaced = [nw.col(new_names[column]).alias(column) for column in replacements]
    return joined.with_columns(*replaced).drop(_JOIN_KEY, *new_names.values())


def _joined_key(columns: Sequence[str]) -> nw.Expr:
    parts = [nw.col(column).cast(nw.String).fill_null(_MISSING_PART) for column in columns]
    return nw.concat_str(parts, separator=_PART_SEPARATOR)


def _left_join(frame: WorkingFrame, lookup: WorkingFrame) -> WorkingFrame:
    if isinstance(frame, nw.LazyFrame):
        lazy_lookup = lookup if isinstance(lookup, nw.LazyFrame) else lookup.lazy()
        return frame.join(lazy_lookup, how="left", on=_JOIN_KEY)
    # A left join is not promised to keep the left order, and callers line rows up by position.
    numbered = frame.with_row_index(_ROW_ORDER)
    joined = numbered.join(collect(lookup), how="left", on=_JOIN_KEY)
    return joined.sort(_ROW_ORDER).drop(_ROW_ORDER)
