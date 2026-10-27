"""The table a plan is applied to, on whichever engine holds it.

Techniques read and replace columns through this class, so the same technique code
runs in memory on Polars and as SQL on DuckDB. It adds the few steps an expression
cannot express on its own: keyed random numbers per row or per entity, lookups of
values computed in Python, and one-row statistics.

Helper columns get the `__da_` prefix and are dropped when the table is handed back.
"""

from collections.abc import Mapping
from itertools import count
from typing import Any

import narwhals as nw

from data_anonymizer.engines import (
    UNIFORMS_PER_DIGEST,
    Engine,
    WorkingFrame,
    add_keyed_uniforms,
    collect,
    lookup_frame,
    to_native,
    to_working_frame,
)

HELPER_PREFIX = "__da_"
_ROW_POSITION = f"{HELPER_PREFIX}row"
ROW_KEY_COLUMN = f"{HELPER_PREFIX}row_key"
_ORIGINAL_PREFIX = f"{HELPER_PREFIX}original_"


class Table:
    """A table being transformed column by column.

    `row_key` names a column that identifies each row, such as a record ID. Random
    numbers per row are keyed on it, so the same row gets the same noise on every
    engine and in every run with the same key. In-memory tables without one use the
    row position. Lazy tables need one, because their row order is not fixed.
    """

    def __init__(self, frame: WorkingFrame, engine: Engine, *, row_key: str | None = None):
        if row_key is not None and row_key not in frame.columns:
            raise ValueError(f"row key {row_key!r} is not in the table")
        self._engine = engine
        self._original_columns = list(frame.columns)
        self._names = count()
        if isinstance(frame, nw.DataFrame):
            # Joins may reorder rows, so in-memory tables remember their order.
            frame = frame.with_row_index(_ROW_POSITION)
            row_key = _ROW_POSITION if row_key is None else row_key
        self._frame = frame
        self._row_key = row_key

    @classmethod
    def from_native(cls, native: Any, *, row_key: str | None = None) -> "Table":
        """Wrap a pandas, Polars or DuckDB frame."""
        frame, engine = to_working_frame(native)
        return cls(frame, engine, row_key=row_key)

    @property
    def engine(self) -> Engine:
        """The engine the user's table came from."""
        return self._engine

    @property
    def frame(self) -> WorkingFrame:
        """The current frame, helper columns included."""
        return self._frame

    @property
    def columns(self) -> list[str]:
        """The user's columns, in their original order."""
        return list(self._original_columns)

    @property
    def row_key(self) -> str:
        """The column that identifies rows, for lookups keyed per row.

        In-memory tables without a named key use a helper column of row positions.
        """
        return self._key_column()

    def dtype(self, column: str) -> Any:
        """The Narwhals type of a column."""
        return self._frame.collect_schema()[column]

    def statistics(self, **aggregations: nw.Expr) -> dict[str, Any]:
        """Compute one-row aggregates, such as a mean or a quantile, as Python values."""
        selected = self._frame.select(**aggregations)
        row = collect(selected).row(0)
        return dict(zip(aggregations, row, strict=True))

    def distinct_values(self, column: str) -> list[Any]:
        """The column's distinct present values."""
        distinct = self._frame.select(nw.col(column)).unique().drop_nulls()
        return collect(distinct).get_column(column).to_list()

    def column_values(self, column: str) -> list[Any]:
        """Every value of a column, in row order for in-memory tables."""
        return collect(self._ordered().select(nw.col(column))).get_column(column).to_list()

    def uniforms(self, key: bytes, how_many: int = 1, *, by: str | None = None) -> list[nw.Expr]:
        """Keyed uniform numbers in (0, 1): one set per row, or per value of `by`.

        With `by`, every row of the same entity gets the same numbers, so for example
        all dates of one person can move by the same offset.
        """
        if how_many > UNIFORMS_PER_DIGEST:
            raise ValueError(f"at most {UNIFORMS_PER_DIGEST} uniforms per call")
        source = self._original_name(self._key_column() if by is None else by)
        names = [self._helper_name() for _ in range(how_many)]
        self._frame = add_keyed_uniforms(
            self._frame, self._engine, key=key, source=source, names=names
        )
        return [nw.col(name) for name in names]

    def lookup(self, column: str, mapping: Mapping[Any, Any], dtype: Any) -> nw.Expr:
        """Join a mapping from the column's values to new values of type `dtype`.

        Values missing from the mapping, and missing values, come back as missing.
        """
        key_name = self._helper_name()
        value_name = self._helper_name()
        lookup = lookup_frame(
            self._frame,
            self._engine,
            {key_name: list(mapping), value_name: list(mapping.values())},
            {key_name: self.dtype(column), value_name: dtype},
        )
        self._join_left(lookup, column, key_name)
        return nw.col(value_name)

    def original_column(self, column: str) -> str:
        """The name under which the column's values before any replacement are kept.

        Grouping by another column reads its original values, so an earlier action
        that replaced that column does not change the groups.
        """
        return self._original_name(column)

    def attach(self, values: nw.Series[Any]) -> nw.Expr:
        """Add a column computed in Python, one value per row in row order.

        Only in-memory tables have a row order to align with.
        """
        if self._engine.is_lazy:
            raise ValueError("this technique runs on in-memory tables only")
        name = self._helper_name()
        ordered = collect(self._ordered())
        self._frame = ordered.with_columns(values.alias(name))
        return nw.col(name)

    def original_pandas(self) -> Any:
        """The user's columns as they were before any replacement, as a pandas frame.

        For techniques still written for pandas columns, which group by other columns.
        """
        ordered = collect(self._ordered())
        selected = ordered.select(
            [nw.col(self._original_name(column)).alias(column) for column in self.columns]
        )
        return selected.to_pandas()

    def reorder_values(self, column: str, sort_key: nw.Expr) -> nw.Expr:
        """Hand the column's present values out again, in the order of `sort_key`.

        The row with the k-th smallest sort key gets the column's k-th smallest value,
        so the column keeps exactly the same values while they move between rows.
        Missing values stay where they are. A random sort key shuffles; the value's
        own rank plus a little noise swaps values with near neighbours.
        """
        value = nw.col(column)
        present_key = nw.when(~value.is_null()).then(sort_key)
        position = self.add(present_key.rank("ordinal"))
        rank_name = self._helper_name()
        value_name = self._helper_name()
        ranked = self._frame.filter(~value.is_null()).select(
            value.rank("ordinal").alias(rank_name), value.alias(value_name)
        )
        self._join_left(ranked, position, rank_name)
        return nw.col(value_name)

    def add(self, expression: nw.Expr) -> str:
        """Store an intermediate result as a helper column and return its name.

        Windows such as `over` need a column name to partition by.
        """
        name = self._helper_name()
        self._frame = self._frame.with_columns(expression.alias(name))
        return name

    def replace(self, column: str, expression: nw.Expr) -> None:
        """Replace a column with the result of an expression.

        The first replacement keeps a copy of the original values, which later
        actions read through `original_column`. Polars shares the data and DuckDB only adds
        a projection, so the copy costs nothing.
        """
        snapshot = _ORIGINAL_PREFIX + column
        if snapshot not in self._frame.columns:
            self._frame = self._frame.with_columns(nw.col(column).alias(snapshot))
        self._frame = self._frame.with_columns(expression.alias(column))

    def to_native(self) -> Any:
        """Hand the table back in the user's form, without helper columns."""
        return to_native(self.released_frame(keep_row_key=False), self._engine)

    def original_frame(self) -> WorkingFrame:
        """The user's columns as they were before any replacement, plus `ROW_KEY_COLUMN`.

        The key column holds each row's key, so this table and `released_frame` can be
        lined up row by row even when the key column itself was replaced.
        """
        columns = [nw.col(self._original_name(name)).alias(name) for name in self.columns]
        key = nw.col(self._original_name(self._key_column())).alias(ROW_KEY_COLUMN)
        return self._ordered().select(*columns, key)

    def released_frame(self, *, keep_row_key: bool = True) -> WorkingFrame:
        """The user's columns as they are now, in the engine's working form.

        With `keep_row_key`, `ROW_KEY_COLUMN` holds each row's original key, so the
        result can be lined up with `original_frame`.
        """
        ordered = self._ordered()
        if not keep_row_key:
            return ordered.select(self._original_columns)
        key = nw.col(self._original_name(self._key_column())).alias(ROW_KEY_COLUMN)
        return ordered.select(*self._original_columns, key)

    def _join_left(self, right: WorkingFrame, left_on: str, right_on: str) -> None:
        # Narwhals joins a lazy frame only with a lazy one, and an eager with an eager.
        joined: WorkingFrame
        if isinstance(self._frame, nw.LazyFrame):
            lazy_right = right if isinstance(right, nw.LazyFrame) else right.lazy()
            joined = self._frame.join(lazy_right, how="left", left_on=left_on, right_on=right_on)
        else:
            joined = self._frame.join(
                collect(right), how="left", left_on=left_on, right_on=right_on
            )
        self._frame = joined.drop(right_on, strict=False)

    def _ordered(self) -> WorkingFrame:
        if _ROW_POSITION in self._frame.columns:
            return self._frame.sort(_ROW_POSITION)
        return self._frame

    def _key_column(self) -> str:
        if self._row_key is None:
            raise ValueError(
                "random numbers per row on a lazy table need a row key; "
                "name a column that identifies each row"
            )
        return self._row_key

    def _original_name(self, column: str) -> str:
        if column not in self._frame.columns:
            raise ValueError(f"column {column!r} is not in the table")
        snapshot = _ORIGINAL_PREFIX + column
        return snapshot if snapshot in self._frame.columns else column

    def _helper_name(self) -> str:
        return f"{HELPER_PREFIX}{next(self._names)}"
