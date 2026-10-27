"""The same rows of the original and the released table, in memory as pandas.

Utility, disclosure and dominance checks compare each released value with the
original one, so they need both tables in memory, row for row. A table too large for
that is cut to a sample first. The sample is chosen by a keyed hash of each row's key,
so the same table and key always give the same rows, whatever order the engine
returns them in.
"""

from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

from oblitus.engines import Engine, WorkingFrame, add_keyed_uniforms, collect, first_row
from oblitus.table import ROW_KEY_COLUMN

# Largest number of rows compared in memory.
MAX_SAMPLE_ROWS = 200_000

_DRAW = "__da_sample_draw"


@dataclass(frozen=True, slots=True)
class AlignedSample:
    """Row `i` of `original` and row `i` of `released` describe the same record."""

    original: pd.DataFrame
    released: pd.DataFrame


class SampleAccumulator:
    """Builds the aligned sample of a table that arrives in chunks, in bounded memory.

    Each chunk adds its original and released rows. Only the rows with the smallest keyed
    draws so far are kept, which are the rows `align_sample` would pick from the whole table,
    so a chunked run compares the same rows as an in-memory run.
    """

    def __init__(self, key: bytes, max_rows: int = MAX_SAMPLE_ROWS) -> None:
        self._key = key
        self._max_rows = max_rows
        self._original: nw.DataFrame[Any] | None = None
        self._released: nw.DataFrame[Any] | None = None

    def add(self, original: "nw.DataFrame[Any]", released: "nw.DataFrame[Any]") -> None:
        """Add the rows of one chunk; both frames carry `ROW_KEY_COLUMN`, in the same order."""
        if self._original is None or self._released is None:
            candidates = (original, released)
        else:
            candidates = (
                nw.concat([self._original, original], how="vertical"),
                nw.concat([self._released, released], how="vertical"),
            )
        self._original, self._released = _in_memory_sample(
            *candidates, Engine.POLARS, self._key, self._max_rows
        )

    def result(self) -> AlignedSample:
        """The rows kept, in the order they arrived, as pandas."""
        if self._original is None or self._released is None:
            raise ValueError("no chunk was added")
        return AlignedSample(
            original=self._original.drop(ROW_KEY_COLUMN).to_pandas(),
            released=self._released.drop(ROW_KEY_COLUMN).to_pandas(),
        )


def align_sample(
    original: WorkingFrame,
    released: WorkingFrame,
    engine: Engine,
    *,
    key: bytes,
    max_rows: int = MAX_SAMPLE_ROWS,
) -> AlignedSample:
    """Cut both tables to the same at most `max_rows` rows and bring them to pandas.

    Both frames carry `ROW_KEY_COLUMN`, as `Table.original_frame` and `Table.released_frame`
    give them. In-memory tables are in the same order, so rows are matched by position.
    Lazy tables are matched by joining on the row key, which must then be unique.
    """
    if engine.is_lazy:
        original_rows, released_rows = _lazy_sample(original, released, engine, key, max_rows)
    else:
        original_rows, released_rows = _in_memory_sample(original, released, engine, key, max_rows)
    return AlignedSample(
        original=original_rows.drop(ROW_KEY_COLUMN).to_pandas(),
        released=released_rows.drop(ROW_KEY_COLUMN).to_pandas(),
    )


def _in_memory_sample(
    original: WorkingFrame, released: WorkingFrame, engine: Engine, key: bytes, max_rows: int
) -> tuple[nw.DataFrame[Any], nw.DataFrame[Any]]:
    original_rows = collect(original)
    released_rows = collect(released)
    if len(original_rows) <= max_rows:
        return original_rows, released_rows
    draws = collect(_draws(original_rows.select(ROW_KEY_COLUMN), engine, key))
    smallest = np.argpartition(draws.get_column(_DRAW).to_numpy(), max_rows - 1)[:max_rows]
    chosen = np.sort(smallest).tolist()
    return original_rows[chosen], released_rows[chosen]


def _lazy_sample(
    original: WorkingFrame, released: WorkingFrame, engine: Engine, key: bytes, max_rows: int
) -> tuple[nw.DataFrame[Any], nw.DataFrame[Any]]:
    original_lazy = original.lazy()
    keys = original_lazy.select(ROW_KEY_COLUMN)
    counts = first_row(
        keys.select(nw.len().alias("rows"), nw.col(ROW_KEY_COLUMN).n_unique().alias("distinct"))
    )
    if counts["distinct"] != counts["rows"]:
        raise ValueError("the row key must be unique to compare a lazy table with its original")
    drawn = _draws(keys, engine, key).lazy()
    chosen = drawn.sort(_DRAW).head(max_rows).select(ROW_KEY_COLUMN)
    original_rows = original_lazy.join(chosen, on=ROW_KEY_COLUMN, how="inner")
    released_rows = released.lazy().join(chosen, on=ROW_KEY_COLUMN, how="inner")
    return (
        original_rows.collect().sort(ROW_KEY_COLUMN),
        released_rows.collect().sort(ROW_KEY_COLUMN),
    )


def _draws(keys: WorkingFrame, engine: Engine, key: bytes) -> WorkingFrame:
    drawn = add_keyed_uniforms(keys, engine, key=key, source=ROW_KEY_COLUMN, names=[_DRAW])
    return drawn.select(ROW_KEY_COLUMN, _DRAW)
