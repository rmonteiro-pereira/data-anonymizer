"""Multivariate microaggregation (MDAV): replace several numeric columns together by group means.

The `microaggregate` technique sorts one column and averages neighbours, so each column is
protected alone. A record's vector over several columns can still be rare. This step groups
records by their distance over all the chosen columns at once, with the Maximum Distance to
Average Vector method (Domingo-Ferrer and Torra, 2005), and gives every record of a group the
group's mean in each column:
1. Standardize the columns, so no column dominates the distance.
2. While at least 3k records remain, take the record farthest from the centroid of the rest,
   and form a group of it and its k - 1 nearest records. Do the same around the record
   farthest from that one.
3. With 2k to 3k - 1 records left, form one group around the record farthest from the
   centroid, and the rest is the last group. With fewer than 2k left, they are one group.

Every group holds at least k records, so every released vector is shared by k records. Each
column keeps its mean. Records with a missing value in any chosen column are left as they
are, since they have no position in the space. The work is quadratic in the number of rows.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

from oblitus.engines import WorkingFrame, collect, to_native, to_working_frame
from oblitus.group_rewrite import overwrite_by_key

_ROW_POSITION = "__da_mdav_row"


@dataclass(frozen=True, slots=True)
class JointMicroaggregationOutcome:
    """The table after microaggregation, in the form it was given, and how many groups formed."""

    data: Any
    groups: int


def microaggregate_jointly(
    frame: Any, columns: Sequence[str], group_size: int, *, row_key: str | None = None
) -> JointMicroaggregationOutcome:
    """Replace the numeric `columns` of every complete record by the mean of its MDAV group.

    Takes a pandas, Polars or DuckDB table, or a Narwhals frame, and returns the same kind.
    `row_key` names a column with a different value in every row. A DuckDB table needs it,
    because its rows have no fixed order. Pandas and Polars tables use the row position.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if len(set(columns)) < 2:
        raise ValueError("joint microaggregation needs at least two different columns")
    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return _microaggregate(frame, list(columns), group_size, row_key)
    working, engine = to_working_frame(frame)
    outcome = _microaggregate(working, list(columns), group_size, row_key)
    return JointMicroaggregationOutcome(to_native(outcome.data, engine), outcome.groups)


def mdav_groups(points: np.ndarray, group_size: int) -> list[np.ndarray]:
    """Split the rows of `points` into groups of at least `group_size` rows, nearest together."""
    remaining = np.arange(len(points))
    groups: list[np.ndarray] = []
    while len(remaining) >= 3 * group_size:
        first_seed = _farthest(points, remaining, points[remaining].mean(axis=0))
        first = _nearest_to(points, remaining, points[first_seed], group_size)
        remaining = np.setdiff1d(remaining, first)
        # The second group forms at the opposite end, around the record farthest from the first.
        second_seed = _farthest(points, remaining, points[first_seed])
        second = _nearest_to(points, remaining, points[second_seed], group_size)
        remaining = np.setdiff1d(remaining, second)
        groups.extend([first, second])
    if len(remaining) >= 2 * group_size:
        seed = _farthest(points, remaining, points[remaining].mean(axis=0))
        group = _nearest_to(points, remaining, points[seed], group_size)
        remaining = np.setdiff1d(remaining, group)
        groups.append(group)
    if len(remaining):
        groups.append(remaining)
    return groups


def _farthest(points: np.ndarray, candidates: np.ndarray, target: np.ndarray) -> int:
    return int(candidates[np.argmax(_distances(points[candidates], target))])


def _nearest_to(
    points: np.ndarray, remaining: np.ndarray, target: np.ndarray, count: int
) -> np.ndarray:
    nearest = np.argsort(_distances(points[remaining], target), kind="stable")[:count]
    return remaining[nearest]


def _distances(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points - target, axis=1)


def _microaggregate(
    frame: WorkingFrame, columns: list[str], group_size: int, row_key: str | None
) -> JointMicroaggregationOutcome:
    schema = frame.collect_schema()
    for column in columns:
        if column not in schema.names():
            raise ValueError(f"column {column!r} is not in the table")
        if not schema[column].is_numeric():
            raise ValueError(f"column {column!r} is not numeric")
    key, numbered = _row_key_of(frame, row_key)
    rows = collect(numbered.select(key, *columns))
    keys = rows.get_column(key).to_list()
    if len(set(keys)) != len(keys):
        raise ValueError(f"row key {key!r} repeats; every row needs its own key")
    values = rows.select(*[nw.col(column).cast(nw.Float64) for column in columns]).to_numpy()
    complete = np.flatnonzero(~np.isnan(values).any(axis=1))
    groups = [
        complete[members] for members in mdav_groups(_standardized(values[complete]), group_size)
    ]
    averaged = values.copy()
    for members in groups:
        averaged[members] = values[members].mean(axis=0)

    replacements = {
        column: (_column_values(averaged[:, position], schema[column]), schema[column])
        for position, column in enumerate(columns)
    }
    rewritten = overwrite_by_key(numbered, [key], [(item,) for item in keys], replacements)
    if row_key is None:
        rewritten = rewritten.drop(key)
    return JointMicroaggregationOutcome(rewritten, len(groups))


def _row_key_of(frame: WorkingFrame, row_key: str | None) -> tuple[str, WorkingFrame]:
    if row_key is not None:
        return row_key, frame
    if isinstance(frame, nw.LazyFrame):
        raise ValueError("a DuckDB table needs a row_key: name a column that identifies each row")
    return _ROW_POSITION, frame.with_row_index(_ROW_POSITION)


def _standardized(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    spread = values.std(axis=0)
    spread[spread == 0] = 1.0
    return (values - values.mean(axis=0)) / spread


def _column_values(values: np.ndarray, dtype: Any) -> list[Any]:
    """Python values for a column of group means, whole numbers again for integer columns."""
    if dtype.is_integer():
        return [None if np.isnan(item) else _round_half_away(item) for item in values]
    return [None if np.isnan(item) else float(item) for item in values]


def _round_half_away(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))
