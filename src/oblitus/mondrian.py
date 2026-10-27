"""Mondrian generalization: k-anonymity by cutting the rows into groups of at least k.

Local suppression keeps the cells it can and blanks the rest. Mondrian (LeFevre, DeWitt and
Ramakrishnan, 2006) keeps every row and makes every quasi-identifier coarser instead:
1. Start with all rows as one group.
2. Pick the quasi-identifier whose values spread the most across the group, relative to the
   whole table, and cut the group at its median. Rows that share a value stay together
   (the strict variant), so a cut never splits equal values.
3. Cut each half the same way while both halves would keep at least k rows.
4. Replace each quasi-identifier cell with the description of its group: the range of its
   numbers or dates, or the set (or closest ancestor) of its categories.

Cutting works on the distinct quasi-identifier combinations and how many rows each has, so
the work does not grow with duplicated rows, and the result is the same on every engine.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np

from oblitus.engines import WorkingFrame, to_native, to_working_frame
from oblitus.generalization_hierarchies import GeneralizationHierarchy
from oblitus.generalized_column import GeneralizedColumn
from oblitus.group_rewrite import distinct_rows, overwrite_by_key
from oblitus.plan import RangeStyle


@dataclass(frozen=True, slots=True)
class MondrianOutcome:
    """The table after generalization, in the form it was given, and what changed."""

    data: Any
    groups: int
    changed_columns: list[str]


def enforce_k_anonymity_by_mondrian(
    frame: Any,
    quasi_identifiers: Sequence[str],
    k: int,
    *,
    hierarchies: Mapping[str, GeneralizationHierarchy] | None = None,
    style: RangeStyle = RangeStyle.RANGE,
) -> MondrianOutcome:
    """Generalize quasi-identifiers so every combination is shared by at least k rows.

    Takes a pandas, Polars or DuckDB table, or a Narwhals frame, and returns the same kind.
    `hierarchies` maps a category column to the hierarchy that describes a group of its
    values. A table with fewer than k rows becomes one group, because no cut can help.
    """
    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return _generalize(frame, quasi_identifiers, k, hierarchies or {}, style)
    working, engine = to_working_frame(frame)
    outcome = _generalize(working, quasi_identifiers, k, hierarchies or {}, style)
    return MondrianOutcome(to_native(outcome.data, engine), outcome.groups, outcome.changed_columns)


def _generalize(
    frame: WorkingFrame,
    quasi_identifiers: Sequence[str],
    k: int,
    hierarchies: Mapping[str, GeneralizationHierarchy],
    style: RangeStyle,
) -> MondrianOutcome:
    names = frame.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    if not columns or k <= 1:
        return MondrianOutcome(frame, 0, [])

    combinations = distinct_rows(frame, columns)
    dimensions = [
        GeneralizedColumn(
            name,
            combinations.schema[name],
            combinations.get_column(name).to_list(),
            hierarchy=hierarchies.get(name),
            style=style,
        )
        for name in columns
    ]
    weights = combinations.get_column("count").to_numpy()
    groups = partition_rows(dimensions, weights, k)
    keys = list(zip(*[combinations.get_column(name).to_list() for name in columns], strict=True))
    replacements = {
        dimension.name: (_group_descriptions(dimension, groups, len(keys)), dimension.output_dtype)
        for dimension in dimensions
    }
    rewritten = overwrite_by_key(frame, columns, keys, replacements)
    return MondrianOutcome(rewritten, len(groups), columns)


def _group_descriptions(
    dimension: GeneralizedColumn, groups: list[np.ndarray], combinations: int
) -> list[Any]:
    """For each combination, the description of the group it landed in."""
    descriptions: list[Any] = [None] * combinations
    for members in groups:
        description = dimension.describe(members)
        for position in members:
            descriptions[position] = description
    return descriptions


def partition_rows(
    dimensions: Sequence[GeneralizedColumn], weights: np.ndarray, k: int
) -> list[np.ndarray]:
    """Cut the combinations into groups holding at least k rows each, by median cuts.

    `weights` is the row count of each combination. Returns the positions of the
    combinations of every group. A combination is never split between groups.
    """
    pending = [np.arange(len(weights))]
    finished: list[np.ndarray] = []
    while pending:
        members = pending.pop()
        halves = _cut(dimensions, weights, members, k)
        if halves is None:
            finished.append(members)
        else:
            pending.extend(halves)
    return finished


def _cut(
    dimensions: Sequence[GeneralizedColumn], weights: np.ndarray, members: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a group in two at the median of its widest column that allows it, or None."""
    if weights[members].sum() < 2 * k:
        return None
    by_width = sorted(dimensions, key=lambda dimension: -dimension.relative_width(members))
    for dimension in by_width:
        if dimension.relative_width(members) == 0:
            break
        halves = _median_cut(dimension.order, weights, members, k)
        if halves is not None:
            return halves
    return None


def _median_cut(
    order: np.ndarray, weights: np.ndarray, members: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    ranked = members[np.argsort(order[members], kind="stable")]
    values = order[ranked]
    cumulative = np.cumsum(weights[ranked])
    total = cumulative[-1]
    # A cut falls only where the value changes, so equal values stay on one side.
    ends_a_value = np.append(values[1:] != values[:-1], False)
    allowed = ends_a_value & (cumulative >= k) & (total - cumulative >= k)
    if not allowed.any():
        return None
    candidates = np.flatnonzero(allowed)
    best = candidates[np.argmin(np.abs(cumulative[candidates] - total / 2))]
    return ranked[: best + 1], ranked[best + 1 :]
