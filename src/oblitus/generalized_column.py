"""A quasi-identifier column as Mondrian sees it, and how a group of its values is written.

Mondrian cuts rows on the column that varies most and then replaces each cell with a
description of its whole group. This module holds what that needs per column:
- an order for the values, so the group can be cut at its median;
- how wide a group is on this column, relative to the whole table;
- the text or number that stands in for a group's values.

Numbers and dates are described by the cuts that bound the group, never by the smallest and
largest value inside it, because one extreme record would otherwise show in the release. A
group between two cuts reads `20 to under 40`, and the outermost groups are open-ended: `under
20` and `90 or more`. The midpoint style publishes the group's median instead. A group that no
cut bounds on this column reads `any value`, unless all the table's values are equal. Categories are
described by the set of their values, or by their closest common ancestor when the plan gives a
hierarchy. A category held by fewer than k rows in the whole table is never named in a set: it
reads `other`, so the set does not show that its few holders are in the table. Missing cells take the description of their group, so that every row of a group
holds the same value.
"""

import math
from collections.abc import Sequence
from datetime import date, datetime, time
from enum import StrEnum
from typing import Any

import narwhals as nw
import numpy as np

from oblitus.generalization_hierarchies import GeneralizationHierarchy
from oblitus.plan import RangeStyle

_EPOCH = datetime(1970, 1, 1)
OPEN = (-math.inf, math.inf)
RARE_CATEGORY_LABEL = "other"


class ColumnShape(StrEnum):
    """How the values of a column are ordered and described."""

    NUMBER = "number"
    DATE = "date"
    CATEGORY = "category"


def shape_of(dtype: Any) -> ColumnShape:
    """The shape a Narwhals type is generalized as; booleans count as categories."""
    if dtype.is_temporal():
        return ColumnShape.DATE
    if dtype.is_numeric():
        return ColumnShape.NUMBER
    return ColumnShape.CATEGORY


class GeneralizedColumn:
    """The distinct values of one quasi-identifier column, ready to be cut and described."""

    def __init__(
        self,
        name: str,
        dtype: Any,
        values: Sequence[Any],
        *,
        hierarchy: GeneralizationHierarchy | None = None,
        style: RangeStyle = RangeStyle.RANGE,
        row_counts: Sequence[int] | None = None,
        rare_below: int = 0,
    ) -> None:
        self.name = name
        self.shape = shape_of(dtype)
        self._dtype = dtype
        self._values = list(values)
        self._hierarchy = hierarchy
        self._style = style
        self._rare_categories = _rare_categories(self._values, row_counts, rare_below)
        self.order = self._encode()
        self._present = np.isfinite(self.order)
        self._extent = self._whole_table_extent()

    @property
    def output_dtype(self) -> Any:
        """The Narwhals type of the column after generalization."""
        if self._style is RangeStyle.MIDPOINT and self.shape is ColumnShape.NUMBER:
            return nw.Float64
        if self._style is RangeStyle.MIDPOINT and self.shape is ColumnShape.DATE:
            return self._dtype
        return nw.String

    def relative_width(self, members: np.ndarray) -> float:
        """How much of the column's whole spread the members cover, from 0 to 1."""
        present = members[self._present[members]]
        if present.size == 0 or self._extent == 0:
            return 0.0
        if self.shape is ColumnShape.CATEGORY:
            return len(set(self.order[present].tolist())) / self._extent
        return float(self.order[present].max() - self.order[present].min()) / self._extent

    def describe(
        self, members: np.ndarray, weights: np.ndarray, bounds: tuple[float, float] = OPEN
    ) -> Any:
        """The value that replaces every cell of the members, or None when all are missing.

        `weights` holds the row count of every combination. `bounds` are the cuts that
        enclose the group on this column, as positions in the column's order: the group
        holds values from the low bound up to, not including, the high bound.
        """
        present = members[self._present[members]]
        if present.size == 0:
            return None
        if self.shape is ColumnShape.CATEGORY:
            return self._describe_categories([self._values[position] for position in present])
        if self._style is RangeStyle.MIDPOINT:
            return self._values[self._median_position(present, weights)]
        if bounds == OPEN and np.unique(self.order[present]).size == 1:
            return self._bound_text(float(self.order[present[0]]))
        return self._range_text(bounds)

    def _median_position(self, present: np.ndarray, weights: np.ndarray) -> int:
        """The combination holding the group's median row, so no single record moves it."""
        ranked = present[np.argsort(self.order[present], kind="stable")]
        cumulative = np.cumsum(weights[ranked])
        return int(ranked[np.searchsorted(cumulative, cumulative[-1] / 2)])

    def _range_text(self, bounds: tuple[float, float]) -> str:
        low, high = bounds
        is_date = self.shape is ColumnShape.DATE
        below, from_here = ("before", "or later") if is_date else ("under", "or more")
        if math.isinf(low) and math.isinf(high):
            return "any value"
        if math.isinf(low):
            return f"{below} {self._bound_text(high)}"
        if math.isinf(high):
            return f"{self._bound_text(low)} {from_here}"
        return f"{self._bound_text(low)} to {below} {self._bound_text(high)}"

    def _bound_text(self, position: float) -> str:
        value = self._values[int(np.flatnonzero(self.order == position)[0])]
        if self.shape is ColumnShape.DATE:
            return _date_text(value)
        return _number_text(value)

    def _encode(self) -> np.ndarray:
        if self.shape is ColumnShape.NUMBER:
            return np.array([_number_position(value) for value in self._values])
        if self.shape is ColumnShape.DATE:
            return np.array([_date_position(value) for value in self._values])
        return self._category_ranks()

    def _category_ranks(self) -> np.ndarray:
        present = {str(value) for value in self._values if value is not None}
        if self._hierarchy is None:
            ordered = sorted(present)
        else:
            ordered = sorted(present, key=self._hierarchy.sort_key)
        rank = {text: position for position, text in enumerate(ordered)}
        return np.array([_rank_or_missing(value, rank) for value in self._values])

    def _whole_table_extent(self) -> float:
        present = self.order[self._present]
        if present.size == 0:
            return 0.0
        if self.shape is ColumnShape.CATEGORY:
            return float(len(set(present.tolist())))
        return float(present.max() - present.min())

    def _describe_categories(self, values: list[Any]) -> str:
        if self._hierarchy is not None:
            level = self._hierarchy.lowest_common_level(values)
            return self._hierarchy.ancestor(values[0], level)
        names = {str(value) for value in values}
        distinct = sorted(names - self._rare_categories)
        if names & self._rare_categories:
            distinct.append(RARE_CATEGORY_LABEL)
        if len(distinct) == 1:
            return distinct[0]
        return "{" + ", ".join(distinct) + "}"


def _rare_categories(
    values: Sequence[Any], row_counts: Sequence[int] | None, rare_below: int
) -> set[str]:
    """Categories held by fewer than `rare_below` rows in the whole table.

    A set that names such a category would show that its few holders are in the table.
    """
    if row_counts is None or rare_below <= 1:
        return set()
    totals: dict[str, int] = {}
    for value, count in zip(values, row_counts, strict=True):
        if value is not None:
            totals[str(value)] = totals.get(str(value), 0) + int(count)
    return {name for name, total in totals.items() if total < rare_below}


def _rank_or_missing(value: Any, rank: dict[str, int]) -> float:
    if value is None:
        return math.inf
    return float(rank[str(value)])


def _number_position(value: Any) -> float:
    if value is None or math.isnan(value):
        return math.inf
    return float(value)


def _date_position(value: date | datetime | None) -> float:
    if value is None:
        return math.inf
    if isinstance(value, datetime):
        return (value.replace(tzinfo=None) - _EPOCH).total_seconds()
    return float(value.toordinal())


def _date_text(value: date | datetime) -> str:
    if isinstance(value, datetime) and value.time() != time.min:
        return value.isoformat(sep=" ")
    return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()


def _number_text(value: Any) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"
