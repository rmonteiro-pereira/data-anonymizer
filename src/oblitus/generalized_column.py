"""A quasi-identifier column as Mondrian sees it, and how a group of its values is written.

Mondrian cuts rows on the column that varies most and then replaces each cell with a
description of its whole group. This module holds what that needs per column:
- an order for the values, so the group can be cut at its median;
- how wide a group is on this column, relative to the whole table;
- the text or number that stands in for a group's values.

Numbers and dates are described by their range, written `lo-hi` for numbers and `lo to hi`
for dates, or by the range's midpoint. Categories are described by the set of their values,
or by their closest common ancestor when the plan gives a hierarchy. Missing cells take the
description of their group, so that every row of a group holds the same value.
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
    ) -> None:
        self.name = name
        self.shape = shape_of(dtype)
        self._dtype = dtype
        self._values = list(values)
        self._hierarchy = hierarchy
        self._style = style
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

    def describe(self, members: np.ndarray) -> Any:
        """The value that replaces every cell of the members, or None when all are missing."""
        present = members[self._present[members]]
        if present.size == 0:
            return None
        if self.shape is ColumnShape.CATEGORY:
            return self._describe_categories([self._values[position] for position in present])
        low = self._values[present[np.argmin(self.order[present])]]
        high = self._values[present[np.argmax(self.order[present])]]
        if self._style is RangeStyle.MIDPOINT:
            return _midpoint(low, high)
        return _range_text(low, high)

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
        distinct = sorted({str(value) for value in values})
        if len(distinct) == 1:
            return distinct[0]
        return "{" + ", ".join(distinct) + "}"


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


def _midpoint(low: Any, high: Any) -> Any:
    if isinstance(low, datetime):
        return low + (high - low) / 2
    if isinstance(low, date):
        return date.fromordinal((low.toordinal() + high.toordinal()) // 2)
    return (float(low) + float(high)) / 2


def _range_text(low: Any, high: Any) -> str:
    if isinstance(low, (date, datetime)):
        low_text = _date_text(low)
        high_text = _date_text(high)
        return low_text if low_text == high_text else f"{low_text} to {high_text}"
    low_text = _number_text(low)
    high_text = _number_text(high)
    return low_text if low_text == high_text else f"{low_text}-{high_text}"


def _date_text(value: date | datetime) -> str:
    if isinstance(value, datetime) and value.time() != time.min:
        return value.isoformat(sep=" ")
    return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()


def _number_text(value: Any) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"
