"""Expression building blocks shared by the techniques that work on dates."""

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import narwhals as nw
import pandas as pd

from oblitus.detection.kinds import as_datetime
from oblitus.table import Table


def as_datetime_expression(table: Table, column: str) -> nw.Expr:
    """The column as date-times: dates widen to midnight, text is parsed.

    Text is parsed once per distinct value, with the same layouts the detector
    accepts. Text that is not a date becomes missing. Time zones of date-time
    columns are kept; parsed text carries none.
    """
    dtype = table.dtype(column)
    if dtype == nw.Date:
        return nw.col(column).cast(nw.Datetime("us"))
    if dtype == nw.Datetime:
        return nw.col(column)
    values = table.distinct_values(column)
    parsed = as_datetime(pd.Series(values, dtype=object))
    mapping = {value: _wall_clock(stamp) for value, stamp in zip(values, parsed, strict=True)}
    return table.lookup(column, mapping, nw.Datetime("us"))


def without_time_zone(dates: nw.Expr, dtype: Any) -> nw.Expr:
    """Drop the time zone but keep the wall-clock time, as a date-time column of `dtype` needs."""
    if dtype == nw.Datetime and dtype.time_zone is not None:
        return dates.dt.replace_time_zone(None)
    return dates


def shift_days(dates: nw.Expr, days: nw.Expr, possible_days: Iterable[int]) -> nw.Expr:
    """Move each date by its own whole number of days.

    `days` must be an integer expression whose values all lie in `possible_days`.
    Engines cannot build a duration from a column, so each possible shift is its
    own branch of a conditional.
    """
    return _shift_by_search(dates, days, sorted(possible_days))


def _shift_by_search(dates: nw.Expr, days: nw.Expr, steps: list[int]) -> nw.Expr:
    # A binary search keeps the expression shallow, so a large range of days stays
    # within what engines and Python can nest.
    if len(steps) == 1:
        return dates.dt.offset_by(f"{steps[0]}d")
    middle = len(steps) // 2
    below = _shift_by_search(dates, days, steps[:middle])
    above = _shift_by_search(dates, days, steps[middle:])
    return nw.when(days < steps[middle]).then(below).otherwise(above)


def _wall_clock(stamp: Any) -> datetime | None:
    if pd.isna(stamp):
        return None
    moment = pd.Timestamp(stamp)
    if moment.tzinfo is not None:
        moment = moment.tz_localize(None)
    return moment.to_pydatetime(warn=False)
