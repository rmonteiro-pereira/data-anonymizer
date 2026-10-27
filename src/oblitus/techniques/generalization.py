"""Techniques that make values coarser so more records share each value."""

import math
from typing import Any

import narwhals as nw
import numpy as np

from oblitus.generalization_hierarchies import GeneralizationHierarchy
from oblitus.table import Table
from oblitus.techniques.context import TechniqueContext
from oblitus.techniques.date_expressions import (
    as_datetime_expression,
    shift_days,
    without_time_zone,
)
from oblitus.techniques.numeric_expressions import (
    as_number,
    in_original_type,
    round_half_away,
)
from oblitus.techniques.value_mapping import lookup_distinct

_TRUNCATION_UNITS = {"year": "1y", "month": "1mo", "day": "1d"}
_ALL_UNITS = (*_TRUNCATION_UNITS, "week")
_DAYS_TO_MONDAY = range(-6, 1)


def bin_numeric(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    bins: int = 5,
    width: float | None = None,
    strategy: str = "quantile",
) -> nw.Expr:
    """Replace numbers with interval labels such as `[20, 30)`.

    With `width`, intervals have that fixed width (ages in decades). Otherwise the
    column is cut into `bins` intervals of equal size (`equal_width`) or with an
    equal number of rows each (`quantile`).
    """
    if width is None and strategy not in ("quantile", "equal_width"):
        raise ValueError("strategy must be 'quantile' or 'equal_width'")
    values = as_number(column)
    edges = _bin_edges(table, values, bins, width, strategy)
    if edges is None:
        return nw.col(column).cast(nw.String)
    if len(edges) < 2:
        # Every value is the same number, so a single closed interval holds them.
        label = f"[{_format_number(edges[0])}, {_format_number(edges[0])}]"
        return nw.when(~values.is_null()).then(nw.lit(label))

    # Fixed-width bins are all right-open; data-driven bins close the last one on the maximum.
    labels = _interval_labels(edges, close_last=width is None)
    return _label_by_interval(values, [float(edge) for edge in edges[1:-1]], labels)


def _bin_edges(
    table: Table, values: nw.Expr, bins: int, width: float | None, strategy: str
) -> np.ndarray | None:
    """The interval edges, or None when the column has no numbers."""
    aggregations = {"present": values.count(), "low": values.min(), "high": values.max()}
    use_quantiles = width is None and strategy == "quantile"
    fractions = np.linspace(0, 1, bins + 1)[1:-1]
    if use_quantiles:
        for position, fraction in enumerate(fractions):
            aggregations[f"quantile_{position}"] = values.quantile(float(fraction), "linear")
    found = table.statistics(**aggregations)
    if found["present"] == 0:
        return None

    low, high = found["low"], found["high"]
    if width is not None:
        upper = (high // width + 1) * width
        # Half a width past the top edge so floating-point steps never drop it.
        return np.arange(low // width * width, upper + width / 2, width)
    if use_quantiles:
        inner = [found[f"quantile_{position}"] for position in range(len(fractions))]
        return np.unique(np.array([low, *inner, high]))
    return np.linspace(low, high, bins + 1)


def _label_by_interval(values: nw.Expr, inner_edges: list[float], labels: list[str]) -> nw.Expr:
    """Pick a label by the first upper edge the value lies below; the last label takes the rest."""
    return nw.when(~values.is_null()).then(_search_labels(values, inner_edges, labels))


def _search_labels(values: nw.Expr, inner_edges: list[float], labels: list[str]) -> nw.Expr:
    # `inner_edges[i]` is the upper edge of `labels[i]`. A binary search keeps the
    # expression shallow, because fixed-width bins can number in the thousands.
    if len(labels) == 1:
        return nw.lit(labels[0])
    middle = len(labels) // 2
    below = _search_labels(values, inner_edges[: middle - 1], labels[:middle])
    above = _search_labels(values, inner_edges[middle:], labels[middle:])
    return nw.when(values < inner_edges[middle - 1]).then(below).otherwise(above)


def _interval_labels(edges: np.ndarray, *, close_last: bool) -> list[str]:
    labels = []
    for index in range(len(edges) - 1):
        is_last = index == len(edges) - 2
        closing = "]" if close_last and is_last else ")"
        labels.append(
            f"[{_format_number(edges[index])}, {_format_number(edges[index + 1])}{closing}"
        )
    return labels


def _format_number(number: float) -> str:
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.4g}"


def round_numeric(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    decimals: int = 0,
    base: float | None = None,
) -> nw.Expr:
    """Round to a number of decimals, or to the nearest multiple of `base` (such as 1000).

    Halves round away from zero. Negative `decimals` round to tens, hundreds and so on.
    """
    values = as_number(column)
    if base is None and decimals < 0:
        # Engines only round to a non-negative number of decimals.
        base = 10**-decimals
    if base is None:
        return in_original_type(round_half_away(values, decimals), table.dtype(column))
    rounded = round_half_away(values / base) * base
    if float(base).is_integer():
        return in_original_type(rounded, table.dtype(column))
    return rounded


def truncate_date(
    table: Table, column: str, context: TechniqueContext, *, unit: str = "year"
) -> nw.Expr:
    """Cut dates down to the start of their year, month, week or day.

    Weeks start on Monday. Time zones are dropped, keeping the wall-clock date.
    """
    if unit not in _ALL_UNITS:
        raise ValueError(f"unit must be one of {', '.join(_ALL_UNITS)}")
    dates = as_datetime_expression(table, column)
    dates = without_time_zone(dates, table.dtype(column))
    if unit == "week":
        return _start_of_week(dates)
    return dates.dt.truncate(_TRUNCATION_UNITS[unit])


def _start_of_week(dates: nw.Expr) -> nw.Expr:
    midnight = dates.dt.truncate("1d")
    # Weekdays count from 1 on Monday, so the way back to Monday is 1 minus the weekday.
    days_to_monday = 1 - midnight.dt.weekday().cast(nw.Int64)
    return shift_days(midnight, days_to_monday, _DAYS_TO_MONDAY)


def truncate_string(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    keep: int = 3,
    pad_char: str | None = None,
) -> nw.Expr:
    """Keep the first characters, such as the region digits of a postal code.

    With `pad_char`, the removed characters are replaced so the length stays the same.
    """
    if pad_char is None:
        return nw.col(column).cast(nw.String).str.slice(0, keep)

    def pad(text: str) -> str:
        return text[:keep] + pad_char * max(0, len(text) - keep)

    return lookup_distinct(table, column, pad)


def top_bottom_code(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    lower: float | None = None,
    upper: float | None = None,
) -> nw.Expr:
    """Clip extreme values, which single people out, to a lower and an upper limit."""
    values = as_number(column)
    limits = {}
    if lower is None:
        limits["lower"] = values.quantile(lower_quantile, "linear")
    if upper is None:
        limits["upper"] = values.quantile(upper_quantile, "linear")
    found = table.statistics(**limits) if limits else {}
    lower_limit = found.get("lower", lower)
    upper_limit = found.get("upper", upper)
    dtype = table.dtype(column)
    if dtype.is_integer():
        # Integers are clipped as integers: above 2**53 a float cannot hold them exactly.
        return nw.col(column).clip(
            lower_bound=_whole_limit(lower_limit), upper_bound=_whole_limit(upper_limit)
        )
    clipped = values.clip(lower_bound=lower_limit, upper_bound=upper_limit)
    return in_original_type(clipped, dtype)


def _whole_limit(limit: float | None) -> int | None:
    # Halves round away from zero, as everywhere else in the library.
    if limit is None:
        return None
    return int(math.copysign(math.floor(abs(limit) + 0.5), limit))


def suppress_rare(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    min_count: int = 5,
    replacement: str | None = None,
) -> nw.Expr:
    """Replace categories seen fewer than `min_count` times, which identify few people.

    A replacement is text, so a column of another type comes back as text.
    """
    values = nw.col(column)
    if replacement is not None and table.dtype(column) != nw.String:
        values = values.cast(nw.String)
    seen = nw.col(column).count().over(column)
    rare = ~nw.col(column).is_null() & (seen < min_count)
    if replacement is None:
        return nw.when(~rare).then(values)
    return nw.when(rare).then(nw.lit(replacement)).otherwise(values)


def generalize_hierarchy(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    hierarchy: dict[str, Any],
    level: int = 1,
) -> nw.Expr:
    """Replace each value with its stand-in at `level` of a hierarchy (postal code to region).

    `hierarchy` is either `{"levels": [{value: stand_in, ...}, ...]}`, one mapping per level
    from specific to general, or `{"prefix_lengths": [5, 3]}` to keep the first characters
    and star the rest. `{"preset": "cep"}` is the Brazilian postal code prefix hierarchy.
    Level 0 keeps the value, and the level past the last one gives `*`.
    """
    parsed = GeneralizationHierarchy.model_validate(hierarchy)
    if not 0 <= level <= parsed.top_level:
        raise ValueError(f"level must be between 0 and {parsed.top_level}")
    return lookup_distinct(table, column, lambda text: parsed.ancestor(text, level))
