"""Techniques that hide a column's level and units while keeping its pattern.

A store's revenue in reais says how big the store is; the same revenue recentred
and rescaled still shows its seasonality, its spikes and how it compares with
others, without the number a competitor would want. The default methods are
affine (shift and scale), so the shape of the distribution is untouched. Methods
that change the shape need `allow_shape_change`.

Limit of affine methods: anyone who knows the true values of two records can solve
for the shift and scale and convert every value back. Each column gets its own
shift and scale, so knowing one column does not unlock another.
"""

import narwhals as nw

from data_anonymizer.table import Table
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.numeric_expressions import as_number, per_group, round_half_away

SHAPE_KEEPING_METHODS = ("robust", "zscore", "minmax")
_SHAPE_CHANGING_METHODS = ("index", "rank", "log_ratio", "growth")
_INDEX_DECIMALS = 6


def normalize(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    method: str = "robust",
    by: str | None = None,
    order_by: str | None = None,
    allow_shape_change: bool = False,
) -> nw.Expr:
    """Recentre and rescale a numeric column so it keeps its pattern but loses its units.

    - `robust`: (value − median) / interquartile range. The default, because a single
      dominant record barely moves the median and the quartiles.
    - `zscore`: (value − mean) / standard deviation.
    - `minmax`: rescaled to the range 0 to 1.
    - `index`: value / mean × 100. With `by`, each group gets its own mean, so every
      store averages 100 and its size disappears entirely.
    - `rank`: percentile rank between 0 and 1.
    - `log_ratio`: log(value / geometric mean); values must be positive.
    - `growth`: change from the previous value, as a fraction, ordered by `order_by`.

    `by` computes the method within each group of another column. Grouping and the
    last four methods change the distribution's shape, so they need
    `allow_shape_change`.
    """
    _check_method(method, by, allow_shape_change)
    group = None if by is None else table.original_column(by)
    values = as_number(column)
    if method == "growth":
        return _growth(table, values, group, order_by)
    _check_values(table, values, method, group)
    return _rescale(values, method, group)


def _check_method(method: str, by: str | None, allow_shape_change: bool) -> None:
    known = SHAPE_KEEPING_METHODS + _SHAPE_CHANGING_METHODS
    if method not in known:
        raise ValueError(f"method must be one of {', '.join(known)}")
    changes_shape = method in _SHAPE_CHANGING_METHODS or by is not None
    if changes_shape and not allow_shape_change:
        reason = f"grouping by {by!r}" if by is not None else f"method {method!r}"
        raise ValueError(
            f"{reason} changes the column's distribution; "
            "set allow_shape_change to true to accept that"
        )


def _check_values(table: Table, values: nw.Expr, method: str, group: str | None) -> None:
    if method == "log_ratio" and table.statistics(lowest=values.min())["lowest"] <= 0:
        raise ValueError("log_ratio needs positive values")
    if method == "index":
        # SQL engines refuse a window inside an aggregate, so the flag is stored first.
        zero_mean = (per_group(values.mean(), group) == 0) & ~values.is_null()
        flag = nw.col(table.add(zero_mean.cast(nw.Int64)))
        if table.statistics(zero=flag.sum())["zero"]:
            raise ValueError("index needs a non-zero mean in every group")


def _rescale(values: nw.Expr, method: str, group: str | None) -> nw.Expr:
    def within(aggregate: nw.Expr) -> nw.Expr:
        return per_group(aggregate, group)

    if method == "robust":
        spread = within(values.quantile(0.75, "linear") - values.quantile(0.25, "linear"))
        median = within(values.quantile(0.5, "linear"))
        return (values - median) / _nonzero(spread, within(values.std(ddof=0)))
    if method == "zscore":
        return (values - within(values.mean())) / _nonzero(within(values.std(ddof=0)))
    if method == "minmax":
        lowest = within(values.min())
        return (values - lowest) / _nonzero(within(values.max()) - lowest)
    if method == "index":
        # Float division turns a flat series into 100 and 100.00000000000001; the last digits
        # carry no meaning, and they would make equal values look different.
        return round_half_away(values / within(values.mean()) * 100, _INDEX_DECIMALS)
    if method == "rank":
        return within(values.rank("average")) / within(values.count())
    return values.log() - within(values.log().mean())


def _nonzero(spread: nw.Expr, fallback: nw.Expr | None = None) -> nw.Expr:
    # A constant group has no spread. Dividing by the fallback spread, or by 1, keeps
    # it at zero instead of failing.
    one = nw.lit(1.0)
    if fallback is not None:
        one = nw.when(fallback == 0).then(one).otherwise(fallback)
    return nw.when(spread == 0).then(one).otherwise(spread)


def _growth(table: Table, values: nw.Expr, group: str | None, order_by: str | None) -> nw.Expr:
    if order_by is None:
        raise ValueError("growth needs order_by, the column that orders the periods")
    order = table.original_column(order_by)
    partition = [] if group is None else [group]
    previous = values.shift(1).over(*partition, order_by=order)
    return values / previous - 1
