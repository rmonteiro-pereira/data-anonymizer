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

import numpy as np
import pandas as pd

from data_anonymizer.techniques.context import TechniqueContext

_SHAPE_KEEPING_METHODS = ("robust", "zscore", "minmax")
_SHAPE_CHANGING_METHODS = ("index", "rank", "log_ratio", "growth")


def normalize(
    values: pd.Series,
    context: TechniqueContext,
    *,
    method: str = "robust",
    by: str | None = None,
    order_by: str | None = None,
    allow_shape_change: bool = False,
) -> pd.Series:
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
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    if by is None:
        groups = pd.Series(0, index=values.index)
    else:
        groups = context.group_keys(by).reindex(values.index)

    if method == "growth":
        return _growth(numbers, groups, context, order_by)
    grouped = numbers.groupby(groups, dropna=False, sort=False)
    return grouped.transform(lambda group: _rescale(group, method))


def _check_method(method: str, by: str | None, allow_shape_change: bool) -> None:
    known = _SHAPE_KEEPING_METHODS + _SHAPE_CHANGING_METHODS
    if method not in known:
        raise ValueError(f"method must be one of {', '.join(known)}")
    changes_shape = method in _SHAPE_CHANGING_METHODS or by is not None
    if changes_shape and not allow_shape_change:
        reason = f"grouping by {by!r}" if by is not None else f"method {method!r}"
        raise ValueError(
            f"{reason} changes the column's distribution; "
            "set allow_shape_change to true to accept that"
        )


def _rescale(group: pd.Series, method: str) -> pd.Series:
    present = group.dropna()
    if present.empty:
        return group
    if method == "robust":
        spread = float(present.quantile(0.75) - present.quantile(0.25))
        return (group - present.median()) / (spread or _fallback_spread(present))
    if method == "zscore":
        return (group - present.mean()) / _fallback_spread(present)
    if method == "minmax":
        span = float(present.max() - present.min()) or 1.0
        return (group - present.min()) / span
    if method == "index":
        mean = float(present.mean())
        if mean == 0:
            raise ValueError("index needs a non-zero mean in every group")
        return group / mean * 100
    if method == "rank":
        return group.rank(pct=True)
    if (present <= 0).any():
        raise ValueError("log_ratio needs positive values")
    return np.log(group) - float(np.log(present).mean())


def _fallback_spread(present: pd.Series) -> float:
    # A constant group has no spread; dividing by 1 keeps it at zero instead of failing.
    return float(present.std(ddof=0)) or 1.0


def _growth(
    numbers: pd.Series,
    groups: pd.Series,
    context: TechniqueContext,
    order_by: str | None,
) -> pd.Series:
    if order_by is None:
        raise ValueError("growth needs order_by, the column that orders the periods")
    order = context.group_keys(order_by).reindex(numbers.index)
    frame = pd.DataFrame({"value": numbers, "group": groups, "order": order})
    ordered = frame.sort_values(["group", "order"], kind="stable")
    change = ordered.groupby("group", dropna=False, sort=False)["value"].pct_change()
    return change.reindex(numbers.index)
