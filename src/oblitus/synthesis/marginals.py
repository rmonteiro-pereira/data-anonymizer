"""Inverse distribution functions: turn uniform numbers in (0, 1) into column values.

Every drawn column is generated as `quantile(u)` for some u in (0, 1). Independent
columns use independent u; correlated columns get u from a Gaussian copula; a
conditional override reuses a row's u with different settings, so the row keeps
its rank and correlations survive.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from oblitus.errors import SchemaError
from oblitus.synthesis.schema import ColumnSchema, Distribution
from oblitus.taxonomy import ColumnKind

Quantile = Callable[[np.ndarray], pd.Series]


def quantile_function(column: ColumnSchema) -> Quantile:
    """Return the inverse distribution function that generates this column."""
    if column.kind in (ColumnKind.INTEGER, ColumnKind.FLOAT):
        return _numeric_quantile(column)
    if column.kind is ColumnKind.BOOLEAN:
        true_share = 0.5 if column.true_share is None else column.true_share
        # High u means True, so a positive correlation pairs True with high values.
        return lambda uniform: pd.Series(uniform >= 1 - true_share)
    if column.kind is ColumnKind.DATETIME:
        return _date_quantile(column)
    if column.kind is ColumnKind.CATEGORY:
        return _category_quantile(column)
    raise SchemaError(f"{column.name}: text columns are not drawn from a distribution")


def _numeric_quantile(column: ColumnSchema) -> Quantile:
    if _distribution_of(column) is Distribution.QUANTILES:
        inverse = _piecewise_quantiles(column)
    else:
        inverse = _truncated(column, _scipy_distribution(column))

    def quantile(uniform: np.ndarray) -> pd.Series:
        values = inverse(uniform)
        if column.kind is ColumnKind.INTEGER:
            return pd.Series(np.round(values).astype("int64"))
        return pd.Series(values.astype(float))

    return quantile


def _distribution_of(column: ColumnSchema) -> Distribution:
    if column.distribution is not None:
        return column.distribution
    if column.quantiles is not None:
        return Distribution.QUANTILES
    if column.mean is not None and column.std is not None:
        return Distribution.NORMAL
    if column.minimum is not None and column.maximum is not None:
        return Distribution.UNIFORM
    raise SchemaError(
        f"{column.name}: give minimum and maximum, mean and std, quantiles, or a distribution"
    )


def _integer_margin(column: ColumnSchema) -> float:
    # Rounding gives each integer the half-unit on either side of it, so a bound on an
    # integer column reaches half a unit further out for its end values to be as likely.
    is_whole = column.kind is ColumnKind.INTEGER
    if is_whole and _distribution_of(column) is not Distribution.POISSON:
        return 0.5
    return 0.0


def _scipy_distribution(column: ColumnSchema):
    distribution = _distribution_of(column)
    mean = column.mean
    std = column.std
    if distribution is Distribution.UNIFORM:
        low, high = _required(column, "minimum", "maximum")
        margin = _integer_margin(column)
        return stats.uniform(loc=low - margin, scale=high - low + 2 * margin)
    if distribution is Distribution.EXPONENTIAL:
        (mean,) = _required(column, "mean")
        start = column.minimum or 0.0
        return stats.expon(loc=start, scale=mean - start)
    if distribution is Distribution.POISSON:
        (mean,) = _required(column, "mean")
        return stats.poisson(mean)
    mean, std = _required(column, "mean", "std")
    if distribution is Distribution.NORMAL:
        return stats.norm(mean, std)
    if mean <= 0:
        raise SchemaError(f"{column.name}: {distribution.value} needs a positive mean")
    if distribution is Distribution.LOGNORMAL:
        # Parameters chosen so the values themselves have the requested mean and std.
        sigma_squared = np.log(1 + (std / mean) ** 2)
        return stats.lognorm(
            s=np.sqrt(sigma_squared), scale=np.exp(np.log(mean) - sigma_squared / 2)
        )
    return stats.gamma(a=(mean / std) ** 2, scale=std**2 / mean)


def _required(column: ColumnSchema, *fields: str) -> list[float]:
    values = [getattr(column, field) for field in fields]
    missing = [field for field, value in zip(fields, values, strict=True) if value is None]
    if missing:
        distribution = _distribution_of(column).value
        raise SchemaError(f"{column.name}: {distribution} needs {', '.join(missing)}")
    return [float(value) for value in values]


def _truncated(column: ColumnSchema, distribution) -> Callable[[np.ndarray], np.ndarray]:
    # Squeezing u into [F(min), F(max)] truncates any shape without redrawing. A discrete
    # shape keeps its minimum, so the lower edge is the probability below it.
    margin = _integer_margin(column)
    lowest = None if column.minimum is None else column.minimum - margin
    highest = None if column.maximum is None else column.maximum + margin
    if lowest is not None and isinstance(distribution.dist, stats.rv_discrete):
        lowest -= 1
    low = 0.0 if lowest is None else float(distribution.cdf(lowest))
    high = 1.0 if highest is None else float(distribution.cdf(highest))
    if high <= low:
        raise SchemaError(f"{column.name}: the bounds leave no room for this distribution")
    return lambda uniform: distribution.ppf(low + uniform * (high - low))


def _piecewise_quantiles(column: ColumnSchema) -> Callable[[np.ndarray], np.ndarray]:
    points = dict(column.quantiles or {})
    margin = _integer_margin(column)
    if column.minimum is not None:
        points.setdefault(0.0, column.minimum - margin)
    if column.maximum is not None:
        points.setdefault(1.0, column.maximum + margin)
    if 0.0 not in points or 1.0 not in points:
        raise SchemaError(
            f"{column.name}: quantiles need the 0 and 1 points, or minimum and maximum"
        )
    probabilities = np.array(sorted(points))
    values = np.array([points[probability] for probability in probabilities])
    if np.any(np.diff(values) < 0):
        raise SchemaError(f"{column.name}: quantile values must not decrease")
    return lambda uniform: np.interp(uniform, probabilities, values)


def _date_quantile(column: ColumnSchema) -> Quantile:
    if column.start is None or column.end is None:
        raise SchemaError(f"{column.name}: datetime columns need start and end")
    start = pd.Timestamp(column.start).value
    end = pd.Timestamp(column.end).value
    return lambda uniform: pd.Series(pd.to_datetime(start + uniform * (end - start)))


def _category_quantile(column: ColumnSchema) -> Quantile:
    if not column.choices:
        raise SchemaError(f"{column.name}: category columns need choices or a category")
    if column.weights is None:
        weights = np.ones(len(column.choices))
    else:
        weights = np.asarray(column.weights, dtype=float)
    cumulative = np.cumsum(weights) / weights.sum()
    choices = np.asarray(column.choices, dtype=object)

    def quantile(uniform: np.ndarray) -> pd.Series:
        indexes = np.minimum(np.searchsorted(cumulative, uniform, side="right"), len(choices) - 1)
        return pd.Series(choices[indexes], dtype=object)

    return quantile


def evenly_spread(uniform: np.ndarray) -> np.ndarray:
    """Replace u by its ranks spread evenly over (0, 1), keeping the order.

    Feeding these to a quantile function reproduces the distribution's shares
    exactly, for example 70% `free` in 1000 rows gives exactly 700 rows.
    """
    ranks = stats.rankdata(uniform, method="ordinal")
    return (ranks - 0.5) / len(uniform)


def match_mean_and_std(values: np.ndarray, column: ColumnSchema) -> np.ndarray:
    """Rescale values to the column's exact mean and std, then clip to its bounds.

    Clipping moves the mean and std again, so the rescale repeats a few times.
    """
    if column.mean is None or column.std is None or len(values) < 2:
        return values
    lower = -np.inf if column.minimum is None else column.minimum
    upper = np.inf if column.maximum is None else column.maximum
    adjusted = values.astype(float)
    for _ in range(5):
        spread = adjusted.std()
        if spread == 0:
            break
        adjusted = column.mean + (adjusted - adjusted.mean()) * (column.std / spread)
        adjusted = np.clip(adjusted, lower, upper)
    return adjusted
