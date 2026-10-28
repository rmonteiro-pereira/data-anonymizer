"""Expression building blocks shared by the techniques that work on numbers."""

import math
from typing import Any

import narwhals as nw


def as_number(column: str) -> nw.Expr:
    """The column as floating-point numbers."""
    return nw.col(column).cast(nw.Float64)


def finite_only(values: nw.Expr) -> nw.Expr:
    """The values with infinities left out, for column statistics.

    One infinite cell makes a mean or a standard deviation infinite, which would turn every
    other value of the column into NaN. The infinite cell itself stays infinite.
    """
    return nw.when(values.is_finite()).then(values)


def round_half_away(expression: nw.Expr, decimals: int = 0) -> nw.Expr:
    """Round to `decimals` places, with halves going away from zero on every engine.

    Polars rounds halves to even and DuckDB away from zero, so the built-in round would
    release different values depending on where the table lives.
    """
    factor = float(10**decimals)
    scaled = expression.abs() * factor
    whole = scaled.floor()
    # Comparing the fraction with a half, rather than flooring value + 0.5, keeps the
    # largest float below 0.5 from rounding up: adding 0.5 to it rounds to exactly 1.
    rounds_up = (scaled - whole) >= 0.5
    magnitude = nw.when(rounds_up).then(whole + 1).otherwise(whole) / factor
    # A negative value that rounds to zero gives zero, not -0.0.
    is_negative = (expression < 0) & (magnitude > 0)
    return nw.when(is_negative).then(-magnitude).otherwise(magnitude)


def in_original_type(expression: nw.Expr, dtype: Any) -> nw.Expr:
    """Round back to whole numbers when the column held integers."""
    if dtype.is_integer():
        return round_half_away(expression).cast(dtype)
    return expression


def standard_normal(first: nw.Expr, second: nw.Expr) -> nw.Expr:
    """A standard normal number from two independent uniforms, by the Box–Muller transform."""
    return (first.log() * -2.0).sqrt() * (second * (2 * math.pi)).cos()


def laplace(uniform: nw.Expr) -> nw.Expr:
    """A standard Laplace number (scale 1) from one uniform, by the inverse CDF."""
    centered = uniform - 0.5
    sign = nw.when(centered < 0).then(-1.0).otherwise(1.0)
    return sign * (1.0 - centered.abs() * 2.0).log() * -1.0


def per_group(expression: nw.Expr, group: str | None) -> nw.Expr:
    """Compute an aggregate within each group of another column, or over the whole column."""
    if group is None:
        return expression
    return expression.over(group)
