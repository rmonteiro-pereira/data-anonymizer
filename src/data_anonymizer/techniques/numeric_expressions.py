"""Expression building blocks shared by the techniques that work on numbers."""

import math
from typing import Any

import narwhals as nw


def as_number(column: str) -> nw.Expr:
    """The column as floating-point numbers."""
    return nw.col(column).cast(nw.Float64)


def in_original_type(expression: nw.Expr, dtype: Any) -> nw.Expr:
    """Round back to whole numbers when the column held integers."""
    if dtype.is_integer():
        return expression.round(0).cast(dtype)
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
