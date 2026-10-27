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


def per_group(expression: nw.Expr, group: str | None) -> nw.Expr:
    """Compute an aggregate within each group of another column, or over the whole column."""
    if group is None:
        return expression
    return expression.over(group)
