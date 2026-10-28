"""Techniques that add random noise to numbers and dates."""

import narwhals as nw

from oblitus.table import Table
from oblitus.techniques.context import TechniqueContext
from oblitus.techniques.date_expressions import as_datetime_expression, shift_days
from oblitus.techniques.numeric_expressions import (
    as_number,
    finite_only,
    in_original_type,
    laplace,
    standard_normal,
)


def gaussian_noise(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    relative_scale: float = 0.1,
    scale: float | None = None,
    preserve_variance: bool = True,
) -> nw.Expr:
    """Add normal noise with deviation `scale`, or `relative_scale` times the column's.

    With `preserve_variance`, the noisy column is rescaled so its mean and standard
    deviation equal the original's. Correlations with other columns weaken by about
    a factor of 1 / sqrt(1 + relative_scale squared).
    """
    values = as_number(column)
    finite = finite_only(values)
    original = table.statistics(mean=finite.mean(), std=finite.std(ddof=0), present=finite.count())
    if original["present"] < 2:
        return nw.col(column)

    noise_scale = relative_scale * original["std"] if scale is None else scale
    first, second = table.uniforms(context.stream_key(column, "gaussian_noise"), 2)
    noisy = nw.col(table.add(values + standard_normal(first, second) * noise_scale))

    if preserve_variance:
        noisy_std = table.statistics(std=finite_only(noisy).std(ddof=0))["std"]
        if noisy_std > 0:
            scale_back = original["std"] / noisy_std
            noisy = (noisy - finite_only(noisy).mean()) * scale_back + original["mean"]
    return in_original_type(noisy, table.dtype(column))


def laplace_noise(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    epsilon: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
) -> nw.Expr:
    """Clip each value to [lower, upper] and add Laplace noise of scale (upper - lower) / epsilon.

    With `lower` and `upper` fixed in advance rather than read from the data, each
    released value satisfies epsilon-local differential privacy. Smaller epsilon means
    more privacy and more noise. Both bounds are required: the column's own minimum
    and maximum would leak, and the guarantee would not hold.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if lower is None or upper is None:
        raise ValueError(
            "laplace_noise needs lower and upper bounds chosen without looking at the data"
        )
    (uniform,) = table.uniforms(context.stream_key(column, "laplace_noise"))
    noise_scale = (upper - lower) / epsilon
    clipped = as_number(column).clip(lower_bound=lower, upper_bound=upper)
    return in_original_type(clipped + laplace(uniform) * noise_scale, table.dtype(column))


def multiplicative_noise(
    table: Table, column: str, context: TechniqueContext, *, relative_scale: float = 0.1
) -> nw.Expr:
    """Multiply each value by a factor drawn around 1, so large values move more.

    Suits amounts such as salaries, where an absolute noise level would swamp small
    values and barely touch large ones. Signs never flip for factors above zero.
    """
    first, second = table.uniforms(context.stream_key(column, "multiplicative_noise"), 2)
    factor = (standard_normal(first, second) * relative_scale + 1.0).clip(lower_bound=0.01)
    return in_original_type(as_number(column) * factor, table.dtype(column))


def date_shift(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    max_days: int = 30,
    by: str | None = None,
    stream: str | None = None,
) -> nw.Expr:
    """Move each date by a random whole number of days between -max_days and +max_days.

    With `by`, every row of the same entity (a person, a store) moves by the same
    offset. Without it, each row gets its own offset, and averaging the many rows of
    one person recovers their true date within days.

    The offsets are keyed on the column's name, so same-named date columns of different
    tables already agree. Give `stream` a name to replace it: date columns of any names
    that share a `stream`, `by` values and `max_days` move each entity by the same offset.
    """
    dates = as_datetime_expression(table, column)
    (uniform,) = table.uniforms(context.stream_key(stream or column, "date_shift"), by=by)
    offsets_possible = 2 * max_days + 1
    offset = (uniform * offsets_possible).floor().cast(nw.Int64) - max_days
    return shift_days(dates, offset, range(-max_days, max_days + 1))
