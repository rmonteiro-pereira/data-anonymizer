"""Techniques that move real values between records or merge them into groups."""

import narwhals as nw

from data_anonymizer.table import Table
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.numeric_expressions import as_number, in_original_type


def shuffle(table: Table, column: str, context: TechniqueContext) -> nw.Expr:
    """Randomly reorder the column's values among the rows.

    The column's distribution stays exact, but its link to every other column in the
    same row is broken. Rows are ordered by a keyed random number, and the sorted
    values are handed out in that order.
    """
    (sort_key,) = table.uniforms(context.stream_key(column, "shuffle"))
    return table.reorder_values(column, sort_key)


def rank_swap(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    max_distance_percent: float = 5.0,
) -> nw.Expr:
    """Move values between records whose ranks are at most `max_distance_percent` apart.

    Each record's sort key is its value's rank plus a random number between 0 and
    `window`, which is `max_distance_percent` of the record count (at least 1). The
    sorted values are handed out in the order of those keys. A value therefore only
    trades places with records within about one window of its rank, and the column's
    distribution stays exact. Its correlations with other columns change little.
    """
    values = nw.col(column)
    present = table.statistics(present=values.count())["present"]
    if present < 2:
        return values
    window = max(1.0, present * max_distance_percent / 100)
    (uniform,) = table.uniforms(context.stream_key(column, "rank_swap"))
    # SQL engines refuse a window inside another window, so the sort key is stored first.
    rank = values.rank("ordinal").cast(nw.Float64)
    sort_key = nw.col(table.add(rank + uniform * window))
    return table.reorder_values(column, sort_key)


def microaggregate(
    table: Table, column: str, context: TechniqueContext, *, group_size: int = 5
) -> nw.Expr:
    """Replace each value by the mean of a group of at least `group_size` neighbouring values.

    Every released value is shared by at least `group_size` records, and the column
    mean is unchanged. The spread shrinks a little because values inside a group merge.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    values = as_number(column)
    present = table.statistics(present=values.count())["present"]
    if present < group_size:
        return in_original_type(values.mean(), table.dtype(column))

    # Groups of consecutive ranks; the remainder joins the last group, so no group
    # is smaller than group_size. Ties are broken by row order in memory.
    last_group = present // group_size - 1
    rank_group = ((values.rank("ordinal") - 1) // group_size).clip(upper_bound=last_group)
    group = table.add(rank_group)
    averaged = nw.when(~values.is_null()).then(values.mean().over(group))
    return in_original_type(averaged, table.dtype(column))


def randomized_response(
    table: Table, column: str, context: TechniqueContext, *, keep_probability: float = 0.75
) -> nw.Expr:
    """Keep each category with `keep_probability`, otherwise replace it with a random category.

    This is randomized response, a form of local differential privacy: any single
    released value is deniable. With k categories, epsilon equals
    ln((p + (1 - p) / k) / ((1 - p) / k)) for keep probability p. Category shares
    across many rows can still be estimated by inverting the mixing.
    """
    if not 0 < keep_probability < 1:
        raise ValueError("keep_probability must be between 0 and 1, exclusive")
    categories = sorted(table.distinct_values(column))
    values = nw.col(column)
    if len(categories) < 2:
        return values

    keep_or_replace, which_category = table.uniforms(
        context.stream_key(column, "randomized_response"), 2
    )
    category_number = table.add((which_category * len(categories)).floor().cast(nw.Int64))
    random_choice = table.lookup(category_number, dict(enumerate(categories)), table.dtype(column))
    replace = ~values.is_null() & (keep_or_replace >= keep_probability)
    return nw.when(replace).then(random_choice).otherwise(values)
