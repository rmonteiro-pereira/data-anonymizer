"""Techniques that move real values between records or merge them into groups."""

import numpy as np
import pandas as pd

from data_anonymizer.techniques.context import TechniqueContext


def shuffle(values: pd.Series, context: TechniqueContext) -> pd.Series:
    """Randomly reorder the column's values among the rows.

    The column's distribution stays exact, but its link to every other column in the
    same row is broken.
    """
    permuted = context.rng.permutation(values.to_numpy())
    return pd.Series(permuted, index=values.index, name=values.name, dtype=values.dtype)


def rank_swap(
    values: pd.Series, context: TechniqueContext, *, max_distance_percent: float = 5.0
) -> pd.Series:
    """Swap each value with another whose rank is at most `max_distance_percent` away.

    The column's distribution stays exact and its correlations with other columns
    change little, because values only trade places with near neighbours.
    """
    present = values.notna()
    numbers = values[present].to_numpy()
    count = len(numbers)
    if count < 2:
        return values.copy()

    order = np.argsort(numbers, kind="stable")
    ranked = numbers[order].copy()
    window = max(1, int(count * max_distance_percent / 100))
    swapped = np.zeros(count, dtype=bool)
    # One random partner per rank keeps this linear; a partner already swapped is skipped.
    for rank in range(count):
        if swapped[rank]:
            continue
        partner = rank + int(context.rng.integers(1, window + 1))
        if partner >= count or swapped[partner]:
            continue
        ranked[rank], ranked[partner] = ranked[partner], ranked[rank]
        swapped[rank] = True
        swapped[partner] = True

    restored = np.empty_like(ranked)
    restored[order] = ranked
    result = values.copy()
    result[present] = restored
    return result


def microaggregate(
    values: pd.Series, context: TechniqueContext, *, group_size: int = 5
) -> pd.Series:
    """Replace each value by the mean of a group of at least `group_size` neighbouring values.

    Every released value is shared by at least `group_size` records, and the column
    mean is unchanged. The spread shrinks a little because values inside a group merge.
    """
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    present = numbers.notna()
    original = numbers[present].to_numpy()
    count = len(original)
    if count < group_size:
        averaged = np.full(count, original.mean() if count else 0.0)
    else:
        averaged = _group_means(original, group_size)

    result = numbers.copy()
    result[present] = averaged
    if pd.api.types.is_integer_dtype(values):
        return result.round().astype(values.dtype)
    return result


def _group_means(original: np.ndarray, group_size: int) -> np.ndarray:
    order = np.argsort(original, kind="stable")
    sorted_values = original[order]
    count = len(sorted_values)
    group_count = count // group_size
    # The remainder joins the last group so no group is smaller than group_size.
    group_of_rank = np.minimum(np.arange(count) // group_size, group_count - 1)
    sums = np.bincount(group_of_rank, weights=sorted_values)
    sizes = np.bincount(group_of_rank)
    means_by_rank = (sums / sizes)[group_of_rank]
    averaged = np.empty(count)
    averaged[order] = means_by_rank
    return averaged


def randomized_response(
    values: pd.Series, context: TechniqueContext, *, keep_probability: float = 0.75
) -> pd.Series:
    """Keep each category with `keep_probability`, otherwise replace it with a random category.

    This is randomized response, a form of local differential privacy: any single
    released value is deniable. With k categories, epsilon equals
    ln((p + (1 - p) / k) / ((1 - p) / k)) for keep probability p. Category shares
    across many rows can still be estimated by inverting the mixing.
    """
    if not 0 < keep_probability < 1:
        raise ValueError("keep_probability must be between 0 and 1, exclusive")
    present = values.notna()
    originals = values[present].to_numpy(dtype=object)
    categories = pd.unique(originals)
    if len(categories) < 2:
        return values.copy()

    replace = context.rng.random(len(originals)) >= keep_probability
    random_choices = context.rng.choice(categories, size=len(originals))
    released = np.where(replace, random_choices, originals)
    result = values.astype(object).copy()
    result[present] = released
    return result
