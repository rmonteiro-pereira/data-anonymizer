"""Techniques that add random noise to numbers and dates."""

import numpy as np
import pandas as pd

from data_anonymizer.detection.kinds import as_datetime
from data_anonymizer.techniques.context import TechniqueContext


def gaussian_noise(
    values: pd.Series,
    context: TechniqueContext,
    *,
    relative_scale: float = 0.1,
    scale: float | None = None,
    preserve_variance: bool = True,
) -> pd.Series:
    """Add normal noise with deviation `scale`, or `relative_scale` times the column's.

    With `preserve_variance`, the noisy column is rescaled so its mean and standard
    deviation equal the original's. Correlations with other columns weaken by about
    a factor of 1 / sqrt(1 + relative_scale squared).
    """
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    present = numbers.notna()
    original = numbers[present].to_numpy()
    if len(original) < 2:
        return values.copy()

    original_std = float(original.std())
    noise_scale = relative_scale * original_std if scale is None else scale
    noisy = original + context.rng.normal(0.0, noise_scale, size=len(original))

    noisy_std = float(noisy.std())
    if preserve_variance and noisy_std > 0:
        noisy = original.mean() + (noisy - noisy.mean()) * (original_std / noisy_std)

    return _write_back(values, numbers, present, noisy)


def laplace_noise(
    values: pd.Series,
    context: TechniqueContext,
    *,
    epsilon: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
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
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    present = numbers.notna()
    original = numbers[present].to_numpy()
    if len(original) == 0:
        return values.copy()

    clipped = np.clip(original, lower, upper)
    sensitivity = upper - lower
    noisy = clipped + context.rng.laplace(0.0, sensitivity / epsilon, size=len(clipped))
    return _write_back(values, numbers, present, noisy)


def multiplicative_noise(
    values: pd.Series, context: TechniqueContext, *, relative_scale: float = 0.1
) -> pd.Series:
    """Multiply each value by a factor drawn around 1, so large values move more.

    Suits amounts such as salaries, where an absolute noise level would swamp small
    values and barely touch large ones. Signs never flip for factors above zero.
    """
    numbers = pd.to_numeric(values, errors="coerce").astype(float)
    present = numbers.notna()
    original = numbers[present].to_numpy()
    factors = context.rng.normal(1.0, relative_scale, size=len(original))
    factors = np.clip(factors, 0.01, None)
    return _write_back(values, numbers, present, original * factors)


def date_shift(
    values: pd.Series,
    context: TechniqueContext,
    *,
    max_days: int = 30,
    by: str | None = None,
) -> pd.Series:
    """Move each date by a random whole number of days between -max_days and +max_days.

    With `by`, every row of the same entity (a person, a store) moves by the same
    offset. Without it, each row gets its own offset, and averaging the many rows of
    one person recovers their true date within days.
    """
    dates = as_datetime(values)
    if by is None:
        offsets = context.rng.integers(-max_days, max_days + 1, size=len(dates))
        return dates + pd.to_timedelta(offsets, unit="D")
    entities = context.group_keys(by).reindex(values.index)
    codes, uniques = pd.factorize(entities, use_na_sentinel=False)
    per_entity = context.rng.integers(-max_days, max_days + 1, size=len(uniques))
    return dates + pd.to_timedelta(per_entity[codes], unit="D")


def _write_back(
    values: pd.Series, numbers: pd.Series, present: pd.Series, noisy: np.ndarray
) -> pd.Series:
    result = numbers.copy()
    result[present] = noisy
    if pd.api.types.is_integer_dtype(values):
        return result.round().astype(values.dtype)
    return result
