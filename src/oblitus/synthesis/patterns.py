"""Planted patterns: conditional overrides, formulas, outliers, exact statistics and gaps.

The generator calls these per column, in dependency order, so every condition and
formula sees the final values of the columns it reads.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from oblitus.synthesis.draw_replay import ColumnDraws, normal_from_uniform
from oblitus.synthesis.expressions import LagSource, evaluate
from oblitus.synthesis.logistic_calibration import calibrate_probabilities
from oblitus.synthesis.marginals import (
    evenly_spread,
    match_mean_and_std,
    quantile_function,
)
from oblitus.synthesis.missingness import null_shares_from_value
from oblitus.synthesis.schema import DRAWN_KINDS, ColumnOverride, ColumnSchema
from oblitus.taxonomy import NUMERIC_KINDS, ColumnKind

RegenerateRows = Callable[[ColumnSchema, int], pd.Series]


def draw(column: ColumnSchema, uniform: np.ndarray) -> pd.Series:
    """Draw a column from its distribution, exactly matching its statistics if asked."""
    if column.exact:
        uniform = evenly_spread(uniform)
    values = quantile_function(column)(uniform)
    if column.exact and column.kind in NUMERIC_KINDS:
        values = _as_kind(match_mean_and_std(values.to_numpy(), column), column)
    return values


def compute_formula(
    column: ColumnSchema,
    frame: pd.DataFrame,
    draws: ColumnDraws,
    lags: LagSource | None = None,
) -> pd.Series:
    """Evaluate a column's formula over the columns generated so far, plus its noise.

    In a panel, `lags` gives the earlier periods that `lag(...)` reads.
    """
    assert column.formula is not None
    computed = evaluate(column.formula, frame, uniforms=draws.next_call_site, lags=lags)
    if column.kind is ColumnKind.BOOLEAN:
        return pd.Series(np.asarray(computed).astype(bool))
    if column.kind not in NUMERIC_KINDS:
        return pd.Series(np.asarray(computed, dtype=object))
    values = np.asarray(computed, dtype=float)
    if column.noise_std:
        values = values + column.noise_std * normal_from_uniform(draws.noise())
    lower = -np.inf if column.minimum is None else column.minimum
    upper = np.inf if column.maximum is None else column.maximum
    values = np.clip(values, lower, upper)
    if column.probability_share is not None:
        values = calibrate_probabilities(values, column.probability_share)
    if column.exact:
        values = match_mean_and_std(values, column)
    return _as_kind(values, column)


def conditional_uniform(column: ColumnSchema, draws: ColumnDraws) -> np.ndarray | None:
    """Return the uniform numbers that rows redrawn by a condition keep their rank from.

    A drawn column reuses the numbers its values came from. A formula column has none, so it
    gets numbers of its own. Other columns have none and are redrawn from the generator.
    """
    if column.formula is None:
        return draws.base()
    if column.conditions and column.kind in DRAWN_KINDS:
        return draws.redraw()
    return None


def apply_conditions(
    column: ColumnSchema,
    values: pd.Series,
    frame: pd.DataFrame,
    uniform: np.ndarray | None,
    regenerate: RegenerateRows,
) -> pd.Series:
    """Redraw the rows each override selects, with the override's settings.

    Drawn columns reuse each row's uniform value, so a row in the top decile stays
    in the top decile of the overridden distribution.
    """
    result = values.copy()
    for override in column.conditions:
        selected = _selected(override, frame)
        if not selected.any():
            continue
        settings = column.with_override(override)
        if uniform is not None:
            redrawn = draw(settings, uniform[selected])
        else:
            redrawn = regenerate(settings, int(selected.sum()))
        result = _assign(result, selected, redrawn)
    return result


def plant_outliers(column: ColumnSchema, values: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Push a share of rows far into either tail, ignoring the column's bounds."""
    if column.outlier_fraction <= 0 or column.kind not in NUMERIC_KINDS:
        return values
    numbers = values.to_numpy(dtype=float).copy()
    count = int(round(column.outlier_fraction * len(numbers)))
    if count == 0:
        return values
    rows = rng.choice(len(numbers), size=count, replace=False)
    center = float(np.mean(numbers))
    spread = float(np.std(numbers)) or 1.0
    directions = rng.choice([-1.0, 1.0], size=count)
    # A little extra exponential distance keeps outliers from piling on one value.
    distances = column.outlier_scale + rng.exponential(0.5, size=count)
    numbers[rows] = center + directions * distances * spread
    if column.kind is ColumnKind.INTEGER:
        return pd.Series(np.round(numbers).astype("int64"))
    return pd.Series(numbers)


def apply_gaps(
    column: ColumnSchema,
    values: pd.Series,
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """Blank cells at the column's null fraction, or an override's, row by row.

    With a `missingness` setting the chance follows the column's own value or another column's,
    and still averages the null fraction.
    """
    shares = _null_shares(column, values, frame)
    for override in column.conditions:
        if override.null_fraction is not None:
            shares[_selected(override, frame)] = override.null_fraction
    missing = rng.random(len(values)) < shares
    if not missing.any():
        return values
    if pd.api.types.is_integer_dtype(values):
        values = values.astype("Int64")
    elif pd.api.types.is_bool_dtype(values):
        values = values.astype("boolean")
    return values.mask(missing)


def _null_shares(column: ColumnSchema, values: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    setting = column.missingness
    if setting is None:
        return np.full(len(values), column.null_fraction)
    driver = values if setting.depends_on is None else frame[setting.depends_on]
    return null_shares_from_value(driver.reset_index(drop=True), setting, column.null_fraction)


def _selected(override: ColumnOverride, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(evaluate(override.when, frame), dtype=bool)


def _assign(values: pd.Series, selected: np.ndarray, redrawn: pd.Series) -> pd.Series:
    result = values.astype(object) if values.dtype != redrawn.dtype else values.copy()
    result.iloc[np.flatnonzero(selected)] = redrawn.to_numpy()
    if pd.api.types.is_numeric_dtype(values) and pd.api.types.is_numeric_dtype(redrawn):
        return pd.Series(pd.to_numeric(result), name=values.name)
    return result


def _as_kind(values: np.ndarray, column: ColumnSchema) -> pd.Series:
    if column.kind is ColumnKind.INTEGER:
        return pd.Series(np.round(values).astype("int64"))
    return pd.Series(values.astype(float))
