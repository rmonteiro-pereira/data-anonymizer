"""Planted patterns: conditional overrides, formulas, outliers, exact statistics and gaps.

The generator calls these per column, in dependency order, so every condition and
formula sees the final values of the columns it reads.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from data_anonymizer.synthesis.expressions import evaluate
from data_anonymizer.synthesis.marginals import (
    evenly_spread,
    match_mean_and_std,
    quantile_function,
)
from data_anonymizer.synthesis.schema import ColumnOverride, ColumnSchema
from data_anonymizer.taxonomy import NUMERIC_KINDS, ColumnKind

# Fields of an override that change the distribution rather than the row selection or gaps.
_NOT_DISTRIBUTION_FIELDS = {"when", "null_fraction"}

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
    column: ColumnSchema, frame: pd.DataFrame, rng: np.random.Generator
) -> pd.Series:
    """Evaluate a column's formula over the columns generated so far, plus its noise."""
    assert column.formula is not None
    computed = evaluate(column.formula, frame, rng)
    if column.kind is ColumnKind.BOOLEAN:
        return pd.Series(np.asarray(computed).astype(bool))
    if column.kind not in NUMERIC_KINDS:
        return pd.Series(np.asarray(computed, dtype=object))
    values = np.asarray(computed, dtype=float)
    if column.noise_std:
        values = values + rng.normal(0.0, column.noise_std, size=len(values))
    lower = -np.inf if column.minimum is None else column.minimum
    upper = np.inf if column.maximum is None else column.maximum
    values = np.clip(values, lower, upper)
    if column.exact:
        values = match_mean_and_std(values, column)
    return _as_kind(values, column)


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
        settings = _merged(column, override)
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
    """Blank cells at the column's null fraction, or an override's, row by row."""
    shares = np.full(len(values), column.null_fraction)
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


def _selected(override: ColumnOverride, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(evaluate(override.when, frame), dtype=bool)


def _merged(column: ColumnSchema, override: ColumnOverride) -> ColumnSchema:
    changes = {
        field: value
        for field, value in override.model_dump().items()
        if field not in _NOT_DISTRIBUTION_FIELDS and value is not None
    }
    if "mean" in changes or "std" in changes or "quantiles" in changes:
        # A new mean or spread replaces the shape inferred from the base settings.
        changes.setdefault("distribution", column.distribution)
    return column.model_copy(update={**changes, "conditions": [], "formula": None})


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
