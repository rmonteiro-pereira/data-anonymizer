"""Build the values of one column: its uniform numbers, its draw or formula, and its patterns.

The generator calls `build_column` once per column for a plain table, and the panel generator
calls it once per column and period. Both get the same patterns applied in the same order.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from data_anonymizer.errors import SchemaError
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.copula import nearest_positive_definite
from data_anonymizer.synthesis.draw_replay import ColumnDraws
from data_anonymizer.synthesis.expressions import LagSource
from data_anonymizer.synthesis.marginals import quantile_function
from data_anonymizer.synthesis.patterns import (
    apply_conditions,
    compute_formula,
    conditional_uniform,
    draw,
    plant_outliers,
)
from data_anonymizer.synthesis.schema import DRAWN_KINDS, ColumnSchema, Correlation
from data_anonymizer.synthesis.unique_values import template_values, unique_integers
from data_anonymizer.taxonomy import ColumnKind, DataCategory

# Uniform values are kept away from 0 and 1, where some inverse functions are infinite.
_EDGE = 1e-9


def build_column(
    column: ColumnSchema,
    frame: pd.DataFrame,
    draws: ColumnDraws,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
    *,
    learned: pd.DataFrame | None = None,
    lags: LagSource | None = None,
) -> pd.Series:
    """Return a column's values for the rows of `frame`, which holds the columns it reads.

    A formula column is computed, a column in `learned` is taken from it, and any other is
    drawn from its distribution. Conditional overrides and planted outliers come last.
    """

    def regenerate(settings: ColumnSchema, count: int) -> pd.Series:
        return undrawn_values(settings, count, rng, surrogates)

    rows = len(frame)
    is_learned = column.formula is None and learned is not None and column.name in learned.columns
    base_uniform = draws.base()
    if column.formula is not None:
        values = compute_formula(column, frame, draws, lags)
    elif is_learned:
        assert learned is not None
        values = learned[column.name].reset_index(drop=True)
    elif base_uniform is not None:
        values = draw(column, base_uniform)
    else:
        values = undrawn_values(column, rows, rng, surrogates)
    uniform = None if is_learned else conditional_uniform(column, draws)
    values = apply_conditions(column, values, frame, uniform, regenerate)
    return plant_outliers(column, values, rng)


def is_drawn(column: ColumnSchema) -> bool:
    """True when the column's values come straight from a distribution and a uniform number."""
    return (
        column.kind in DRAWN_KINDS
        and column.category is None
        and column.formula is None
        and not column.unique
    )


def draw_uniforms(
    columns: Sequence[ColumnSchema],
    correlations: Sequence[Correlation],
    rows: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw a uniform number per row for every drawn column, correlated where asked."""
    uniforms = {column.name: rng.random(rows) for column in columns if is_drawn(column)}
    correlated = list(dict.fromkeys(name for item in correlations for name in item.columns))
    if not correlated or rows == 0:
        return uniforms

    position = {name: index for index, name in enumerate(correlated)}
    matrix = np.eye(len(correlated))
    for correlation in correlations:
        first, second = (position[name] for name in correlation.columns)
        # A Gaussian copula with Pearson r gives Spearman rho = 6/pi * arcsin(r/2).
        pearson = 2 * np.sin(np.pi * correlation.value / 6)
        matrix[first, second] = pearson
        matrix[second, first] = pearson
    scores = rng.multivariate_normal(
        np.zeros(len(correlated)), nearest_positive_definite(matrix), size=rows
    )
    correlated_uniforms = np.clip(stats.norm.cdf(scores), _EDGE, 1 - _EDGE)
    for name, index in position.items():
        uniforms[name] = correlated_uniforms[:, index]
    return uniforms


def undrawn_values(
    column: ColumnSchema,
    rows: int,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.Series:
    """Values for columns that are not drawn from a distribution: keys, fakes and text."""
    if column.category is not None and column.kind in (ColumnKind.TEXT, ColumnKind.CATEGORY):
        return pd.Series(surrogates.many(column.category, rows, rng, unique=column.unique))
    if column.unique and column.kind is ColumnKind.INTEGER:
        return pd.Series(unique_integers(column, rows, rng))
    if column.kind in DRAWN_KINDS:
        return _drawn_values(column, rows, rng)
    if column.template is not None:
        return pd.Series(template_values(column, rows, rng))
    if column.choices:
        return quantile_function(column.model_copy(update={"kind": ColumnKind.CATEGORY}))(
            rng.random(rows)
        )
    return pd.Series(surrogates.many(DataCategory.GENERIC_TEXT, rows, rng, unique=column.unique))


def _drawn_values(column: ColumnSchema, rows: int, rng: np.random.Generator) -> pd.Series:
    values = draw(column, rng.random(rows))
    if column.unique and values.duplicated().any():
        raise SchemaError(f"{column.name}: this distribution cannot give {rows} unique values")
    return values
