"""Generate synthetic tables from a schema alone, or from a schema and a reference table.

One table is built in four steps:
1. draw a uniform number per row for every drawn column, correlated through a
   Gaussian copula where the schema fixes correlations;
2. in dependency order, turn them into values, compute formulas, apply conditional
   overrides and plant outliers;
3. when the schema has constraints, drop the rows that break one, and the rows that repeat
   a unique key, and draw more until enough rows are kept;
4. blank cells at each column's null fraction.

Constraints see the table before cells are blanked, so a null never removes a row.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from data_anonymizer.errors import SchemaError
from data_anonymizer.plan import Fidelity
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.copula import fit_stratified_copula, nearest_positive_definite
from data_anonymizer.synthesis.expressions import evaluate
from data_anonymizer.synthesis.marginals import quantile_function
from data_anonymizer.synthesis.patterns import (
    apply_conditions,
    apply_gaps,
    compute_formula,
    draw,
    plant_outliers,
)
from data_anonymizer.synthesis.schema import DRAWN_KINDS, ColumnSchema, DatasetSchema
from data_anonymizer.synthesis.unique_values import (
    template_values,
    unique_integer_capacity,
    unique_integers,
)
from data_anonymizer.taxonomy import ColumnKind, DataCategory

# Uniform values are kept away from 0 and 1, where some inverse functions are infinite.
_EDGE = 1e-9

# Constraint filtering draws batches sized from the share of rows that passed so far. It
# gives up after drawing this many rows per requested row, within the two bounds below.
_DRAWS_PER_REQUESTED_ROW = 200
_MIN_DRAW_BUDGET = 2_000_000
_MAX_DRAW_BUDGET = 20_000_000
_MIN_BATCH_ROWS = 16
_MAX_BATCH_ROWS = 250_000

TableDraw = Callable[[int], pd.DataFrame]


def generate_from_schema(
    schema: DatasetSchema,
    rows: int,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.DataFrame:
    """Generate a table that follows the schema's columns, correlations and patterns."""

    def draw_table(count: int) -> pd.DataFrame:
        return _draw_table(schema, count, rng, surrogates, learned=None)

    return _build_table(schema, rows, draw_table, rng)


def generate_from_reference(
    reference: pd.DataFrame,
    schema: DatasetSchema,
    rows: int,
    *,
    fidelity: Fidelity,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
    stratify_by: Sequence[str] = (),
) -> pd.DataFrame:
    """Generate a table that resembles `reference`.

    With statistical fidelity, the non-identifying columns are drawn jointly from a
    Gaussian copula fitted to the reference, so correlations survive. With privacy
    fidelity, every column is drawn independently from the schema, so no row-level
    structure of the reference survives. Patterns in the schema apply either way.

    `stratify_by` names category or boolean columns. With statistical fidelity, each of their
    levels that has enough rows gets its own copula, so the level keeps its own distributions
    and correlations; the small levels share one. It has no effect with privacy fidelity.
    """
    if fidelity is Fidelity.PRIVACY:
        return generate_from_schema(schema, rows, rng=rng, surrogates=surrogates)

    learned_kinds = {
        column.name: column.kind for column in schema.columns if _learnable(column, reference)
    }
    _check_stratify_by(stratify_by, learned_kinds)
    copula = fit_stratified_copula(reference, learned_kinds, stratify_by) if learned_kinds else None

    def draw_table(count: int) -> pd.DataFrame:
        learned = copula.sample(count, rng) if copula is not None else None
        return _draw_table(schema, count, rng, surrogates, learned=learned)

    return _build_table(schema, rows, draw_table, rng)


def _check_stratify_by(stratify_by: Sequence[str], learned_kinds: dict[str, ColumnKind]) -> None:
    for name in stratify_by:
        if learned_kinds.get(name) not in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN):
            raise SchemaError(
                f"stratify_by: {name!r} must be a category or boolean column of the "
                "schema that is learned from the reference table"
            )


def _learnable(column: ColumnSchema, reference: pd.DataFrame) -> bool:
    if column.name not in reference.columns or column.kind not in DRAWN_KINDS:
        return False
    # Identifiers are regenerated, never learned: unique columns and fake-value columns.
    return column.category is None and not column.unique and column.formula is None


def _is_drawn(column: ColumnSchema) -> bool:
    return (
        column.kind in DRAWN_KINDS
        and column.category is None
        and column.formula is None
        and not column.unique
    )


def _draw_table(
    schema: DatasetSchema,
    rows: int,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
    *,
    learned: pd.DataFrame | None,
) -> pd.DataFrame:
    uniforms = _uniforms(schema, rows, rng)
    frame = pd.DataFrame(index=range(rows))

    def regenerate(settings: ColumnSchema, count: int) -> pd.Series:
        return _undrawn_values(settings, count, rng, surrogates)

    for name in schema.generation_order():
        column = schema.column(name)
        uniform = uniforms.get(name)
        if column.formula is not None:
            values = compute_formula(column, frame, rng)
        elif learned is not None and name in learned.columns:
            values = learned[name].reset_index(drop=True)
            uniform = None
        elif uniform is not None:
            values = draw(column, uniform)
        else:
            values = _undrawn_values(column, rows, rng, surrogates)
        values = apply_conditions(column, values, frame, uniform, regenerate)
        frame[name] = plant_outliers(column, values, rng).to_numpy()

    return frame


def _uniforms(schema: DatasetSchema, rows: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    uniforms = {column.name: rng.random(rows) for column in schema.columns if _is_drawn(column)}
    correlated = list(dict.fromkeys(name for item in schema.correlations for name in item.columns))
    if not correlated or rows == 0:
        return uniforms

    position = {name: index for index, name in enumerate(correlated)}
    matrix = np.eye(len(correlated))
    for correlation in schema.correlations:
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


def _undrawn_values(
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


def _build_table(
    schema: DatasetSchema, rows: int, draw_table: TableDraw, rng: np.random.Generator
) -> pd.DataFrame:
    table = _kept_rows(schema, rows, draw_table)
    gapped = {
        column.name: apply_gaps(column, table[column.name], table, rng) for column in schema.columns
    }
    return pd.DataFrame(gapped)


def _kept_rows(schema: DatasetSchema, rows: int, draw_table: TableDraw) -> pd.DataFrame:
    """Draw batches until `rows` rows pass every constraint and repeat no unique key."""
    if not schema.constraints:
        return draw_table(rows)
    capacity = _batch_capacity(schema, rows)
    seen_keys: dict[str, set[Any]] = {
        column.name: set() for column in schema.columns if column.unique
    }
    budget = min(max(_MIN_DRAW_BUDGET, _DRAWS_PER_REQUESTED_ROW * rows), _MAX_DRAW_BUDGET)
    kept: list[pd.DataFrame] = []
    have = 0
    drawn = 0
    while have < rows and drawn < budget:
        size = _batch_size(rows - have, have, drawn, capacity)
        batch = draw_table(size)
        drawn += size
        accepted = batch[_passes_constraints(schema, batch)]
        accepted = _without_repeated_keys(accepted, seen_keys)
        kept.append(accepted)
        have += len(accepted)
    if have < rows:
        raise SchemaError(
            f"constraints accepted only {have} of {drawn} drawn rows ({have / drawn:.3%}), "
            f"so {rows} rows could not be found. Check that the constraints can hold together."
        )
    return pd.concat(kept, ignore_index=True).head(rows)


def _batch_capacity(schema: DatasetSchema, rows: int) -> int:
    """Return the largest batch that the schema's bounded unique integer keys can fill."""
    capacity = _MAX_BATCH_ROWS
    for column in schema.columns:
        keys_that_fit = unique_integer_capacity(column) if column.unique else None
        if keys_that_fit is None:
            continue
        if keys_that_fit < rows:
            raise SchemaError(
                f"{column.name}: {rows} unique integers do not fit in a range of {keys_that_fit}"
            )
        capacity = min(capacity, keys_that_fit)
    return capacity


def _batch_size(needed: int, have: int, drawn: int, capacity: int) -> int:
    # The acceptance estimate starts at 1 and is smoothed so that zero passes stay finite.
    acceptance = 1.0 if drawn == 0 else (have + 1) / (drawn + 2)
    wanted = math.ceil(needed / acceptance * 1.2)
    return max(_MIN_BATCH_ROWS, min(wanted, capacity))


def _without_repeated_keys(accepted: pd.DataFrame, seen_keys: dict[str, set[Any]]) -> pd.DataFrame:
    """Drop rows that repeat a unique key already kept, and remember the keys that stay."""
    keep = np.ones(len(accepted), dtype=bool)
    for name, seen in seen_keys.items():
        keep &= ~accepted[name].duplicated().to_numpy()
        keep &= ~accepted[name].isin(seen).to_numpy()
    unique_rows = accepted[keep]
    for name, seen in seen_keys.items():
        seen.update(unique_rows[name])
    return unique_rows


def _passes_constraints(schema: DatasetSchema, frame: pd.DataFrame) -> np.ndarray:
    passing = np.ones(len(frame), dtype=bool)
    for constraint in schema.constraints:
        outcome = np.asarray(evaluate(constraint, frame), dtype=bool)
        passing &= outcome
    return passing
