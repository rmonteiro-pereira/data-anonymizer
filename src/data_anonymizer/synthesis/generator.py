"""Generate synthetic tables from a schema alone, or from a schema and a reference table.

One table is built in four steps:
1. draw a uniform number per row for every drawn column, correlated through a
   Gaussian copula where the schema fixes correlations;
2. in dependency order, turn them into values, compute formulas, apply conditional
   overrides and plant outliers;
3. blank cells at each column's null fraction;
4. when the schema has constraints, drop the rows that break one and draw more.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from data_anonymizer.errors import SchemaError
from data_anonymizer.plan import Fidelity
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.copula import fit_gaussian_copula, nearest_positive_definite
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
from data_anonymizer.taxonomy import ColumnKind, DataCategory

_UPPERCASE = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

# Uniform values are kept away from 0 and 1, where some inverse functions are infinite.
_EDGE = 1e-9

# How many extra batches constraint filtering may draw before giving up.
_CONSTRAINT_ATTEMPTS = 25

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

    return _with_constraints(schema, rows, draw_table, rng, surrogates)


def generate_from_reference(
    reference: pd.DataFrame,
    schema: DatasetSchema,
    rows: int,
    *,
    fidelity: Fidelity,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.DataFrame:
    """Generate a table that resembles `reference`.

    With statistical fidelity, the non-identifying columns are drawn jointly from a
    Gaussian copula fitted to the reference, so correlations survive. With privacy
    fidelity, every column is drawn independently from the schema, so no row-level
    structure of the reference survives. Patterns in the schema apply either way.
    """
    if fidelity is Fidelity.PRIVACY:
        return generate_from_schema(schema, rows, rng=rng, surrogates=surrogates)

    learned_kinds = {
        column.name: column.kind for column in schema.columns if _learnable(column, reference)
    }
    copula = fit_gaussian_copula(reference, learned_kinds, rng) if learned_kinds else None

    def draw_table(count: int) -> pd.DataFrame:
        learned = copula.sample(count, rng) if copula is not None else None
        return _draw_table(schema, count, rng, surrogates, learned=learned)

    return _with_constraints(schema, rows, draw_table, rng, surrogates)


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

    gapped = {
        column.name: apply_gaps(column, frame[column.name], frame, rng) for column in schema.columns
    }
    return pd.DataFrame(gapped)


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
        return pd.Series(_unique_integers(column, rows, rng))
    if column.kind in DRAWN_KINDS:
        return draw(column, rng.random(rows))
    if column.template is not None:
        return pd.Series([_fill_template(column.template, rng) for _ in range(rows)])
    if column.choices:
        return quantile_function(column.model_copy(update={"kind": ColumnKind.CATEGORY}))(
            rng.random(rows)
        )
    return pd.Series(surrogates.many(DataCategory.GENERIC_TEXT, rows, rng))


def _unique_integers(column: ColumnSchema, rows: int, rng: np.random.Generator) -> np.ndarray:
    lower = 1 if column.minimum is None else int(np.ceil(column.minimum))
    upper = lower + rows * 10 if column.maximum is None else int(np.floor(column.maximum))
    available = upper - lower + 1
    if available < rows:
        raise SchemaError(
            f"{column.name}: {rows} unique integers do not fit between {lower} and {upper}"
        )
    if available <= rows * 4:
        return rng.permutation(np.arange(lower, upper + 1))[:rows]
    # Drawing from a huge range and dropping repeats avoids building the whole range.
    drawn: set[int] = set()
    while len(drawn) < rows:
        batch = rng.integers(lower, upper + 1, size=rows - len(drawn))
        drawn.update(int(value) for value in batch)
    return rng.permutation(np.fromiter(drawn, dtype="int64"))


def _fill_template(template: str, rng: np.random.Generator) -> str:
    characters = []
    for character in template:
        if character == "#":
            characters.append(str(rng.integers(0, 10)))
        elif character == "?":
            characters.append(str(rng.choice(_UPPERCASE)))
        else:
            characters.append(character)
    return "".join(characters)


def _with_constraints(
    schema: DatasetSchema,
    rows: int,
    draw_table: TableDraw,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.DataFrame:
    if not schema.constraints:
        return draw_table(rows)

    kept: list[pd.DataFrame] = []
    have = 0
    acceptance = 1.0
    for _ in range(_CONSTRAINT_ATTEMPTS):
        needed = rows - have
        batch = draw_table(max(16, int(needed / max(acceptance, 0.05) * 1.2)))
        passing = _passes_constraints(schema, batch)
        acceptance = max(float(passing.mean()), 0.01)
        kept.append(batch[passing])
        have += int(passing.sum())
        if have >= rows:
            break
    else:
        raise SchemaError(
            f"constraints rejected too many rows; only {have} of {rows} passed. "
            "Check that the constraints can hold together."
        )

    result = pd.concat(kept, ignore_index=True).head(rows).reset_index(drop=True)
    # Batches were drawn separately, so unique keys are drawn again over the final rows.
    for column in schema.columns:
        if column.unique:
            values = _undrawn_values(column, rows, rng, surrogates)
            result[column.name] = apply_gaps(column, values, result, rng).to_numpy()
    return result


def _passes_constraints(schema: DatasetSchema, frame: pd.DataFrame) -> np.ndarray:
    passing = np.ones(len(frame), dtype=bool)
    for constraint in schema.constraints:
        outcome = np.asarray(evaluate(constraint, frame), dtype=bool)
        passing &= outcome
    return passing
