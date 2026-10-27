"""Generate synthetic tables from a schema alone, or from a schema and a reference table."""

import numpy as np
import pandas as pd

from data_anonymizer.errors import SchemaError
from data_anonymizer.plan import Fidelity
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.copula import fit_gaussian_copula
from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema
from data_anonymizer.taxonomy import ColumnKind, DataCategory

# Normal draws outside the bounds are redrawn this many times before being clipped.
_REDRAWS = 10

_UPPERCASE = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

_COPULA_KINDS = {
    ColumnKind.INTEGER,
    ColumnKind.FLOAT,
    ColumnKind.DATETIME,
    ColumnKind.CATEGORY,
    ColumnKind.BOOLEAN,
}


def generate_from_schema(
    schema: DatasetSchema,
    rows: int,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.DataFrame:
    """Generate every column independently from the schema's description of it."""
    data = {
        column.name: _with_nulls(_generate_column(column, rows, rng, surrogates), column, rng)
        for column in schema.columns
    }
    return pd.DataFrame(data)


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
    structure of the reference survives.
    """
    if fidelity is Fidelity.PRIVACY:
        return generate_from_schema(schema, rows, rng=rng, surrogates=surrogates)

    copula_columns = {
        column.name: column.kind for column in schema.columns if _fits_copula(column, reference)
    }
    sampled = pd.DataFrame(index=range(rows))
    if copula_columns:
        copula = fit_gaussian_copula(reference, copula_columns, rng)
        sampled = copula.sample(rows, rng)

    data = {}
    for column in schema.columns:
        if column.name in copula_columns:
            values = sampled[column.name]
        else:
            values = _generate_column(column, rows, rng, surrogates)
        data[column.name] = _with_nulls(values, column, rng)
    return pd.DataFrame(data)


def _fits_copula(column: ColumnSchema, reference: pd.DataFrame) -> bool:
    if column.name not in reference.columns or column.kind not in _COPULA_KINDS:
        return False
    # Identifiers are regenerated, never learned: unique columns and fake-value columns.
    return column.category is None and not column.unique


def _generate_column(
    column: ColumnSchema,
    rows: int,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.Series:
    if column.category is not None and column.kind in (ColumnKind.TEXT, ColumnKind.CATEGORY):
        return pd.Series(surrogates.many(column.category, rows, rng, unique=column.unique))
    if column.kind in (ColumnKind.INTEGER, ColumnKind.FLOAT):
        return _numbers(column, rows, rng)
    if column.kind is ColumnKind.BOOLEAN:
        true_share = 0.5 if column.true_share is None else column.true_share
        return pd.Series(rng.random(rows) < true_share)
    if column.kind is ColumnKind.DATETIME:
        return _dates(column, rows, rng)
    if column.kind is ColumnKind.CATEGORY:
        return _choices(column, rows, rng)
    return _text(column, rows, rng, surrogates)


def _numbers(column: ColumnSchema, rows: int, rng: np.random.Generator) -> pd.Series:
    if column.unique and column.kind is ColumnKind.INTEGER:
        return pd.Series(_unique_integers(column, rows, rng))

    if column.mean is not None and column.std is not None:
        values = _bounded_normal(column, column.mean, column.std, rows, rng)
    elif column.minimum is not None and column.maximum is not None:
        values = rng.uniform(column.minimum, column.maximum, size=rows)
    else:
        raise SchemaError(f"{column.name}: give minimum and maximum, or mean and std")

    if column.kind is ColumnKind.INTEGER:
        return pd.Series(np.round(values).astype("int64"))
    return pd.Series(values)


def _bounded_normal(
    column: ColumnSchema, mean: float, std: float, rows: int, rng: np.random.Generator
) -> np.ndarray:
    lower = -np.inf if column.minimum is None else column.minimum
    upper = np.inf if column.maximum is None else column.maximum
    values = rng.normal(mean, std, size=rows)
    for _ in range(_REDRAWS):
        outside = (values < lower) | (values > upper)
        if not outside.any():
            break
        values[outside] = rng.normal(mean, std, size=int(outside.sum()))
    return np.clip(values, lower, upper)


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


def _dates(column: ColumnSchema, rows: int, rng: np.random.Generator) -> pd.Series:
    if column.start is None or column.end is None:
        raise SchemaError(f"{column.name}: datetime columns need start and end")
    start = pd.Timestamp(column.start).value
    end = pd.Timestamp(column.end).value
    nanoseconds = rng.integers(start, end + 1, size=rows)
    return pd.Series(pd.to_datetime(nanoseconds))


def _choices(column: ColumnSchema, rows: int, rng: np.random.Generator) -> pd.Series:
    if not column.choices:
        raise SchemaError(f"{column.name}: category columns need choices or a category")
    probabilities = None
    if column.weights is not None:
        weights = np.asarray(column.weights, dtype=float)
        probabilities = weights / weights.sum()
    picked = rng.choice(np.asarray(column.choices, dtype=object), size=rows, p=probabilities)
    return pd.Series(picked, dtype=object)


def _text(
    column: ColumnSchema,
    rows: int,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> pd.Series:
    if column.template is not None:
        return pd.Series([_fill_template(column.template, rng) for _ in range(rows)])
    if column.choices:
        return _choices(column, rows, rng)
    return pd.Series(surrogates.many(DataCategory.GENERIC_TEXT, rows, rng))


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


def _with_nulls(values: pd.Series, column: ColumnSchema, rng: np.random.Generator) -> pd.Series:
    values = values.reset_index(drop=True)
    if column.null_fraction <= 0:
        return values
    missing = rng.random(len(values)) < column.null_fraction
    if pd.api.types.is_integer_dtype(values):
        values = values.astype("Int64")
    elif pd.api.types.is_bool_dtype(values):
        values = values.astype("boolean")
    return values.mask(missing)
