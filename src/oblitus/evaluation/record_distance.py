"""Distance from each record to its nearest records in another table, for mixed columns.

A synthetic table that memorized its training data sits unusually close to it. This module
measures that closeness with a Gower-like distance, the mean over columns of a per-column
distance between 0 and 1:
- a category column counts 0 when the values match and 1 when they differ;
- a number or date column counts the gap in robust standard deviations of the reference
  table, divided by `FULL_MISMATCH_GAP` and capped at 1;
- a blank matches only another blank.

The distances are computed in chunks of query rows, so memory stays at a few million cells
whatever the table sizes. Time grows with the product of the two row counts, which is why
callers cut the tables to `MAX_COMPARED_ROWS` first.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Largest number of rows of each table compared pair by pair. 10,000 x 10,000 rows is 10^8
# pairs per column, a matter of seconds; the full tables would take hours.
MAX_COMPARED_ROWS = 10_000

# A gap of this many robust standard deviations counts as a complete mismatch.
FULL_MISMATCH_GAP = 3.0

# Cells of the distance block held in memory at once.
_CHUNK_CELLS = 4_000_000

# Standard deviation of a normal distribution is the interquartile range divided by this.
_IQR_PER_SIGMA = 1.349


@dataclass(frozen=True, slots=True)
class NearestDistances:
    """For each query row, the distances to its nearest and second-nearest reference rows."""

    nearest: np.ndarray
    second_nearest: np.ndarray


@dataclass(frozen=True, slots=True)
class EncodedColumns:
    """A table as float columns (numbers and dates, in robust units) and integer category codes."""

    numbers: np.ndarray
    categories: np.ndarray
    number_scales: np.ndarray

    @property
    def n_columns(self) -> int:
        return self.numbers.shape[1] + self.categories.shape[1]


def encode_tables(
    reference: pd.DataFrame, others: list[pd.DataFrame], columns: list[str]
) -> list[EncodedColumns]:
    """Encode the reference table and the others on one shared scale and one set of codes.

    The scale of each number column comes from the reference table, so every table is
    measured with the same ruler. Returns the reference first, then the others in order.
    """
    tables = [reference, *others]
    number_columns = [name for name in columns if _is_measured_as_number(reference[name])]
    category_columns = [name for name in columns if name not in number_columns]
    numbers = [_number_matrix(table, number_columns) for table in tables]
    scales = _robust_scales(numbers[0])
    filled_numbers = [_fill_blanks(matrix, numbers[0]) for matrix in numbers]
    category_codes = _category_codes(tables, category_columns)
    return [
        EncodedColumns(filled, codes, scales)
        for filled, codes in zip(filled_numbers, category_codes, strict=True)
    ]


def nearest_distances(queries: EncodedColumns, reference: EncodedColumns) -> NearestDistances:
    """Find each query row's two nearest reference rows by Gower-like distance."""
    reference_rows = reference.numbers.shape[0] or reference.categories.shape[0]
    query_rows = queries.numbers.shape[0] or queries.categories.shape[0]
    chunk_rows = max(1, _CHUNK_CELLS // max(reference_rows, 1))
    nearest = np.empty(query_rows)
    second_nearest = np.full(query_rows, np.inf)
    for start in range(0, query_rows, chunk_rows):
        stop = min(start + chunk_rows, query_rows)
        distances = _distance_block(queries, reference, start, stop)
        smallest = _two_smallest(distances)
        nearest[start:stop] = smallest[:, 0]
        second_nearest[start:stop] = smallest[:, 1]
    return NearestDistances(nearest=nearest, second_nearest=second_nearest)


def _distance_block(
    queries: EncodedColumns, reference: EncodedColumns, start: int, stop: int
) -> np.ndarray:
    block_rows = stop - start
    reference_rows = reference.numbers.shape[0]
    total = np.zeros((block_rows, reference_rows), dtype=np.float32)
    for column in range(queries.numbers.shape[1]):
        gap = np.abs(queries.numbers[start:stop, column, None] - reference.numbers[None, :, column])
        total += np.minimum(gap / (FULL_MISMATCH_GAP * queries.number_scales[column]), 1.0)
    for column in range(queries.categories.shape[1]):
        differs = (
            queries.categories[start:stop, column, None] != reference.categories[None, :, column]
        )
        total += differs
    return total / queries.n_columns


def _two_smallest(distances: np.ndarray) -> np.ndarray:
    if distances.shape[1] == 1:
        return np.hstack([distances, np.full_like(distances, np.inf)])
    pair = np.partition(distances, 1, axis=1)[:, :2]
    return np.sort(pair, axis=1)


def _is_measured_as_number(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return False
    return pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values)


def _number_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = [_as_floats(table[name]) for name in columns]
    if not parts:
        return np.empty((len(table), 0), dtype=np.float64)
    return np.column_stack(parts)


def _as_floats(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(values):
        dates = pd.to_datetime(values)
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
        nanoseconds = dates.astype("datetime64[ns]").to_numpy().astype(np.int64)
        floats = nanoseconds.astype(np.float64)
        floats[dates.isna().to_numpy()] = np.nan
        return floats
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def _robust_scales(reference_numbers: np.ndarray) -> np.ndarray:
    scales = np.ones(reference_numbers.shape[1])
    for column in range(reference_numbers.shape[1]):
        values = reference_numbers[:, column]
        values = values[~np.isnan(values)]
        scales[column] = _scale_of(values)
    return scales


def _scale_of(values: np.ndarray) -> float:
    if len(values) < 2:
        return 1.0
    q1, q3 = np.percentile(values, [25, 75])
    spread = (q3 - q1) / _IQR_PER_SIGMA
    if spread > 0:
        return float(spread)
    # More than half of the values are equal, so the interquartile range is zero.
    spread = float(np.std(values))
    return spread if spread > 0 else 1.0


def _fill_blanks(matrix: np.ndarray, reference_numbers: np.ndarray) -> np.ndarray:
    """Replace blank numbers with a value far from every real one, so blanks match only blanks."""
    if matrix.shape[1] == 0:
        return matrix
    filled = matrix.copy()
    for column in range(matrix.shape[1]):
        blank = np.isnan(filled[:, column])
        filled[blank, column] = _blank_marker(reference_numbers[:, column])
    return filled


def _blank_marker(reference_values: np.ndarray) -> float:
    observed = reference_values[~np.isnan(reference_values)]
    if len(observed) == 0:
        return 0.0
    return float(observed.max() + 1e6 * (np.ptp(observed) + 1.0))


def _category_codes(tables: list[pd.DataFrame], columns: list[str]) -> list[np.ndarray]:
    if not columns:
        return [np.empty((len(table), 0), dtype=np.int32) for table in tables]
    sizes = [len(table) for table in tables]
    codes = []
    for name in columns:
        joined = pd.concat([table[name] for table in tables], ignore_index=True)
        codes.append(pd.factorize(joined.astype("string").fillna("<blank>"))[0].astype(np.int32))
    stacked = np.column_stack(codes)
    boundaries = np.cumsum(sizes)[:-1]
    return np.split(stacked, boundaries)
