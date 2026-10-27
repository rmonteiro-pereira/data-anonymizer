"""Decide the storage kind of a column from its dtype and values."""

import warnings

import pandas as pd

from oblitus.taxonomy import ColumnKind, DataCategory

# A text column with at most this many distinct values, or where distinct values
# are under this share of the rows, is treated as a set of categories.
_MAX_CATEGORIES = 50
_MAX_CATEGORY_SHARE = 0.5

# Share of text values that must parse as dates before the column counts as dates.
_DATE_PARSE_SHARE = 0.9


_TEXTUAL = frozenset({ColumnKind.TEXT, ColumnKind.CATEGORY})
# Documents, names and addresses are text, or digits that integer storage can hold.
_IDENTIFYING = _TEXTUAL | {ColumnKind.INTEGER}
_AMOUNTS = frozenset({ColumnKind.INTEGER, ColumnKind.FLOAT, ColumnKind.TEXT})
_KINDS_THAT_CAN_HOLD: dict[DataCategory, frozenset[ColumnKind]] = {
    DataCategory.AGE: frozenset({ColumnKind.INTEGER, ColumnKind.FLOAT}),
    DataCategory.GENDER: _IDENTIFYING | {ColumnKind.BOOLEAN},
    DataCategory.GEO_COORDINATE: frozenset({ColumnKind.FLOAT, ColumnKind.INTEGER, ColumnKind.TEXT}),
    DataCategory.DATE_OF_BIRTH: _IDENTIFYING | {ColumnKind.DATETIME},
    DataCategory.INCOME: _AMOUNTS,
    DataCategory.BUSINESS_REVENUE: _AMOUNTS,
    DataCategory.RENT: _AMOUNTS,
    DataCategory.FLOOR_AREA: _AMOUNTS,
    DataCategory.HEALTH: _IDENTIFYING | {ColumnKind.BOOLEAN},
    DataCategory.SPECIAL_CATEGORY: _IDENTIFYING | {ColumnKind.BOOLEAN},
}


def can_hold(kind: ColumnKind, category: DataCategory) -> bool:
    """True when a column stored as this kind could hold values of this category.

    A boolean or float column cannot hold an email, and a float cannot hold a gender, so a
    column name that suggests either is a coincidence.
    """
    return kind in _KINDS_THAT_CAN_HOLD.get(category, _IDENTIFYING)


def infer_kind(values: pd.Series) -> ColumnKind:
    """Return the kind of column these values form."""
    if pd.api.types.is_bool_dtype(values):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_integer_dtype(values):
        return ColumnKind.INTEGER
    if pd.api.types.is_float_dtype(values):
        return _float_or_integer(values)
    if pd.api.types.is_datetime64_any_dtype(values):
        return ColumnKind.DATETIME

    present = values.dropna()
    if present.empty:
        return ColumnKind.TEXT
    if _looks_like_dates(present):
        return ColumnKind.DATETIME
    if _looks_like_categories(present):
        return ColumnKind.CATEGORY
    return ColumnKind.TEXT


def _float_or_integer(values: pd.Series) -> ColumnKind:
    # pandas stores an integer column with gaps as float, so whole numbers stay integers.
    present = values.dropna()
    if not present.empty and (present % 1 == 0).all():
        return ColumnKind.INTEGER
    return ColumnKind.FLOAT


def _looks_like_dates(present: pd.Series) -> bool:
    sample = present.astype(str).head(200)
    if sample.str.fullmatch(r"\d+").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= _DATE_PARSE_SHARE


def _looks_like_categories(present: pd.Series) -> bool:
    distinct = present.astype(str).nunique()
    if distinct <= _MAX_CATEGORIES:
        return True
    return distinct / len(present) < _MAX_CATEGORY_SHARE


def as_nanoseconds(values: pd.Series) -> pd.Series:
    """Return dates as int64 nanoseconds since the epoch, whatever their stored resolution."""
    dates = as_datetime(values)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(None)
    return dates.astype("datetime64[ns]").astype("int64")


def as_datetime(values: pd.Series) -> pd.Series:
    """Convert a column the detector classed as dates into pandas datetimes."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(values, errors="coerce", format="mixed")
