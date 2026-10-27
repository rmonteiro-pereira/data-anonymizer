"""Scan a table and report, per column, what kind of data it holds and how sensitive it is."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oblitus.detection.calibration import calibrated_confidence
from oblitus.detection.column_names import (
    category_from_name,
    has_negative_hint,
    is_person_word_column,
)
from oblitus.detection.free_text import classify_free_text
from oblitus.detection.kinds import as_datetime, can_hold, infer_kind
from oblitus.detection.missing_values import blank_out_placeholders
from oblitus.detection.patterns import RECOGNIZERS
from oblitus.detection.person_names import (
    looks_like_known_names,
    looks_like_labelled_names,
    looks_like_person_names,
)
from oblitus.detection.value_evidence import (
    Candidate,
    decide_from_values,
    decide_from_weak_match,
    value_candidates,
)
from oblitus.errors import TableFormatError
from oblitus.taxonomy import (
    NUMERIC_KINDS,
    ColumnKind,
    DataCategory,
    Sensitivity,
    sensitivity_of,
)

# Columns with nearly one distinct value per row behave like record identifiers.
_IDENTIFIER_UNIQUE_SHARE = 0.98
_MIN_ROWS_FOR_KEYS = 20
# Keys are handed out densely, so their range stays close to the number of rows. Epoch
# timestamps, amounts in cents and populations spread over a range far wider than that.
_KEY_RANGE_PER_ROW = 10

# A name that points at a category is only a guess, so it is always put up for review.
_NAME_ONLY_CONFIDENCE = 0.55
_NAME_CONTRADICTED_CONFIDENCE = 0.4
_NAME_CANNOT_FIT_CONFIDENCE = 0.3
_NAME_AND_SHAPE_CONFIDENCE = 0.7
_GENDER_LABELS_CONFIDENCE = 0.7
_UNIQUENESS_ONLY_CONFIDENCE = 0.5
_BIRTH_DATE_CONFIDENCE = 0.5
_KNOWN_NAMES_CONFIDENCE = 0.65

# Birth dates lie in the past, within a human lifetime, and spread over decades. The ages
# of a group of people are mostly adult, with a typical age below a hundred years.
_MIN_DATES = 20
_OLDEST_AGE_YEARS = 120
_MIN_BIRTH_YEAR_SPREAD = 8
_MAX_BIRTH_YEAR_SPREAD = 30
_ADULT_AGE_YEARS = 18
_ADULT_SHARE = 0.8
_TYPICAL_AGE_LIMIT_YEARS = 100
_DAYS_PER_YEAR = 365.25

_GENDER_VALUES = {"m", "f", "male", "female", "masculino", "feminino", "man", "woman", "other"}

_NESTED_TYPES = (list, tuple, dict, set, frozenset, np.ndarray)

# Postal codes and cities are left out because the recognizers only know Brazil, so a US zip
# or a Lisbon column would wrongly look contradicted by its own values.
_VALUE_CHECKED_CATEGORIES = {recognizer.category for recognizer in RECOGNIZERS} - {
    DataCategory.POSTAL_CODE,
    DataCategory.CITY,
}


@dataclass(frozen=True, slots=True)
class ColumnFinding:
    """What the detector concluded about one column, with the evidence behind it."""

    column: str
    kind: ColumnKind
    category: DataCategory
    sensitivity: Sensitivity
    confidence: float
    evidence: str


def scan(
    frame: pd.DataFrame, *, sample_size: int = 1000, seed: int = 0, calibrated: bool = True
) -> list[ColumnFinding]:
    """Return one finding per column of the table, in column order.

    The confidence is calibrated on a labelled benchmark (see `calibration.py`); pass
    `calibrated=False` for the raw rule-based score.

    Raises `ValueError` when `sample_size` is below 1 and `TableFormatError` when two
    columns share a name.
    """
    _require_valid_sample_size(sample_size)
    names = [str(column) for column in frame.columns]
    _require_unique_names(names)
    return [
        scan_column(
            name, frame.iloc[:, position], sample_size=sample_size, seed=seed, calibrated=calibrated
        )
        for position, name in enumerate(names)
    ]


def scan_column(
    column: str,
    values: pd.Series,
    *,
    sample_size: int = 1000,
    seed: int = 0,
    calibrated: bool = True,
) -> ColumnFinding:
    """Classify a single column from its name and a sample of its values."""
    _require_valid_sample_size(sample_size)
    values = blank_out_placeholders(_nested_values_as_text(values))
    kind = infer_kind(values)
    present = values.dropna()
    sample = _sample(present, sample_size, seed)

    category, confidence, evidence = _classify(column, kind, present, sample)
    if calibrated:
        found_personal_data = sensitivity_of(category) is not Sensitivity.NON_SENSITIVE
        confidence = calibrated_confidence(confidence, found_personal_data=found_personal_data)
    return ColumnFinding(
        column=column,
        kind=kind,
        category=category,
        sensitivity=sensitivity_of(category),
        confidence=round(confidence, 3),
        evidence=evidence,
    )


def _require_valid_sample_size(sample_size: int) -> None:
    if sample_size < 1:
        raise ValueError(f"sample_size must be at least 1, got {sample_size}")


def _require_unique_names(names: list[str]) -> None:
    repeated = [name for name, count in pd.Series(names).value_counts().items() if count > 1]
    if repeated:
        listed = ", ".join(repr(name) for name in repeated)
        raise TableFormatError(f"duplicate column names: {listed}; rename them before scanning")


def _nested_values_as_text(values: pd.Series) -> pd.Series:
    """Replace lists, dicts, sets and arrays by their text, since they cannot be counted."""
    if values.dtype != object:
        return values
    return values.map(lambda cell: str(cell) if isinstance(cell, _NESTED_TYPES) else cell)


def _sample(present: pd.Series, sample_size: int, seed: int) -> pd.Series:
    if len(present) <= sample_size:
        return present
    return present.sample(sample_size, random_state=seed)


def _classify(
    column: str, kind: ColumnKind, present: pd.Series, sample: pd.Series
) -> tuple[DataCategory, float, str]:
    name_category = category_from_name(column)
    if sample.empty:
        return _from_name_or_generic(kind, name_category, "the column has no values")

    candidates: list[Candidate] = []
    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY, ColumnKind.INTEGER):
        candidates = value_candidates(
            sample,
            name_category=name_category,
            integers=kind is ColumnKind.INTEGER,
            negative_hint=has_negative_hint(column),
        )
        by_value = decide_from_values(candidates)
        if by_value is not None:
            return by_value

    by_name = _from_name_and_values(column, kind, name_category, sample)
    if by_name is not None:
        return by_name
    return _from_values_alone(kind, present, sample, candidates)


def _from_name_and_values(
    column: str, kind: ColumnKind, name_category: DataCategory | None, sample: pd.Series
) -> tuple[DataCategory, float, str] | None:
    if _holds_person_names(column, kind, name_category, sample):
        evidence = "the column name and the values both look like person names"
        return DataCategory.PERSON_NAME, _NAME_AND_SHAPE_CONFIDENCE, evidence
    if name_category is not None:
        return _from_name(kind, name_category)
    return None


def _holds_person_names(
    column: str, kind: ColumnKind, name_category: DataCategory | None, sample: pd.Series
) -> bool:
    if kind not in (ColumnKind.TEXT, ColumnKind.CATEGORY):
        return False
    named_after_people = name_category is DataCategory.PERSON_NAME or (
        name_category is None and is_person_word_column(column)
    )
    return named_after_people and looks_like_person_names(sample)


def _from_values_alone(
    kind: ColumnKind, present: pd.Series, sample: pd.Series, candidates: list[Candidate]
) -> tuple[DataCategory, float, str]:
    if kind is ColumnKind.DATETIME and _looks_like_birth_dates(sample):
        return (
            DataCategory.DATE_OF_BIRTH,
            _BIRTH_DATE_CONFIDENCE,
            "dates lie in the past and spread over decades, like birth dates",
        )

    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY):
        embedded = classify_free_text(sample)
        if embedded is not None:
            return embedded

    if kind is ColumnKind.CATEGORY and _looks_like_gender(sample):
        return DataCategory.GENDER, _GENDER_LABELS_CONFIDENCE, "values are gender labels"

    weak_match = decide_from_weak_match(candidates)
    if weak_match is not None:
        return weak_match

    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY) and looks_like_known_names(sample):
        evidence = "most values are Brazilian first names followed by surnames"
        return DataCategory.PERSON_NAME, _KNOWN_NAMES_CONFIDENCE, evidence

    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY) and looks_like_labelled_names(sample):
        evidence = "most values are a Brazilian name with at most one other word"
        return DataCategory.PERSON_NAME, _KNOWN_NAMES_CONFIDENCE, evidence

    if _looks_like_identifier(kind, present):
        return (
            DataCategory.RECORD_ID,
            _UNIQUENESS_ONLY_CONFIDENCE,
            "nearly every row has its own value, so the column can single out a record",
        )
    return _generic(kind), 1.0, "no sign of personal data in the name or the values"


def _from_name(kind: ColumnKind, name_category: DataCategory) -> tuple[DataCategory, float, str]:
    suggestion = f"the column name suggests {name_category.value}"
    if not can_hold(kind, name_category):
        evidence = f"{suggestion}, but a {kind.value} column cannot hold it"
        return name_category, _NAME_CANNOT_FIT_CONFIDENCE, evidence
    if name_category in _VALUE_CHECKED_CATEGORIES:
        return (
            name_category,
            _NAME_CONTRADICTED_CONFIDENCE,
            f"{suggestion}, but the values do not match",
        )
    return name_category, _NAME_ONLY_CONFIDENCE, suggestion


def _from_name_or_generic(
    kind: ColumnKind, name_category: DataCategory | None, reason: str
) -> tuple[DataCategory, float, str]:
    if name_category is not None:
        return name_category, _NAME_ONLY_CONFIDENCE, f"{reason}; the name suggests it"
    return _generic(kind), 1.0, reason


def _looks_like_birth_dates(sample: pd.Series) -> bool:
    dates = as_datetime(sample).dropna()
    if len(dates) < _MIN_DATES:
        return False
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    today = pd.Timestamp.today().normalize()
    oldest = today - pd.DateOffset(years=_OLDEST_AGE_YEARS)
    in_lifetime = bool(dates.between(oldest, today).all())
    no_time_of_day = bool((dates == dates.dt.normalize()).all())
    if not (in_lifetime and no_time_of_day):
        return False
    return _ages_fit_a_group_of_people(dates, today)


def _ages_fit_a_group_of_people(dates: pd.Series, today: pd.Timestamp) -> bool:
    spread = dates.dt.year.std()
    if not _MIN_BIRTH_YEAR_SPREAD <= spread <= _MAX_BIRTH_YEAR_SPREAD:
        return False
    ages = (today - dates).dt.days / _DAYS_PER_YEAR
    mostly_adults = float((ages >= _ADULT_AGE_YEARS).mean()) >= _ADULT_SHARE
    typical_age = float(ages.astype(int).mode().min())
    return mostly_adults and typical_age < _TYPICAL_AGE_LIMIT_YEARS


def _looks_like_gender(sample: pd.Series) -> bool:
    labels = set(sample.astype(str).str.strip().str.lower().unique())
    return len(labels) <= 4 and labels <= _GENDER_VALUES


def _looks_like_identifier(kind: ColumnKind, present: pd.Series) -> bool:
    if kind not in (ColumnKind.INTEGER, ColumnKind.TEXT) or len(present) < _MIN_ROWS_FOR_KEYS:
        return False
    unique_share = present.nunique() / len(present)
    if unique_share < _IDENTIFIER_UNIQUE_SHARE:
        return False
    if kind is ColumnKind.TEXT:
        # Long unique strings are usually prose, not keys.
        return bool(present.astype(str).str.len().max() <= 40)
    return _looks_like_keys(present)


def _looks_like_keys(integers: pd.Series) -> bool:
    smallest = int(integers.min())
    largest = int(integers.max())
    return smallest >= 0 and largest - smallest < _KEY_RANGE_PER_ROW * len(integers)


def _generic(kind: ColumnKind) -> DataCategory:
    if kind in NUMERIC_KINDS:
        return DataCategory.GENERIC_NUMERIC
    if kind is ColumnKind.DATETIME:
        return DataCategory.GENERIC_DATETIME
    if kind in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN):
        return DataCategory.GENERIC_CATEGORY
    return DataCategory.GENERIC_TEXT
