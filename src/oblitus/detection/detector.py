"""Scan a table and report, per column, what kind of data it holds and how sensitive it is."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from oblitus.detection.calibration import calibrated_confidence
from oblitus.detection.column_names import (
    category_from_name,
    has_negative_hint,
    is_person_word_column,
    names_a_measure,
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
# Unique numbers of at least this many digits, nearly all of one length, are codes.
_MIN_CODE_DIGITS = 6
_SAME_LENGTH_SHARE = 0.95
# Random codes: nearly unique, and spread evenly within each digit length. A few repeats
# are allowed, because codes are often entered twice. A digit length needs enough values to
# test, and the tested lengths must hold nearly all the column.
_CODE_UNIQUE_SHARE = 0.85
_MIN_ROWS_PER_LENGTH = 30
_TESTED_SHARE = 0.75
_EVENNESS_BINS = 10
_EVENNESS_P_VALUE = 0.001
# Epoch times also have one length, and are times, not codes.
_EPOCH_SECONDS_LOW = 1_000_000_000
_EPOCH_SECONDS_HIGH = 2_100_000_000
_EPOCH_SCALES = (1, 1_000, 1_000_000)

# A name that points at a category is only a guess, so it is always put up for review.
_NAME_ONLY_CONFIDENCE = 0.55
_NAME_CONTRADICTED_CONFIDENCE = 0.4
_NAME_CANNOT_FIT_CONFIDENCE = 0.3
_NAME_AND_SHAPE_CONFIDENCE = 0.7
_GENDER_LABELS_CONFIDENCE = 0.7
_UNIQUENESS_ONLY_CONFIDENCE = 0.5
_BIRTH_DATE_CONFIDENCE = 0.5
_REPORTING_PERIOD_CONFIDENCE = 0.85
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
    measure = names_a_measure(column)
    return _from_values_alone(kind, present, sample, candidates, measure)


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
    kind: ColumnKind,
    present: pd.Series,
    sample: pd.Series,
    candidates: list[Candidate],
    named_as_measure: bool,
) -> tuple[DataCategory, float, str]:
    if kind is ColumnKind.DATETIME and _looks_like_birth_dates(sample):
        return (
            DataCategory.DATE_OF_BIRTH,
            _BIRTH_DATE_CONFIDENCE,
            "dates lie in the past and spread over decades, like birth dates",
        )
    if kind is ColumnKind.DATETIME and _looks_like_reporting_periods(sample):
        return (
            DataCategory.REPORTING_PERIOD,
            _REPORTING_PERIOD_CONFIDENCE,
            "every date is the first of a month at midnight, like a reporting period",
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

    if _looks_like_identifier(kind, present, named_as_measure):
        return (
            DataCategory.RECORD_ID,
            _UNIQUENESS_ONLY_CONFIDENCE,
            "nearly every row has its own value, so the column can single out a record",
        )
    if kind is ColumnKind.DATETIME:
        # A plain date is still a quasi-identifier, so "no sign of personal data" would mislead.
        evidence = "an event date: with a city or an age it can single people out"
        return DataCategory.GENERIC_DATETIME, 1.0, evidence
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


def _looks_like_reporting_periods(sample: pd.Series) -> bool:
    """Monthly, quarterly or yearly periods: every date is the first of a month at midnight."""
    dates = as_datetime(sample).dropna()
    if dates.nunique() < 2:
        return False
    starts_a_month = (dates.dt.day == 1) & (dates == dates.dt.normalize())
    return bool(starts_a_month.all())


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


def _looks_like_identifier(kind: ColumnKind, present: pd.Series, named_as_measure: bool) -> bool:
    if kind not in (ColumnKind.INTEGER, ColumnKind.TEXT) or len(present) < _MIN_ROWS_FOR_KEYS:
        return False
    unique_share = present.nunique() / len(present)
    # An evenly spread measure, such as amounts drawn at random, looks like random codes; only
    # its name tells them apart.
    may_be_random_codes = kind is ColumnKind.INTEGER and not named_as_measure
    if may_be_random_codes and unique_share >= _CODE_UNIQUE_SHARE:
        return _looks_like_random_codes(present) or _looks_like_keys(present)
    if unique_share < _IDENTIFIER_UNIQUE_SHARE:
        return False
    if kind is ColumnKind.TEXT:
        # Long unique strings are usually prose, not keys.
        return bool(present.astype(str).str.len().max() <= 40)
    return _looks_like_keys(present)


def _looks_like_keys(integers: pd.Series) -> bool:
    smallest = int(integers.min())
    largest = int(integers.max())
    if smallest < 0:
        return False
    return largest - smallest < _KEY_RANGE_PER_ROW * len(integers) or _looks_like_codes(integers)


def _looks_like_codes(integers: pd.Series) -> bool:
    """Unique numbers that nearly all have the same number of digits, such as account numbers.

    Such codes are spread too thinly for the dense-range test, but rounding them still leaves
    a value that points to one row for anyone who knows the code.
    """
    if _looks_like_epoch_times(integers):
        return False
    digit_counts = integers.astype("int64").astype(str).str.len()
    typical_count = int(digit_counts.mode().min())
    same_length_share = float((digit_counts == typical_count).mean())
    return typical_count >= _MIN_CODE_DIGITS and same_length_share >= _SAME_LENGTH_SHARE


def _looks_like_random_codes(integers: pd.Series) -> bool:
    """Numbers handed out at random: spread evenly within each number of digits.

    A code drawn at random is as likely to be 3,141,592 as 8,271,828, so within each digit
    length its values fill the range they cover evenly, whatever the lengths, the sign or a few
    repeats. A few short values mixed in (placeholders, test rows) are not tested.
    Measurements such as amounts, ages or populations crowd around typical values instead.
    Rounding a code still leaves a value that points to one row for anyone who knows it.
    """
    if _looks_like_epoch_times(integers):
        return False
    # Codes are all positive or, rarely, all negative. Values on both sides of zero are
    # differences or changes.
    if integers.min() < 0 < integers.max():
        return False
    magnitudes = integers.astype("int64").abs()
    lengths = magnitudes.astype(str).str.len()
    tested = 0
    for length, members in magnitudes.groupby(lengths):
        if len(members) < _MIN_ROWS_PER_LENGTH or int(length) < 2:
            continue
        if not _fills_range_evenly(members.to_numpy()):
            return False
        tested += len(members)
    return tested >= _TESTED_SHARE * len(integers)


def _fills_range_evenly(values: np.ndarray) -> bool:
    """A chi-square test of the values against an even spread from their lowest to highest.

    The range is the one the values cover, not the whole digit length, because a sequence of
    codes often stops short of the next power of ten (1 to 90,000, say).
    """
    counts, _edges = np.histogram(values, bins=_EVENNESS_BINS)
    return bool(scipy_stats.chisquare(counts).pvalue >= _EVENNESS_P_VALUE)


def _looks_like_epoch_times(integers: pd.Series) -> bool:
    """Seconds, milliseconds or microseconds since 1970, between 2001 and about 2036."""
    smallest = int(integers.min())
    largest = int(integers.max())
    for scale in _EPOCH_SCALES:
        if _EPOCH_SECONDS_LOW * scale <= smallest and largest <= _EPOCH_SECONDS_HIGH * scale:
            return True
    return False


def _generic(kind: ColumnKind) -> DataCategory:
    if kind in NUMERIC_KINDS:
        return DataCategory.GENERIC_NUMERIC
    if kind is ColumnKind.DATETIME:
        return DataCategory.GENERIC_DATETIME
    if kind in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN):
        return DataCategory.GENERIC_CATEGORY
    return DataCategory.GENERIC_TEXT
