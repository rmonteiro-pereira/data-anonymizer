"""Scan a table and report, per column, what kind of data it holds and how sensitive it is."""

from dataclasses import dataclass

import pandas as pd

from data_anonymizer.detection.column_names import category_from_name, has_negative_hint
from data_anonymizer.detection.kinds import as_datetime, infer_kind
from data_anonymizer.detection.patterns import RECOGNIZERS, TEXT_RECOGNIZERS
from data_anonymizer.detection.value_evidence import decide_from_values, value_candidates
from data_anonymizer.taxonomy import (
    NUMERIC_KINDS,
    ColumnKind,
    DataCategory,
    Sensitivity,
    sensitivity_of,
)

# Free text is flagged when at least this share of values contains personal data.
_EMBEDDED_MATCH_SHARE = 0.1

# Columns with nearly one distinct value per row behave like record identifiers.
_IDENTIFIER_UNIQUE_SHARE = 0.98

_NAME_ONLY_CONFIDENCE = 0.7
_NAME_CONTRADICTED_CONFIDENCE = 0.4
_UNIQUENESS_ONLY_CONFIDENCE = 0.5
_BIRTH_DATE_CONFIDENCE = 0.5
_FREE_TEXT_MIN_WORDS = 2

# Birth dates lie in the past, within a human lifetime, and spread over decades.
_OLDEST_AGE_YEARS = 120
_BIRTH_YEAR_SPREAD = 8

_GENDER_VALUES = {"m", "f", "male", "female", "masculino", "feminino", "man", "woman", "other"}

# Postal codes are left out because the recognizer only knows the Brazilian format,
# so a US zip column would wrongly look contradicted by its own values.
_VALUE_CHECKED_CATEGORIES = {recognizer.category for recognizer in RECOGNIZERS} - {
    DataCategory.POSTAL_CODE
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


def scan(frame: pd.DataFrame, *, sample_size: int = 1000, seed: int = 0) -> list[ColumnFinding]:
    """Return one finding per column of the table, in column order."""
    return [
        scan_column(str(column), frame[column], sample_size=sample_size, seed=seed)
        for column in frame.columns
    ]


def scan_column(
    column: str, values: pd.Series, *, sample_size: int = 1000, seed: int = 0
) -> ColumnFinding:
    """Classify a single column from its name and a sample of its values."""
    kind = infer_kind(values)
    name_category = category_from_name(column)
    present = values.dropna()
    sample = _sample(present, sample_size, seed)

    category, confidence, evidence = _classify(
        kind, name_category, present, sample, ignore_bare_digits=has_negative_hint(column)
    )
    return ColumnFinding(
        column=column,
        kind=kind,
        category=category,
        sensitivity=sensitivity_of(category),
        confidence=round(confidence, 3),
        evidence=evidence,
    )


def _sample(present: pd.Series, sample_size: int, seed: int) -> pd.Series:
    if len(present) <= sample_size:
        return present
    return present.sample(sample_size, random_state=seed)


def _classify(
    kind: ColumnKind,
    name_category: DataCategory | None,
    present: pd.Series,
    sample: pd.Series,
    *,
    ignore_bare_digits: bool,
) -> tuple[DataCategory, float, str]:
    if sample.empty:
        return _from_name_or_generic(kind, name_category, "the column has no values")

    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY, ColumnKind.INTEGER):
        candidates = value_candidates(
            sample,
            name_category=name_category,
            integers=kind is ColumnKind.INTEGER,
            ignore_bare_digits=ignore_bare_digits,
        )
        by_value = decide_from_values(candidates)
        if by_value is not None:
            return by_value

    if name_category is not None:
        return _from_name(name_category)

    if kind is ColumnKind.DATETIME and _looks_like_birth_dates(sample):
        return (
            DataCategory.DATE_OF_BIRTH,
            _BIRTH_DATE_CONFIDENCE,
            "dates lie in the past and spread over decades, like birth dates",
        )

    if kind in (ColumnKind.TEXT, ColumnKind.CATEGORY):
        embedded = _classify_free_text(sample)
        if embedded is not None:
            return embedded

    if kind is ColumnKind.CATEGORY and _looks_like_gender(sample):
        return DataCategory.GENDER, _NAME_ONLY_CONFIDENCE, "values are gender labels"

    if _looks_like_identifier(kind, present):
        return (
            DataCategory.RECORD_ID,
            _UNIQUENESS_ONLY_CONFIDENCE,
            "nearly every row has its own value, so the column can single out a record",
        )
    return _generic(kind), 1.0, "no sign of personal data in the name or the values"


def _from_name(name_category: DataCategory) -> tuple[DataCategory, float, str]:
    if name_category in _VALUE_CHECKED_CATEGORIES:
        evidence = f"the column name suggests {name_category.value}, but the values do not match"
        return name_category, _NAME_CONTRADICTED_CONFIDENCE, evidence
    return name_category, _NAME_ONLY_CONFIDENCE, f"the column name suggests {name_category.value}"


def _from_name_or_generic(
    kind: ColumnKind, name_category: DataCategory | None, reason: str
) -> tuple[DataCategory, float, str]:
    if name_category is not None:
        return name_category, _NAME_ONLY_CONFIDENCE, f"{reason}; the name suggests it"
    return _generic(kind), 1.0, reason


def _classify_free_text(sample: pd.Series) -> tuple[DataCategory, float, str] | None:
    texts = sample.astype(str)
    # One identifier per cell is a column of identifiers, not prose that mentions one.
    if texts.map(lambda text: len(text.split())).mean() < _FREE_TEXT_MIN_WORDS:
        return None
    contains_any = texts.map(_contains_personal_data)
    share = float(contains_any.mean())
    if share < _EMBEDDED_MATCH_SHARE:
        return None
    evidence = f"{share:.0%} of sampled texts contain an email, document number or phone"
    return DataCategory.FREE_TEXT_WITH_PII, min(1.0, 0.5 + share), evidence


def _contains_personal_data(text: str) -> bool:
    return any(recognizer.find_spans(text) for recognizer in TEXT_RECOGNIZERS)


def _looks_like_birth_dates(sample: pd.Series) -> bool:
    dates = as_datetime(sample).dropna()
    if len(dates) < 20:
        return False
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    today = pd.Timestamp.today().normalize()
    oldest = today - pd.DateOffset(years=_OLDEST_AGE_YEARS)
    in_lifetime = bool(dates.between(oldest, today).all())
    no_time_of_day = bool((dates == dates.dt.normalize()).all())
    return in_lifetime and no_time_of_day and dates.dt.year.std() >= _BIRTH_YEAR_SPREAD


def _looks_like_gender(sample: pd.Series) -> bool:
    labels = set(sample.astype(str).str.strip().str.lower().unique())
    return len(labels) <= 4 and labels <= _GENDER_VALUES


def _looks_like_identifier(kind: ColumnKind, present: pd.Series) -> bool:
    if kind not in (ColumnKind.INTEGER, ColumnKind.TEXT) or len(present) < 20:
        return False
    unique_share = present.nunique() / len(present)
    if unique_share < _IDENTIFIER_UNIQUE_SHARE:
        return False
    if kind is ColumnKind.TEXT:
        # Long unique strings are usually prose, not keys.
        return bool(present.astype(str).str.len().max() <= 40)
    return True


def _generic(kind: ColumnKind) -> DataCategory:
    if kind in NUMERIC_KINDS:
        return DataCategory.GENERIC_NUMERIC
    if kind is ColumnKind.DATETIME:
        return DataCategory.GENERIC_DATETIME
    if kind in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN):
        return DataCategory.GENERIC_CATEGORY
    return DataCategory.GENERIC_TEXT
