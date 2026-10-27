"""Infer a generation schema from a sample table and the detector's findings.

The inferred schema never copies a real identifier. Columns the detector flags as
direct identifiers get a fake-value category instead of their observed values.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from data_anonymizer.detection import ColumnFinding, scan
from data_anonymizer.detection.kinds import as_datetime
from data_anonymizer.surrogates import SURROGATE_CATEGORIES
from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema
from data_anonymizer.taxonomy import NUMERIC_KINDS, ColumnKind, DataCategory, Sensitivity

_RARE_REPLACEMENT = "OTHER"


def infer_schema(
    frame: pd.DataFrame,
    findings: Sequence[ColumnFinding] | None = None,
    *,
    min_category_count: int = 1,
) -> DatasetSchema:
    """Return a schema that generates tables shaped like `frame`.

    Categories seen fewer than `min_category_count` times are left out of the
    choices, because a rare value can identify the one person who has it.
    """
    findings_by_column = {finding.column: finding for finding in findings or scan(frame)}
    columns = []
    for column in frame.columns:
        name = str(column)
        finding = findings_by_column.get(name)
        if finding is None:
            finding = scan(frame[[column]])[0]
        columns.append(_infer_column(name, frame[column], finding, min_category_count))
    return DatasetSchema(columns=columns)


def _infer_column(
    name: str, values: pd.Series, finding: ColumnFinding, min_category_count: int
) -> ColumnSchema:
    present = values.dropna()
    null_fraction = 1 - len(present) / len(values) if len(values) else 0.0
    common: dict[str, Any] = {"name": name, "null_fraction": round(null_fraction, 4)}
    fake_category = _fake_category(finding)

    if fake_category is not None:
        can_be_unique = finding.kind not in (ColumnKind.BOOLEAN, ColumnKind.DATETIME)
        unique = can_be_unique and len(present) > 1 and present.nunique() == len(present)
        return ColumnSchema(kind=finding.kind, category=fake_category, unique=unique, **common)
    if finding.kind in NUMERIC_KINDS:
        return _numeric_column(finding, present, common)
    if finding.kind is ColumnKind.BOOLEAN:
        true_share = float(present.astype(bool).mean()) if len(present) else 0.5
        return ColumnSchema(kind=finding.kind, true_share=true_share, **common)
    if finding.kind is ColumnKind.DATETIME:
        dates = as_datetime(present).dropna()
        return ColumnSchema(kind=finding.kind, start=dates.min(), end=dates.max(), **common)
    if finding.kind is ColumnKind.CATEGORY:
        return _category_column(present, min_category_count, common)
    return ColumnSchema(kind=ColumnKind.TEXT, category=DataCategory.GENERIC_TEXT, **common)


def _fake_category(finding: ColumnFinding) -> DataCategory | None:
    if finding.kind in NUMERIC_KINDS:
        # Numeric identifiers are regenerated as unique numbers in the observed range.
        return None
    if finding.category in SURROGATE_CATEGORIES and finding.kind is ColumnKind.TEXT:
        # Varied text such as postal codes or URLs is regenerated in the same format.
        return finding.category
    if finding.sensitivity is not Sensitivity.DIRECT_IDENTIFIER:
        return None
    if finding.category in SURROGATE_CATEGORIES:
        return finding.category
    # Identifiers with no fake generator, such as free text with personal data,
    # become neutral sentences.
    return DataCategory.GENERIC_TEXT


def _numeric_column(
    finding: ColumnFinding, present: pd.Series, common: dict[str, Any]
) -> ColumnSchema:
    numbers = pd.to_numeric(present, errors="coerce").dropna()
    if numbers.empty:
        return ColumnSchema(kind=finding.kind, minimum=0, maximum=1, **common)
    if finding.sensitivity is Sensitivity.DIRECT_IDENTIFIER:
        unique = numbers.nunique() == len(numbers)
        # Unique identifiers get no maximum, so a synthetic table larger than the
        # sample still finds enough distinct numbers.
        return ColumnSchema(
            kind=finding.kind,
            minimum=float(numbers.min()),
            maximum=None if unique else float(numbers.max()),
            unique=unique,
            **common,
        )
    return ColumnSchema(
        kind=finding.kind,
        minimum=float(numbers.min()),
        maximum=float(numbers.max()),
        mean=float(numbers.mean()),
        std=float(numbers.std(ddof=0)),
        **common,
    )


def _category_column(
    present: pd.Series, min_category_count: int, common: dict[str, Any]
) -> ColumnSchema:
    counts = present.astype(str).value_counts()
    kept = counts[counts >= min_category_count]
    rare_total = int(counts[counts < min_category_count].sum())
    choices = [str(value) for value in kept.index]
    weights = [float(count) for count in kept.to_numpy()]
    if rare_total:
        choices.append(_RARE_REPLACEMENT)
        weights.append(float(rare_total))
    return ColumnSchema(kind=ColumnKind.CATEGORY, choices=choices, weights=weights, **common)
