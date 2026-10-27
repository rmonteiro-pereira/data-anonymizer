"""Infer a generation schema from a sample table and the detector's findings.

The inferred schema never copies a real identifier. Columns the detector flags as
direct identifiers get a fake-value category instead of their observed values.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from oblitus.detection import ColumnFinding, scan
from oblitus.detection.kinds import as_datetime
from oblitus.surrogates import SURROGATE_CATEGORIES
from oblitus.synthesis.rare_values import (
    MIN_RECORDS_STATISTICAL,
    merge_rare_categories,
    pull_in_extremes,
)
from oblitus.synthesis.schema import ColumnSchema, DatasetSchema, Distribution
from oblitus.taxonomy import NUMERIC_KINDS, ColumnKind, DataCategory, Sensitivity

# Finer tail points are added down to this many observations beyond the outermost point.
_MIN_TAIL_OBSERVATIONS = 10


def infer_schema(
    frame: pd.DataFrame,
    findings: Sequence[ColumnFinding] | None = None,
    *,
    min_category_count: int = MIN_RECORDS_STATISTICAL,
) -> DatasetSchema:
    """Return a schema that generates tables shaped like `frame`.

    Categories seen fewer than `min_category_count` times are merged into `OTHER`, because a
    rare value can identify the one person who has it. Fewer than 5 is always merged. The
    same count is the number of records that must lie beyond each numeric quantile point, and
    the minimum and maximum are the values with that many records beyond them, so one extreme
    record cannot shape the learned tail.
    """
    min_records = max(min_category_count, MIN_RECORDS_STATISTICAL)
    findings_by_column = {finding.column: finding for finding in findings or scan(frame)}
    columns = []
    for column in frame.columns:
        name = str(column)
        finding = findings_by_column.get(name)
        if finding is None:
            finding = scan(frame[[column]])[0]
        columns.append(_infer_column(name, frame[column], finding, min_records))
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
        return _numeric_column(finding, present, common, min_category_count)
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
    finding: ColumnFinding, present: pd.Series, common: dict[str, Any], min_records: int
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
    # Quantile points follow skewed, heavy-tailed and zero-inflated shapes that a normal
    # distribution with the same mean and std would distort.
    sorted_numbers = np.sort(numbers.to_numpy(dtype=float))
    probabilities = _quantile_probabilities(len(numbers), min_records)
    points = np.quantile(sorted_numbers, probabilities)
    pulled_in = pull_in_extremes(sorted_numbers, min_records)
    points[0] = pulled_in[0]
    points[-1] = pulled_in[-1]
    return ColumnSchema(
        kind=finding.kind,
        minimum=float(pulled_in[0]),
        maximum=float(pulled_in[-1]),
        distribution=Distribution.QUANTILES,
        quantiles={
            float(probability): float(point)
            for probability, point in zip(probabilities, points, strict=True)
        },
        **common,
    )


def _quantile_probabilities(count: int, min_records: int) -> list[float]:
    """Return the probabilities whose quantiles describe a numeric column.

    These are the deciles, the 5% and 95% points, and finer points in each tail (2%, 1%,
    0.5%, 0.2%, ...) while enough observations remain beyond them. Without the tail points, a
    single extreme value would be stretched in a straight line over the whole outer 5%. A
    point needs at least `min_records` observations beyond it, so no record controls it; the
    0 and 1 points are the schema's minimum and maximum, which are capped the same way.
    """
    tail = [0.05]
    scale = 0.01
    while scale * count >= _MIN_TAIL_OBSERVATIONS:
        tail.extend(scale * mantissa for mantissa in (2, 1, 0.5))
        scale /= 10
    lower = sorted(tail)
    upper = [1 - probability for probability in reversed(lower)]
    deciles = [step / 10 for step in range(1, 10)]
    candidates = {round(value, 10) for value in (*lower, *deciles, *upper)}
    last_position = count - 1
    supported = {
        probability
        for probability in candidates
        if min_records <= probability * last_position <= last_position - min_records
    }
    return sorted({0.0, 1.0, *supported})


def _category_column(
    present: pd.Series, min_category_count: int, common: dict[str, Any]
) -> ColumnSchema:
    merged = merge_rare_categories(present.astype(str).to_numpy(dtype=object), min_category_count)
    counts = pd.Series(merged).value_counts()
    choices = [str(value) for value in counts.index]
    weights = [float(count) for count in counts.to_numpy()]
    return ColumnSchema(kind=ColumnKind.CATEGORY, choices=choices, weights=weights, **common)
