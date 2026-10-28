"""Measure how far an anonymized or synthetic table drifts from the original statistically."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from oblitus.detection.kinds import as_nanoseconds

_NANOSECONDS_PER_DAY = 86_400 * 10**9


@dataclass(frozen=True, slots=True)
class ColumnUtility:
    """How one column's distribution changed.

    For numeric columns: relative change of mean and standard deviation, and the
    Kolmogorov-Smirnov statistic (0 means identical distributions, 1 disjoint ones).
    For date columns the mean change is the shift of the mean in standard deviations of the
    original dates, since a ratio to days since 1970 would read every shift as tiny.
    For categorical columns: total variation distance between category shares
    (0 identical, 1 disjoint).
    The statistics compare the cells that are filled in each table. `added_blank_share` is
    the share of rows blank in the release but not in the original, because local suppression
    blanks rare cells rather than random ones, and the filled cells alone can look unchanged.
    """

    column: str
    comparison: str
    mean_change: float | None = None
    std_change: float | None = None
    ks_statistic: float | None = None
    total_variation: float | None = None
    added_blank_share: float | None = None


@dataclass(frozen=True, slots=True)
class UtilityReport:
    """Per-column drift plus the largest change in any pairwise numeric correlation.

    The correlation change covers only columns numeric in both tables: a column binned into
    labels drops out, so its lost correlations do not show here.
    """

    columns: list[ColumnUtility]
    max_correlation_change: float | None

    def for_column(self, column: str) -> ColumnUtility | None:
        """Return the drift measured for one column."""
        for item in self.columns:
            if item.column == column:
                return item
        return None


def compare_tables(original: pd.DataFrame, changed: pd.DataFrame) -> UtilityReport:
    """Compare every column the two tables share."""
    shared = [column for column in original.columns if column in changed.columns]
    columns = [_compare_column(column, original[column], changed[column]) for column in shared]
    numeric_shared = [
        column
        for column in shared
        if _is_numeric(original[column]) and _is_numeric(changed[column])
    ]
    return UtilityReport(
        columns=columns,
        max_correlation_change=_max_correlation_change(original, changed, numeric_shared),
    )


def _is_numeric(values: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)


def _compare_column(column: str, before: pd.Series, after: pd.Series) -> ColumnUtility:
    measured = _compare_filled_cells(column, before, after)
    blanked = after.isna().to_numpy() & ~before.isna().to_numpy()
    return replace(measured, added_blank_share=float(blanked.mean()) if len(blanked) else None)


def _compare_filled_cells(column: str, before: pd.Series, after: pd.Series) -> ColumnUtility:
    if _is_numeric(before) and _is_numeric(after):
        return _compare_numeric(column, before.dropna(), after.dropna())
    if pd.api.types.is_datetime64_any_dtype(before) and pd.api.types.is_datetime64_any_dtype(after):
        before_days = as_nanoseconds(before.dropna()) / _NANOSECONDS_PER_DAY
        after_days = as_nanoseconds(after.dropna()) / _NANOSECONDS_PER_DAY
        compared = _compare_numeric(column, before_days, after_days)
        return replace(compared, mean_change=_mean_shift_in_std(before_days, after_days))
    return ColumnUtility(
        column=column,
        comparison="categorical",
        total_variation=_total_variation(before, after),
    )


def _mean_shift_in_std(before: pd.Series, after: pd.Series) -> float | None:
    if before.empty or after.empty:
        return None
    spread = float(before.std())
    shift = abs(float(after.mean()) - float(before.mean()))
    return shift if not spread else shift / spread


def _compare_numeric(column: str, before: pd.Series, after: pd.Series) -> ColumnUtility:
    if before.empty or after.empty:
        return ColumnUtility(column=column, comparison="numeric")
    return ColumnUtility(
        column=column,
        comparison="numeric",
        mean_change=relative_change(float(before.mean()), float(after.mean())),
        std_change=relative_change(float(before.std()), float(after.std())),
        # Only the statistic is reported, so the exact p-value, which warns on large ties,
        # is not needed.
        ks_statistic=float(stats.ks_2samp(before, after, method="asymp").statistic),
    )


def relative_change(before: float, after: float) -> float:
    """Return |after - before| / |before|, or |after| when before is zero."""
    if before == 0:
        return abs(after)
    return abs(after - before) / abs(before)


def _total_variation(before: pd.Series, after: pd.Series) -> float:
    before_shares = before.astype(str).value_counts(normalize=True)
    after_shares = after.astype(str).value_counts(normalize=True)
    aligned = pd.concat([before_shares, after_shares], axis=1).fillna(0.0)
    return float(0.5 * np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1]).sum())


def _max_correlation_change(
    original: pd.DataFrame, changed: pd.DataFrame, columns: list[str]
) -> float | None:
    if len(columns) < 2:
        return None
    before = original[columns].astype(float).corr().to_numpy()
    after = changed[columns].astype(float).corr().to_numpy()
    difference = np.abs(before - after)
    if np.all(np.isnan(difference)):
        return None
    return float(np.nanmax(difference))
