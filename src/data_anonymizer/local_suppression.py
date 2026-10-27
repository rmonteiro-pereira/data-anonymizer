"""Local suppression: blank quasi-identifier cells until every group has at least k rows.

Column techniques make values coarser, but a rare combination (the only 90-year-old
in a small town) can still single a person out. This step finds the rows in groups
smaller than k and blanks their quasi-identifiers, most varied column first, until
the groups merge. It changes only the rows at risk.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SuppressionOutcome:
    """The table after suppression and how many cells were blanked."""

    data: pd.DataFrame
    suppressed_cells: int


def enforce_k_anonymity(
    frame: pd.DataFrame, quasi_identifiers: Sequence[str], k: int
) -> SuppressionOutcome:
    """Blank quasi-identifier cells in rows whose group is smaller than k.

    When even fully blanked rows number fewer than k, they still form one group
    below k; the privacy report shows it, and dropping those rows is left to the caller.
    """
    columns = [column for column in quasi_identifiers if column in frame.columns]
    result = frame.copy()
    if not columns or k <= 1:
        return SuppressionOutcome(result, 0)

    suppressed = 0
    most_varied_first = sorted(columns, key=lambda column: -result[column].nunique())
    for column in most_varied_first:
        at_risk = _rows_in_small_groups(result, columns, k)
        if not at_risk.any():
            break
        to_blank = at_risk & result[column].notna()
        suppressed += int(to_blank.sum())
        result[column] = result[column].astype(object).where(~to_blank, None)
    return SuppressionOutcome(result, suppressed)


def _rows_in_small_groups(frame: pd.DataFrame, columns: list[str], k: int) -> pd.Series:
    keys = frame[columns].astype(str).assign(_row=1)
    sizes = keys.groupby(columns, dropna=False)["_row"].transform("size")
    return sizes < k
