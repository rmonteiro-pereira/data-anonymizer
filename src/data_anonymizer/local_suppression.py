"""Local suppression: blank quasi-identifier cells until every group has at least k rows.

Column techniques make values coarser, but a rare combination (the only 90-year-old
in a small town) can still single a person out. This step finds the rows in groups
smaller than k and blanks their quasi-identifiers, most varied column first, until
the groups merge. It changes only the rows at risk.

It runs on the whole table in the engine that holds it, with group sizes as windows.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw

from data_anonymizer.engines import WorkingFrame, first_row, to_native, to_working_frame
from data_anonymizer.evaluation.privacy import distinct_present

_AT_RISK = "__da_at_risk"


@dataclass(frozen=True, slots=True)
class SuppressionOutcome:
    """The table after suppression, in the form it was given, and how many cells were blanked."""

    data: Any
    suppressed_cells: int


def enforce_k_anonymity(frame: Any, quasi_identifiers: Sequence[str], k: int) -> SuppressionOutcome:
    """Blank quasi-identifier cells in rows whose group is smaller than k.

    Takes a pandas, Polars or DuckDB table, or a Narwhals frame, and returns the same
    kind. When even fully blanked rows number fewer than k, they still form one group
    below k; the privacy report shows it, and dropping those rows is left to the caller.
    """
    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return _suppress(frame, quasi_identifiers, k)
    working, engine = to_working_frame(frame)
    outcome = _suppress(working, quasi_identifiers, k)
    return SuppressionOutcome(to_native(outcome.data, engine), outcome.suppressed_cells)


def _suppress(frame: WorkingFrame, quasi_identifiers: Sequence[str], k: int) -> SuppressionOutcome:
    names = frame.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    if not columns or k <= 1:
        return SuppressionOutcome(frame, 0)

    suppressed = 0
    for column in _most_varied_first(frame, columns):
        flagged = frame.with_columns(_in_small_group(columns, k).alias(_AT_RISK))
        counts = first_row(
            flagged.select(
                nw.col(_AT_RISK).sum().alias("at_risk"),
                (nw.col(_AT_RISK) & ~nw.col(column).is_null()).sum().alias("to_blank"),
            )
        )
        if not counts["at_risk"]:
            break
        suppressed += int(counts["to_blank"])
        keep_value = ~nw.col(_AT_RISK) | nw.col(column).is_null()
        frame = flagged.with_columns(nw.when(keep_value).then(nw.col(column)).alias(column)).drop(
            _AT_RISK
        )
    return SuppressionOutcome(frame, suppressed)


def _most_varied_first(frame: WorkingFrame, columns: list[str]) -> list[str]:
    distinct = first_row(
        frame.select(*[distinct_present(column).alias(column) for column in columns])
    )
    return sorted(columns, key=lambda column: -distinct[column])


def _in_small_group(columns: list[str], k: int) -> nw.Expr:
    return nw.len().over(*columns) < k
