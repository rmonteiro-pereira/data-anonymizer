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

from oblitus.engines import (
    WorkingFrame,
    collect,
    first_row,
    to_native,
    to_working_frame,
)
from oblitus.evaluation.privacy import distinct_present

_AT_RISK = "__da_at_risk"
_GROUP_ROWS = "__da_group_rows"


@dataclass(frozen=True, slots=True)
class SuppressionOutcome:
    """The table after suppression, in the form it was given, and how many cells were blanked."""

    data: Any
    suppressed_cells: int


def enforce_k_anonymity(frame: Any, quasi_identifiers: Sequence[str], k: int) -> SuppressionOutcome:
    """Blank quasi-identifier cells in rows whose group is smaller than k.

    Takes a pandas, Polars or DuckDB table, or a Narwhals frame, and returns the same
    kind. When even fully blanked rows number fewer than k, the smallest complete group is
    blanked too, so they form one group of at least k.
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
    frame, merged = _merge_leftover_rows(frame, columns, k)
    return SuppressionOutcome(frame, suppressed + merged)


def _merge_leftover_rows(
    frame: WorkingFrame, columns: list[str], k: int
) -> tuple[WorkingFrame, int]:
    """Blank the smallest complete group too when fully blanked rows number fewer than k.

    Rows left at risk have every quasi-identifier blank, so they form one group. Blanking
    the smallest group that already has k rows merges it into theirs, which costs the
    fewest cells and reaches k without dropping any row.
    """
    groups = frame.group_by(columns).agg(nw.len().alias(_GROUP_ROWS))
    all_blank = nw.all_horizontal(
        *[nw.col(column).is_null() for column in columns], ignore_nulls=False
    )
    leftover = first_row(groups.filter(all_blank).select(nw.col(_GROUP_ROWS).sum().alias("rows")))
    if not leftover["rows"] or leftover["rows"] >= k:
        return frame, 0
    complete = groups.filter(~all_blank & (nw.col(_GROUP_ROWS) >= k)).sort(_GROUP_ROWS)
    smallest = collect(complete.head(1))
    if smallest.shape[0] == 0:
        return frame, 0
    donor = dict(zip(smallest.columns, smallest.row(0), strict=True))
    in_donor = nw.all_horizontal(
        *[_matches(column, donor[column]) for column in columns], ignore_nulls=False
    )
    blanked = sum(donor[column] is not None for column in columns) * donor[_GROUP_ROWS]
    cleared = [nw.when(~in_donor).then(nw.col(column)).alias(column) for column in columns]
    return frame.with_columns(*cleared), int(blanked)


def _matches(column: str, value: Any) -> nw.Expr:
    if value is None:
        return nw.col(column).is_null()
    return nw.col(column) == value


def _most_varied_first(frame: WorkingFrame, columns: list[str]) -> list[str]:
    distinct = first_row(
        frame.select(*[distinct_present(column).alias(column) for column in columns])
    )
    return sorted(columns, key=lambda column: -distinct[column])


def _in_small_group(columns: list[str], k: int) -> nw.Expr:
    return nw.len().over(*columns) < k
