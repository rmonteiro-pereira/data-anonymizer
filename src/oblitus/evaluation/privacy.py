"""Measure how easily records in a table can be singled out.

k-anonymity: every combination of quasi-identifier values is shared by at least k
rows. l-diversity: inside each such group, the sensitive attribute takes at least
l different values, so belonging to a group does not reveal the attribute.

Both are computed on the whole table by the engine that holds it, as group-bys, so a
large Polars or DuckDB table is never converted to pandas.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw

from oblitus.engines import WorkingFrame, as_working_frame, first_row

GROUP_SIZE = "__da_group_size"
_ROWS = "__da_rows"


@dataclass(frozen=True, slots=True)
class PrivacyReport:
    """k-anonymity and l-diversity of a table over its quasi-identifiers."""

    quasi_identifiers: list[str]
    k_anonymity: int
    k_target: int
    rows_below_target: int
    share_below_target: float
    l_diversity: dict[str, int] = field(default_factory=dict)

    @property
    def meets_target(self) -> bool:
        """True when every quasi-identifier group has at least k_target rows."""
        return self.k_anonymity >= self.k_target


def assess_privacy(
    frame: Any,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    *,
    k_target: int = 5,
) -> PrivacyReport:
    """Compute k-anonymity and l-diversity of a pandas, Polars or DuckDB table."""
    working = as_working_frame(frame)
    names = working.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    counts = _counts(working, [a for a in sensitive_attributes if a in names])
    total = counts[_ROWS]
    if not columns or total == 0:
        return PrivacyReport(
            quasi_identifiers=columns,
            k_anonymity=total,
            k_target=k_target,
            rows_below_target=0,
            share_below_target=0.0,
        )

    # A sensitive column released blank, or with one value for everyone, teaches an attacker
    # nothing about any one person, so it is not scored. A column indexed per entity is all 100
    # when every entity's value is flat.
    scored = [
        attribute for attribute, distinct in counts.items() if attribute != _ROWS and distinct >= 2
    ]
    groups = working.group_by(columns).agg(
        nw.len().alias(GROUP_SIZE),
        *[distinct_present(attribute).alias(attribute) for attribute in scored],
    )
    sizes = nw.col(GROUP_SIZE)
    summary = first_row(
        groups.select(
            sizes.min().alias("smallest"),
            nw.when(sizes < k_target).then(sizes).otherwise(0).sum().alias("below"),
            *[nw.col(attribute).min().alias(attribute) for attribute in scored],
        )
    )
    return PrivacyReport(
        quasi_identifiers=columns,
        k_anonymity=int(summary["smallest"]),
        k_target=k_target,
        rows_below_target=int(summary["below"]),
        share_below_target=int(summary["below"]) / total,
        l_diversity={attribute: int(summary[attribute]) for attribute in scored},
    )


def group_sizes(frame: WorkingFrame, columns: Sequence[str]) -> WorkingFrame:
    """One row per distinct combination of `columns`, with its row count in `GROUP_SIZE`.

    Blank cells form a group of their own, as they do for an attacker who sees them.
    """
    return frame.group_by(list(columns)).agg(nw.len().alias(GROUP_SIZE))


def _counts(frame: WorkingFrame, attributes: Sequence[str]) -> dict[str, int]:
    row = first_row(
        frame.select(
            nw.len().alias(_ROWS),
            *[distinct_present(attribute).alias(attribute) for attribute in attributes],
        )
    )
    return {name: int(value) for name, value in row.items()}


def distinct_present(column: str) -> nw.Expr:
    """Aggregate: how many different values a column holds, not counting blank cells.

    Blank cells are not a value an attacker learns, so they add no diversity. SQL engines
    cannot drop nulls inside an aggregate, so the blank is subtracted instead.
    """
    values = nw.col(column)
    return values.n_unique() - values.is_null().any().cast(nw.Int64)
