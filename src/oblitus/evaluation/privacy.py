"""Measure how easily records in a table can be singled out.

k-anonymity: every combination of quasi-identifier values is shared by at least k
rows. l-diversity: inside each such group, the sensitive attribute takes at least
l different values, so belonging to a group does not reveal the attribute.

t-closeness: inside each such group, the sensitive attribute is distributed like it is in
the whole table, within a distance `t` (Li, Li and Venkatasubramanian, 2007). The distance is
the earth mover's distance, from 0 (the same distribution) to 1. Numbers and dates use the
ordered distance, where moving mass to a neighbouring value is cheap, and other values use the
equal distance, where every move costs the same.

k-anonymity and l-diversity are computed on the whole table by the engine that holds it, as
group-bys, so a large Polars or DuckDB table is never converted to pandas. t-closeness
collects the distinct groups and values, which is small unless a continuous column is both
a quasi-identifier and sensitive.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import narwhals as nw
import numpy as np

from oblitus.engines import WorkingFrame, as_working_frame, collect, first_row

GROUP_SIZE = "__da_group_size"
_ROWS = "__da_rows"


@dataclass(frozen=True, slots=True)
class PrivacyReport:
    """k-anonymity, l-diversity and, when asked for, t-closeness over the quasi-identifiers.

    `t_closeness` holds, per sensitive attribute, the largest distance of any group from the
    whole table. It is filled only when `assess_privacy` is given a `t_target`.
    """

    quasi_identifiers: list[str]
    k_anonymity: int
    k_target: int
    rows_below_target: int
    share_below_target: float
    l_diversity: dict[str, int] = field(default_factory=dict)
    t_closeness: dict[str, float] = field(default_factory=dict)
    t_target: float | None = None

    @property
    def meets_target(self) -> bool:
        """True when every quasi-identifier group has at least k_target rows."""
        return self.k_anonymity >= self.k_target

    @property
    def meets_t_target(self) -> bool:
        """True when no group is farther than `t_target` from the whole table, or none is set."""
        if self.t_target is None:
            return True
        return all(distance <= self.t_target for distance in self.t_closeness.values())


def assess_privacy(
    frame: Any,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    *,
    k_target: int = 5,
    t_target: float | None = None,
) -> PrivacyReport:
    """Compute k-anonymity and l-diversity of a pandas, Polars or DuckDB table.

    With a `t_target`, t-closeness is measured too and held against that threshold.
    """
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
        t_closeness=t_closeness(working, columns, scored) if t_target is not None else {},
        t_target=t_target,
    )


def t_closeness(
    frame: Any, quasi_identifiers: Sequence[str], sensitive_attributes: Sequence[str]
) -> dict[str, float]:
    """The largest earth mover's distance between a group and the table, per sensitive attribute.

    Blank cells are left out of both distributions. A table without quasi-identifiers is a
    single group, so its distance is 0.
    """
    working = as_working_frame(frame)
    names = working.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    return {
        attribute: _worst_group_distance(working, columns, attribute)
        for attribute in sensitive_attributes
        if attribute in names
    }


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


def _worst_group_distance(frame: WorkingFrame, columns: list[str], attribute: str) -> float:
    dtype = frame.collect_schema()[attribute]
    ordered = bool(dtype.is_numeric() or dtype.is_temporal())
    rows = collect(
        frame.drop_nulls(attribute).group_by([*columns, attribute]).agg(nw.len().alias(_ROWS))
    ).rows()
    if not rows:
        return 0.0
    values = sorted({row[-2] for row in rows})
    position_of = {value: position for position, value in enumerate(values)}
    table_counts = np.zeros(len(values))
    group_counts: dict[tuple[Any, ...], dict[int, int]] = {}
    for *group, value, count in rows:
        position = position_of[value]
        table_counts[position] += count
        group_counts.setdefault(tuple(group), {})[position] = count
    table_shares = table_counts / table_counts.sum()
    distance = _ordered_distance if ordered else _equal_distance
    return max(distance(counts, table_shares) for counts in group_counts.values())


def _equal_distance(group_counts: dict[int, int], table_shares: np.ndarray) -> float:
    """Half the total difference between two shares: every move between values costs the same."""
    total = sum(group_counts.values())
    positions = np.array(list(group_counts))
    group_shares = np.array(list(group_counts.values())) / total
    overlap = np.abs(group_shares - table_shares[positions]).sum()
    outside = 1.0 - table_shares[positions].sum()
    return float((overlap + outside) / 2)


def _ordered_distance(group_counts: dict[int, int], table_shares: np.ndarray) -> float:
    """Earth mover's distance on ordered values, each step to a neighbour costing 1/(m - 1).

    The distance sums, over every value, how far the group's running share is from the
    table's. The group's running share is constant between its own values, so each stretch is
    summed from prefix sums of the table's running share without visiting every value.
    """
    value_count = len(table_shares)
    if value_count < 2:
        return 0.0
    table_running = np.cumsum(table_shares)
    prefix = np.concatenate([[0.0], np.cumsum(table_running)])
    positions = sorted(group_counts)
    total = sum(group_counts.values())
    running = 0.0
    start = 0
    area = 0.0
    for position in positions:
        area += _stretch_area(table_running, prefix, start, position, running)
        running += group_counts[position] / total
        start = position
    area += _stretch_area(table_running, prefix, start, value_count, running)
    return float(min(1.0, area / (value_count - 1)))


def _stretch_area(
    table_running: np.ndarray, prefix: np.ndarray, start: int, stop: int, level: float
) -> float:
    """Sum of |level - table_running| over positions start..stop-1; the table's share rises."""
    if stop <= start:
        return 0.0
    below_end = start + int(np.searchsorted(table_running[start:stop], level, side="left"))
    below = level * (below_end - start) - (prefix[below_end] - prefix[start])
    above = (prefix[stop] - prefix[below_end]) - level * (stop - below_end)
    return float(below + above)
