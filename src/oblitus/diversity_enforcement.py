"""l-diversity enforcement: merge groups until each holds at least l sensitive values.

k-anonymity hides which row is yours inside a group, but a group whose rows all share one
diagnosis still tells an attacker the diagnosis. This step finds the quasi-identifier
groups that hold fewer than `l` distinct values of a sensitive attribute and blanks their
quasi-identifiers, so those rows join the one group whose quasi-identifiers are all blank.
When that merged group is still short of `l` values, the smallest remaining groups are
blanked into it, as local suppression does for k, until it has enough.

Whole groups are blanked, never single cells, because a group that keeps some cells keeps
the sensitive values that identify it. A sensitive attribute with fewer than `l` values in
the whole table cannot be helped, so it is asked for as many values as the table holds.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import narwhals as nw

from oblitus.engines import WorkingFrame, collect, to_native, to_working_frame
from oblitus.group_rewrite import distinct_rows, overwrite_by_key


@dataclass(frozen=True, slots=True)
class DiversityOutcome:
    """The table after merging, in the form it was given, and how many cells were blanked."""

    data: Any
    suppressed_cells: int


def enforce_l_diversity(
    frame: Any,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    l_target: int,
) -> DiversityOutcome:
    """Blank the quasi-identifiers of groups with fewer than `l_target` sensitive values.

    Takes a pandas, Polars or DuckDB table, or a Narwhals frame, and returns the same kind.
    """
    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return _enforce(frame, quasi_identifiers, sensitive_attributes, l_target)
    working, engine = to_working_frame(frame)
    outcome = _enforce(working, quasi_identifiers, sensitive_attributes, l_target)
    return DiversityOutcome(to_native(outcome.data, engine), outcome.suppressed_cells)


def _enforce(
    frame: WorkingFrame,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    l_target: int,
) -> DiversityOutcome:
    names = frame.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    attributes = [attribute for attribute in sensitive_attributes if attribute in names]
    if not columns or not attributes or l_target <= 1:
        return DiversityOutcome(frame, 0)

    combinations = distinct_rows(frame, columns)
    keys = list(zip(*[combinations.get_column(name).to_list() for name in columns], strict=True))
    counts = combinations.get_column("count").to_list()
    values = {
        attribute: _values_per_group(frame, columns, attribute, keys) for attribute in attributes
    }
    required = {
        attribute: min(l_target, len(set().union(*values[attribute]))) for attribute in attributes
    }
    merged = _groups_to_merge(keys, counts, values, required)

    blanked = [key for position, key in enumerate(keys) if position in merged]
    newly_blanked = [key for key in blanked if any(part is not None for part in key)]
    if not newly_blanked:
        return DiversityOutcome(frame, 0)
    replacements = {
        column: (
            [None if position in merged else key[index] for position, key in enumerate(keys)],
            combinations.schema[column],
        )
        for index, column in enumerate(columns)
    }
    rewritten = overwrite_by_key(frame, columns, keys, replacements)
    cells = sum(
        counts[position] * sum(part is not None for part in keys[position]) for position in merged
    )
    return DiversityOutcome(rewritten, cells)


def _values_per_group(
    frame: WorkingFrame, columns: list[str], attribute: str, keys: list[tuple[Any, ...]]
) -> list[set[Any]]:
    """For each group, in the order of `keys`, the set of present values of the attribute."""
    pairs = collect(frame.select(*columns, attribute).drop_nulls(attribute).unique())
    position_of = {key: position for position, key in enumerate(keys)}
    found: dict[int, set[Any]] = defaultdict(set)
    for row in pairs.iter_rows():
        found[position_of[tuple(row[:-1])]].add(row[-1])
    return [found[position] for position in range(len(keys))]


def _groups_to_merge(
    keys: list[tuple[Any, ...]],
    counts: list[int],
    values: dict[str, list[set[Any]]],
    required: dict[str, int],
) -> set[int]:
    """The groups to blank so that together they hold enough values of every attribute."""
    merged = {
        position
        for position, key in enumerate(keys)
        if all(part is None for part in key)
        or any(len(values[attribute][position]) < required[attribute] for attribute in values)
    }
    if not merged:
        return merged
    held = {
        attribute: set().union(*[values[attribute][position] for position in merged])
        for attribute in values
    }
    by_size = sorted(range(len(keys)), key=lambda position: (counts[position], position))
    spare = iter([position for position in by_size if position not in merged])
    while _short_attributes(held, required):
        position = next(spare, None)
        if position is None:
            break
        merged.add(position)
        for attribute in values:
            held[attribute] |= values[attribute][position]
    return merged


def _short_attributes(held: dict[str, set[Any]], required: dict[str, int]) -> list[str]:
    return [attribute for attribute in held if len(held[attribute]) < required[attribute]]
