"""Check what several anonymized tables reveal when they are joined through their links.

`apply_many` anonymizes related tables under one throwaway key, so a column such as an account
ID gets the same code in every table. That is what lets a model use the tables together, and
it is also a join key. Each table can be k-anonymous on its own while the entities, seen
through all their tables at once, are not: an account in district 1 (first table) and in band 1
(second table) may be the only one that is both.

Two kinds of link join tables:
- a column that several anonymizing plans release with the same name and consistent codes;
- two columns with different names that code the same entities, found because every entity
  occurs a distinctive number of times in both (see `frequency_links`).

Tables that are linked, directly or through other tables, form one component. A bridge table
that holds an account and a case code joins a table that has only the account to a table that
has only the case code, though no single link spans all three. The tables of each component are
joined through all their links, and so is every connected subset of them (a table with only
some of the entities shrinks the full join and could hide a pair that singles a person out).
Two checks run on each join, counting entities and not rows:
- k-anonymity: every combination of the tables' quasi-identifiers is shared by at least the
  largest `k_target` of those plans' entities;
- inference: no combination lets a sensitive value of one table be guessed with high
  confidence above its base rate, by the same rule as the single-table check.
An entity is the link whose values are fewest in the joined result, the coarsest key, which is
the person when the other links are visits or cases of that person.
The checks take the released tables and the plans and return problem texts.
"""

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import narwhals as nw
import pandas as pd

from oblitus.engines import as_working_frame, collect
from oblitus.evaluation.inference import confident_inferences
from oblitus.frequency_links import frequency_links
from oblitus.plan import AnonymizationPlan, Protection

# These leave a column's values the same from table to table within one anonymizing call.
_CONSISTENT_TECHNIQUES = {"keep", "scramble_categories"}
_DISCRETE_LIMIT = 50
# Every connected subset of tables is joined and checked up to this many tables in a component.
MAX_ENUMERATED_TABLES = 6
_LINK_PREFIX = "__link"


@dataclass(frozen=True, slots=True)
class _Contribution:
    """The quasi-identifiers and sensitive attributes one table adds to a join."""

    quasi: list[str]
    sensitive: list[str]


@dataclass(frozen=True, slots=True)
class _Link:
    """One entity key as several tables hold it: each table's column, and how to recode it.

    `recodes` maps a table's values to the values of the first table, for links whose codes
    differ from table to table.
    """

    description: str
    columns: dict[str, str]
    recodes: dict[str, dict[Hashable, Hashable]] = field(default_factory=dict)


def joined_table_problems(
    released: Mapping[str, Any], plans: Mapping[str, AnonymizationPlan]
) -> dict[str, list[str]]:
    """The problems each table has because of what its links allow, by table name.

    Only anonymizing plans are looked at: pseudonymized tables join by design.
    """
    tables = {
        name: _released_as_pandas(data)
        for name, data in released.items()
        if plans[name].protection is Protection.ANONYMIZATION
    }
    links = _name_links(tables, plans)
    links += _count_links(tables, plans)
    problems: dict[str, list[str]] = {}
    for component in _components(list(tables), links):
        for names in _connected_subsets(component, links):
            subset_links = _links_within(names, links)
            found = _problems_of_join(
                subset_links,
                {name: tables[name] for name in names},
                {name: plans[name] for name in names},
            )
            through = ", ".join(link.description for link in subset_links)
            for name in names:
                others = ", ".join(repr(other) for other in names if other != name)
                problems.setdefault(name, []).extend(
                    f"joined with {others} on {through}: {text}" for text in found
                )
    return problems


def _links_within(names: Sequence[str], links: Sequence[_Link]) -> list[_Link]:
    """The links as they hold among `names` only, leaving out those that join fewer than two."""
    chosen = set(names)
    restricted = []
    for link in links:
        columns = {name: column for name, column in link.columns.items() if name in chosen}
        if len(columns) < 2:
            continue
        recodes = {name: pairing for name, pairing in link.recodes.items() if name in chosen}
        restricted.append(_Link(link.description, columns, recodes))
    return restricted


def _connected_subsets(component: list[str], links: Sequence[_Link]) -> list[list[str]]:
    """Every set of two or more tables of a component that is joined by its own links.

    A third table that holds only some of the entities shrinks the inner join of all three, so
    the join of two of them can single out a person the full join hides. Components of up to
    `MAX_ENUMERATED_TABLES` tables are checked in every connected subset; larger ones in every
    connected pair and the full join, which keeps the work from doubling with each table.
    """
    every_size = len(component) <= MAX_ENUMERATED_TABLES
    largest = len(component) if every_size else 2
    subsets = [
        list(chosen)
        for size in range(2, largest + 1)
        for chosen in combinations(component, size)
        if _is_connected(chosen, links)
    ]
    if not every_size:
        subsets.append(list(component))
    return subsets


def _is_connected(names: Sequence[str], links: Sequence[_Link]) -> bool:
    groups = _components(list(names), _links_within(names, links))
    return len(groups) == 1 and len(groups[0]) == len(names)


def _released_as_pandas(data: Any) -> pd.DataFrame:
    return collect(as_working_frame(data)).to_pandas()


def _name_links(
    tables: Mapping[str, pd.DataFrame], plans: Mapping[str, AnonymizationPlan]
) -> list[_Link]:
    """Columns that two or more anonymized tables release with the same codes."""
    holders: dict[str, list[str]] = {}
    for name, frame in tables.items():
        for column in frame.columns:
            if _is_consistent(plans[name], column) and frame[column].notna().any():
                holders.setdefault(column, []).append(name)
    return [
        _Link(repr(column), {name: column for name in names})
        for column, names in holders.items()
        if len(names) >= 2
    ]


def _is_consistent(plan: AnonymizationPlan, column: str) -> bool:
    action = plan.action_for(column)
    return action is None or action.technique in _CONSISTENT_TECHNIQUES


def _count_links(
    tables: Mapping[str, pd.DataFrame],
    plans: Mapping[str, AnonymizationPlan],
) -> list[_Link]:
    """Links between differently named columns whose values occur equally often.

    A column that already sits in a named link is compared too: a third table that copies it
    must not hide that it also matches a differently named column of another table.
    """
    candidates = {
        name: [
            column
            for column in frame.columns
            if _is_consistent(plans[name], column) and column not in _measured_columns(plans[name])
        ]
        for name, frame in tables.items()
    }
    return [
        _Link(
            f"{found.left_table}.{found.left_column} ~ {found.right_table}.{found.right_column} "
            "(same row counts)",
            {found.left_table: found.left_column, found.right_table: found.right_column},
            {found.right_table: found.pairing},
        )
        for found in frequency_links(tables, candidates)
    ]


def _measured_columns(plan: AnonymizationPlan) -> set[str]:
    return {*plan.released_quasi_identifiers, *plan.sensitive_attributes}


def _components(names: list[str], links: Sequence[_Link]) -> list[list[str]]:
    """The groups of tables that are linked, directly or through other tables."""
    group_of = {name: name for name in names}

    def root(name: str) -> str:
        while group_of[name] != name:
            name = group_of[name]
        return name

    for link in links:
        first, *rest = list(link.columns)
        for other in rest:
            group_of[root(other)] = root(first)
    components: dict[str, list[str]] = {}
    for name in names:
        components.setdefault(root(name), []).append(name)
    return [group for group in components.values() if len(group) >= 2]


def _problems_of_join(
    links: Sequence[_Link],
    tables: Mapping[str, pd.DataFrame],
    plans: Mapping[str, AnonymizationPlan],
) -> list[str]:
    contributions = {name: _contribution(name, links, tables[name], plans[name]) for name in tables}
    combined = _join(links, tables, contributions)
    quasi = [f"{name}.{column}" for name, own in contributions.items() for column in own.quasi]
    if combined.empty or not quasi:
        return []
    entity = _coarsest_link(combined)
    target = max(plan.k_target for plan in plans.values())
    entities = combined.groupby(quasi, dropna=False)[entity].nunique()
    problems = []
    if entities.min() < target:
        problems.append(
            f"{int((entities < target).sum())} combination(s) of {', '.join(quasi)} are shared by "
            f"fewer than {target} entities (the smallest by {entities.min()}), so an entity "
            "can be singled out by what the tables show together; drop the link column, or "
            "release the tables in separate calls so each gets its own codes"
        )
    problems += _joined_inferences(entity, combined, quasi, contributions, plans)
    return problems


def _coarsest_link(combined: pd.DataFrame) -> str:
    """The link column with the fewest distinct values, which stands for the entity."""
    keys = [column for column in combined.columns if column.startswith(_LINK_PREFIX)]
    return min(keys, key=lambda key: combined[key].nunique())


def _contribution(
    name: str, links: Sequence[_Link], frame: pd.DataFrame, plan: AnonymizationPlan
) -> _Contribution:
    used = {link.columns[name] for link in links if name in link.columns}
    present = [column for column in frame.columns if column not in used]
    return _Contribution(
        quasi=[column for column in plan.released_quasi_identifiers if column in present],
        sensitive=[column for column in plan.sensitive_attributes if column in present],
    )


def _join(
    links: Sequence[_Link],
    tables: Mapping[str, pd.DataFrame],
    contributions: Mapping[str, _Contribution],
) -> pd.DataFrame:
    """One row per entity and distinct combination of what the linked tables show about it."""
    prepared = {
        name: _keyed(name, frame, links, contributions[name]) for name, frame in tables.items()
    }
    pending = list(prepared)
    combined = prepared[pending.pop(0)]
    while pending:
        name = next((n for n in pending if _shared_keys(combined, prepared[n])), pending[0])
        pending.remove(name)
        combined = combined.merge(
            prepared[name], on=_shared_keys(combined, prepared[name]), how="inner"
        )
    return combined


def _shared_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [
        column
        for column in left.columns
        if column.startswith(_LINK_PREFIX) and column in right.columns
    ]


def _keyed(
    name: str, frame: pd.DataFrame, links: Sequence[_Link], own: _Contribution
) -> pd.DataFrame:
    """A table's distinct rows: its link columns under shared key names, and its own columns."""
    columns = [*own.quasi, *own.sensitive]
    kept = frame[columns].copy()
    kept.columns = [f"{name}.{column}" for column in columns]
    keys = []
    for number, link in enumerate(links):
        if name not in link.columns:
            continue
        values = frame[link.columns[name]]
        if name in link.recodes:
            values = values.map(link.recodes[name])
        key = f"{_LINK_PREFIX}{number}"
        kept[key] = values
        keys.append(key)
    return kept.dropna(subset=keys).drop_duplicates()


def _joined_inferences(
    entity: str,
    combined: pd.DataFrame,
    quasi: list[str],
    contributions: Mapping[str, _Contribution],
    plans: Mapping[str, AnonymizationPlan],
) -> list[str]:
    problems = []
    for name, own in contributions.items():
        for attribute in own.sensitive:
            joined_name = f"{name}.{attribute}"
            entity_rows = combined[[entity, *quasi, joined_name]].drop_duplicates()
            findings = confident_inferences(
                nw.from_native(entity_rows),
                quasi,
                [joined_name],
                confidence=plans[name].inference_confidence,
                gain=plans[name].inference_gain,
                discrete_limit=_DISCRETE_LIMIT,
            )
            problems += [
                f"{joined_name}: a group of {finding.group_rows} entities with "
                f"{finding.group!r} has {finding.group_share:.0%} {finding.value!r} against "
                f"{finding.table_share:.0%} overall; drop the link column, or release the "
                "tables in separate calls"
                for finding in findings
            ]
    return problems
