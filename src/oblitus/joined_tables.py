"""Check what several anonymized tables reveal when they are joined on a shared column.

`apply_many` anonymizes related tables under one throwaway key, so a column such as an account
ID gets the same code in every table. That is what lets a model use the tables together, and
it is also a join key. Each table can be k-anonymous on its own while the entities, seen
through all their tables at once, are not: an account in district 1 (first table) and in band 1
(second table) may be the only one that is both.

For every column that several anonymizing plans release with consistent codes, the released
tables are joined on it, and two checks run on the result, counting entities and not rows:
- k-anonymity: every combination of the tables' quasi-identifiers is shared by at least the
  largest `k_target` of those plans' entities;
- inference: no combination lets a sensitive value of one table be guessed with high
  confidence above its base rate, by the same rule as the single-table check.
The checks take the released tables and the plans and return problem texts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import pandas as pd

from oblitus.engines import as_working_frame, collect
from oblitus.evaluation.inference import confident_inferences
from oblitus.plan import AnonymizationPlan, Protection

# These leave a column's values the same from table to table within one anonymizing call.
_CONSISTENT_TECHNIQUES = {"keep", "scramble_categories"}
_DISCRETE_LIMIT = 50


@dataclass(frozen=True, slots=True)
class _Contribution:
    """The quasi-identifiers and sensitive attributes one table adds to a join."""

    quasi: list[str]
    sensitive: list[str]


def joined_table_problems(
    released: Mapping[str, Any], plans: Mapping[str, AnonymizationPlan]
) -> dict[str, list[str]]:
    """The problems each table has because of what its linking columns allow, by table name.

    Only anonymizing plans are looked at: pseudonymized tables join by design.
    """
    tables = {
        name: _released_as_pandas(data)
        for name, data in released.items()
        if plans[name].protection is Protection.ANONYMIZATION
    }
    problems: dict[str, list[str]] = {}
    for link, names in _linking_columns(tables, plans).items():
        found = _problems_of_join(
            link, {name: tables[name] for name in names}, {name: plans[name] for name in names}
        )
        for name in names:
            others = ", ".join(repr(other) for other in names if other != name)
            problems.setdefault(name, []).extend(
                f"joined with {others} on {link!r}: {text}" for text in found
            )
    return problems


def _released_as_pandas(data: Any) -> pd.DataFrame:
    return collect(as_working_frame(data)).to_pandas()


def _linking_columns(
    tables: Mapping[str, pd.DataFrame], plans: Mapping[str, AnonymizationPlan]
) -> dict[str, list[str]]:
    """Columns that two or more anonymized tables release with the same codes."""
    holders: dict[str, list[str]] = {}
    for name, frame in tables.items():
        for column in frame.columns:
            action = plans[name].action_for(column)
            consistent = action is None or action.technique in _CONSISTENT_TECHNIQUES
            if consistent and frame[column].notna().any():
                holders.setdefault(column, []).append(name)
    return {column: names for column, names in holders.items() if len(names) >= 2}


def _problems_of_join(
    link: str, tables: Mapping[str, pd.DataFrame], plans: Mapping[str, AnonymizationPlan]
) -> list[str]:
    contributions = {name: _contribution(link, tables[name], plans[name]) for name in tables}
    combined = _join(link, tables, contributions)
    quasi = [f"{name}.{column}" for name, own in contributions.items() for column in own.quasi]
    if combined.empty or not quasi:
        return []
    target = max(plan.k_target for plan in plans.values())
    entities = combined.groupby(quasi, dropna=False)[link].nunique()
    problems = []
    if entities.min() < target:
        problems.append(
            f"{int((entities < target).sum())} combination(s) of {', '.join(quasi)} are shared by "
            f"fewer than {target} entities (the smallest by {entities.min()}), so an entity "
            "can be singled out by what the tables show together; drop the link column, or "
            "release the tables in separate calls so each gets its own codes"
        )
    problems += _joined_inferences(link, combined, quasi, contributions, plans)
    return problems


def _contribution(link: str, frame: pd.DataFrame, plan: AnonymizationPlan) -> _Contribution:
    present = [column for column in frame.columns if column != link]
    return _Contribution(
        quasi=[column for column in plan.quasi_identifiers if column in present],
        sensitive=[column for column in plan.sensitive_attributes if column in present],
    )


def _join(
    link: str, tables: Mapping[str, pd.DataFrame], contributions: Mapping[str, _Contribution]
) -> pd.DataFrame:
    """One row per entity and distinct combination of what the tables show about it."""
    combined = pd.DataFrame()
    for position, (name, frame) in enumerate(tables.items()):
        own = contributions[name]
        columns = [*own.quasi, *own.sensitive]
        kept = frame[[link, *columns]].dropna(subset=[link]).drop_duplicates()
        kept = kept.rename(columns={column: f"{name}.{column}" for column in columns})
        combined = kept if position == 0 else combined.merge(kept, on=link, how="inner")
    return combined


def _joined_inferences(
    link: str,
    combined: pd.DataFrame,
    quasi: list[str],
    contributions: Mapping[str, _Contribution],
    plans: Mapping[str, AnonymizationPlan],
) -> list[str]:
    problems = []
    for name, own in contributions.items():
        for attribute in own.sensitive:
            joined_name = f"{name}.{attribute}"
            entity_rows = combined[[link, *quasi, joined_name]].drop_duplicates()
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
