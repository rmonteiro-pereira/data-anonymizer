"""Find groups in which a sensitive value can be guessed with high confidence.

l-diversity counts the distinct sensitive values in a quasi-identifier group, not how they
are spread. A group of 100 rows with 99 `diabetes` and 1 `healthy` holds two values and still
tells an attacker who knows a person's group that the person has diabetes. t-closeness catches
this only when the plan sets a distance. This check needs no tuning for the usual case: it
flags a group when both
- the share of one sensitive value in the group is at least `confidence`, and
- that share exceeds the value's share in the whole table by at least `gain`.

The first condition says the guess is usually right. The second says the release taught the
attacker something: a table that is 95% `healthy` makes "healthy" a safe guess with no help
from the group. This follows the idea behind beta-likeness (Cao and Karras, 2012), which bounds
the gain over the base rate, with the gain measured in percentage points.

The counts are group-bys run by the engine that holds the table; only the distinct groups and
values are collected.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import narwhals as nw

from oblitus.engines import WorkingFrame, collect

# A guess right nine times in ten is a confident one. Thirty points over the base rate keeps a
# naturally lopsided attribute (95% of rows `healthy`) from being flagged for what everyone
# already knows, while a 50/50 attribute is flagged as soon as a group reaches 90%.
DEFAULT_CONFIDENCE = 0.9
DEFAULT_GAIN = 0.3

_ROWS = "__da_rows"


@dataclass(frozen=True, slots=True)
class ConfidentInference:
    """The most exposed group of one sensitive attribute and how many groups are exposed."""

    attribute: str
    group: dict[str, object]
    value: object
    group_share: float
    table_share: float
    group_rows: int
    exposed_groups: int


def confident_inferences(
    frame: WorkingFrame,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    gain: float = DEFAULT_GAIN,
    discrete_limit: int = 50,
    entity_column: str | None = None,
) -> list[ConfidentInference]:
    """For each sensitive attribute, its most exposed group, if any group is exposed.

    Blank sensitive cells are left out. A numeric attribute with more than `discrete_limit`
    distinct values is skipped, since no single value of it has a meaningful share. A text
    attribute is checked however many values it has: a diagnosis column with hundreds of
    codes can still give away its most common code for one group. With
    `entity_column`, an entity counts once per group and value, however many rows it has.
    """
    names = frame.collect_schema().names()
    columns = [column for column in quasi_identifiers if column in names]
    if not columns:
        return []
    entity = entity_column if entity_column in names and entity_column not in columns else None
    findings = []
    for attribute in sensitive_attributes:
        if attribute not in names:
            continue
        finding = _worst_exposure(
            frame, columns, attribute, confidence, gain, discrete_limit, entity
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _has_numeric_dtype(frame: WorkingFrame, attribute: str) -> bool:
    return frame.collect_schema()[attribute].is_numeric()


def _worst_exposure(
    frame: WorkingFrame,
    columns: list[str],
    attribute: str,
    confidence: float,
    gain: float,
    discrete_limit: int,
    entity: str | None,
) -> ConfidentInference | None:
    present = frame.drop_nulls(attribute)
    size = nw.len() if entity is None else nw.col(entity).n_unique()
    counted = collect(present.group_by([*columns, attribute]).agg(size.alias(_ROWS))).rows()
    table_counts: dict[object, int] = {}
    group_totals: dict[tuple[object, ...], int] = {}
    for *group, value, count in counted:
        table_counts[value] = table_counts.get(value, 0) + count
        group_totals[tuple(group)] = group_totals.get(tuple(group), 0) + count
    if len(table_counts) < 2:
        return None
    if _has_numeric_dtype(present, attribute) and len(table_counts) > discrete_limit:
        return None
    table_rows = sum(table_counts.values())
    exposed = 0
    worst: ConfidentInference | None = None
    for *group, value, count in counted:
        key = tuple(group)
        group_share = count / group_totals[key]
        table_share = table_counts[value] / table_rows
        if group_share < confidence or group_share - table_share < gain:
            continue
        exposed += 1
        if worst is None or group_share - table_share > worst.group_share - worst.table_share:
            worst = ConfidentInference(
                attribute=attribute,
                group=dict(zip(columns, key, strict=True)),
                value=value,
                group_share=group_share,
                table_share=table_share,
                group_rows=group_totals[key],
                exposed_groups=0,
            )
    if worst is None:
        return None
    return replace(worst, exposed_groups=exposed)
