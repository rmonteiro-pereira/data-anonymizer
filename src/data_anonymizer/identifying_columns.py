"""Released columns that tell rows or entities apart on their own.

Two kinds, both counted on the whole released table:
- a fingerprint: a column with a different value in nearly every row. Anyone who holds that
  value elsewhere finds the row. A text column that a technique only partly changed (a
  masked ID, a JSON or note column with its patterns redacted) is a fingerprint too, since
  the rest of its text still differs from row to row;
- an entity key: a text or integer column that was never treated and repeats across rows
  like an account number, a card or a device. Each value ties together all the rows of one
  entity, so whoever knows one row of the entity can follow it through the whole table.
"""

from collections.abc import Mapping
from typing import Any

import narwhals as nw

from data_anonymizer.engines import WorkingFrame, first_row
from data_anonymizer.evaluation.privacy import distinct_present
from data_anonymizer.plan import AnonymizationPlan, ColumnAction
from data_anonymizer.techniques import Technique

# A column with at least this share of distinct values singles out rows.
FINGERPRINT_SHARE = 0.9
FINGERPRINT_MIN_ROWS = 20

# An entity key has thousands of distinct values, each on several rows. A product, city or
# category code has hundreds at most, and a value on one row is the fingerprint case above.
ENTITY_KEY_MIN_DISTINCT = 1000


def fingerprint_columns(
    plan: AnonymizationPlan, techniques: Mapping[str, Technique], released: WorkingFrame
) -> list[str]:
    """Describe every column whose released values are nearly all different."""
    schema = released.collect_schema()
    candidates = [
        column
        for column, dtype in schema.items()
        if _can_fingerprint(plan, techniques, column, dtype)
    ]
    if not candidates:
        return []
    present, distinct = _present_and_distinct(released, candidates)
    problems = []
    for column in candidates:
        if present[column] < FINGERPRINT_MIN_ROWS:
            continue
        unique_share = distinct[column] / present[column]
        if unique_share >= FINGERPRINT_SHARE:
            problems.append(_fingerprint_problem(plan, column, unique_share))
    return problems


def entity_key_columns(plan: AnonymizationPlan, released: WorkingFrame) -> list[str]:
    """Describe every untreated column that repeats across rows like an account key."""
    schema = released.collect_schema()
    candidates = [
        column for column, dtype in schema.items() if _can_be_entity_key(plan, column, dtype)
    ]
    if not candidates:
        return []
    present, distinct = _present_and_distinct(released, candidates)
    problems = []
    for column in candidates:
        if present[column] < FINGERPRINT_MIN_ROWS or distinct[column] < ENTITY_KEY_MIN_DISTINCT:
            continue
        if distinct[column] / present[column] >= FINGERPRINT_SHARE:
            continue
        rows_per_value = present[column] / distinct[column]
        problems.append(
            f"{column} was not treated and holds {distinct[column]:,} distinct values on "
            f"{rows_per_value:.1f} rows each, like an account or entity key; it links every "
            "row of an entity, so replace it or list it in the plan"
        )
    return problems


def _can_fingerprint(
    plan: AnonymizationPlan, techniques: Mapping[str, Technique], column: str, dtype: Any
) -> bool:
    action = plan.action_for(column)
    if action is None or action.technique == "keep":
        return column not in plan.direct_identifiers
    if column in plan.direct_identifiers and action.technique != "redact_entities":
        # Its technique either replaces the values or is reported as an untreated identifier.
        return False
    return dtype == nw.String and _leaves_text_distinct(action, techniques)


def _leaves_text_distinct(action: ColumnAction, techniques: Mapping[str, Technique]) -> bool:
    if action.technique == "redact_entities":
        return True
    return not techniques[action.technique].replaces_identifiers


def _can_be_entity_key(plan: AnonymizationPlan, column: str, dtype: Any) -> bool:
    action = plan.action_for(column)
    if action is not None and action.technique != "keep":
        return False
    listed = {*plan.direct_identifiers, *plan.quasi_identifiers, *plan.sensitive_attributes}
    if column in listed:
        return False
    # Decimals and timestamps are measurements, which many rows share only by chance.
    return dtype == nw.String or dtype == nw.Categorical or dtype.is_integer()


def _present_and_distinct(
    released: WorkingFrame, columns: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    present = first_row(released.select(*[nw.col(c).count().alias(c) for c in columns]))
    distinct = first_row(released.select(*[distinct_present(c).alias(c) for c in columns]))
    return present, distinct


def _fingerprint_problem(plan: AnonymizationPlan, column: str, unique_share: float) -> str:
    action = plan.action_for(column)
    if action is None or action.technique == "keep":
        return (
            f"{column} is released unchanged and {unique_share:.0%} of its values are "
            "distinct, so it can single out rows"
        )
    return (
        f"{column} is still {unique_share:.0%} distinct after {action.technique}, which does "
        "not replace its values, so it can single out rows"
    )
