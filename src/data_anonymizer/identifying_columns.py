"""Released columns that tell rows or entities apart on their own.

Two kinds, both counted on the whole released table:
- a fingerprint: a column with a different value in nearly every row. Anyone who holds that
  value elsewhere finds the row. A text column that a technique only partly changed (a
  masked ID, a JSON or note column with its patterns redacted) is a fingerprint too, since
  the rest of its text still differs from row to row;
- an entity key: a text or whole-number column, or a pair of them, that was never treated and
  repeats across rows like an account number, a card or a device. Each value ties together all
  the rows of one entity, so whoever knows one row of the entity can follow it through the
  whole table.
"""

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import narwhals as nw

from data_anonymizer.engines import WorkingFrame, first_row
from data_anonymizer.evaluation.privacy import distinct_present
from data_anonymizer.plan import AnonymizationPlan, ColumnAction
from data_anonymizer.techniques import Technique

# A column with at least this share of distinct values singles out rows.
FINGERPRINT_SHARE = 0.9
FINGERPRINT_MIN_ROWS = 20

# A discrete column with this many distinct values, each on several rows, can act as an entity
# key: whoever knows their own value finds their rows. Statistics cannot tell an account number
# from a product code, so every such column that the plan leaves alone counts.
DISCRETE_MIN_DISTINCT = 20

# A decimal column counts as discrete when this share of its present cells are whole numbers, so
# one stray fractional cell does not hide an account column.
WHOLE_NUMBER_SHARE = 0.99

# Two columns can be a key only together. Pairs are tried among the untreated discrete columns
# with the most distinct values, to keep the cost bounded. A pair is key-like when it holds at
# least this many distinct pairs and the typical pair covers at most this many rows.
PAIR_MAX_COLUMNS = 10
PAIR_MIN_JOINT_VALUES = 100
PAIR_MAX_MEDIAN_ROWS = 10


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
    """Describe every untreated discrete column, or pair of them, that can act as an entity key."""
    columns = untreated_discrete_columns(plan, released)
    if not columns:
        return []
    present, distinct = _present_and_distinct(released, columns)
    problems = [
        f"{column} was not treated and holds {distinct[column]:,} distinct values on "
        f"{present[column] / distinct[column]:.1f} rows each; it can link rows to anyone who "
        "knows their own value, so replace it or list it in the plan"
        for column in columns
    ]
    return problems + _key_pairs(released, columns, distinct)


def untreated_discrete_columns(plan: AnonymizationPlan, frame: WorkingFrame) -> list[str]:
    """Name the untreated columns whose repeated values group rows like an account number.

    A column is discrete when it is text, a category, an integer, or a decimal column that holds
    whole numbers in nearly every present cell. It needs at least 20 distinct values, and not
    nearly one per row, which is the fingerprint case. Fractional decimals and timestamps are
    measurements, which many rows share only by chance.
    """
    schema = frame.collect_schema()
    candidates = [
        column for column, dtype in schema.items() if _can_be_entity_key(plan, column, dtype)
    ]
    if not candidates:
        return []
    present, distinct = _present_and_distinct(frame, candidates)
    columns = []
    for column in candidates:
        if present[column] < FINGERPRINT_MIN_ROWS or distinct[column] < DISCRETE_MIN_DISTINCT:
            continue
        if distinct[column] / present[column] >= FINGERPRINT_SHARE:
            continue
        if schema[column].is_float() and _whole_number_share(frame, column) < WHOLE_NUMBER_SHARE:
            continue
        columns.append(column)
    return columns


def _key_pairs(
    released: WorkingFrame, columns: list[str], distinct: Mapping[str, Any]
) -> list[str]:
    """Describe pairs of untreated columns whose values are unique only together."""
    widest = sorted(columns, key=lambda column: distinct[column], reverse=True)
    problems = []
    for first, second in combinations(widest[:PAIR_MAX_COLUMNS], 2):
        joint_values, median_rows = _joint_value_counts(released, first, second)
        if joint_values >= PAIR_MIN_JOINT_VALUES and median_rows <= PAIR_MAX_MEDIAN_ROWS:
            problems.append(
                f"{first} and {second} were not treated and together hold {joint_values:,} "
                f"distinct value pairs on {median_rows:.0f} rows each; the pair can link rows to "
                "anyone who knows their own values, so replace them or list them in the plan"
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
    # Decimals are checked for whole numbers later; timestamps are measurements.
    return dtype == nw.String or dtype == nw.Categorical or dtype.is_integer() or dtype.is_float()


def _whole_number_share(frame: WorkingFrame, column: str) -> float:
    reading = nw.col(column)
    whole = (reading.round(0) == reading).cast(nw.Float64).mean().alias(column)
    return float(first_row(frame.select(whole))[column])


def _joint_value_counts(frame: WorkingFrame, first: str, second: str) -> tuple[int, float]:
    """Return how many distinct value pairs two columns hold, and the typical rows per pair."""
    both_present = ~nw.col(first).is_null() & ~nw.col(second).is_null()
    rows_per_pair = frame.filter(both_present).group_by(first, second).agg(nw.len().alias("rows"))
    summary = rows_per_pair.select(
        nw.col("rows").len().alias("pairs"), nw.col("rows").median().alias("median_rows")
    )
    counts = first_row(summary)
    return int(counts["pairs"]), float(counts["median_rows"] or 0.0)


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
