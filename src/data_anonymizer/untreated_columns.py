"""Proposal steps that read the table's values to treat columns the detector left alone.

The detector judges a column by its name and a sample of its values. Under anonymization,
three kinds of column it leaves untreated would still single out rows, so the proposal
handles them after reading the whole table:
- a text column that groups rows like an account or card number gets fresh codes;
- a timestamp or number with a different value in nearly every row is generalized: to the
  day and three significant figures, which keep its distribution, or to the year and quintile
  bands under privacy fidelity;
- a column with few distinct values joins the quasi-identifiers, so k-anonymity covers the
  same columns the protection report counts when it looks for unique rows.

Pseudonymization plans pass through unchanged, since the key holder may need the values.
"""

import math
from typing import Any

import narwhals as nw

from data_anonymizer.engines import WorkingFrame, first_row
from data_anonymizer.evaluation.privacy import distinct_present
from data_anonymizer.identifying_columns import (
    FINGERPRINT_MIN_ROWS,
    FINGERPRINT_SHARE,
    untreated_discrete_columns,
)
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.protection import DISCRETE_LIMIT

# A fingerprint number keeps three significant figures, which leaves its distribution intact;
# privacy fidelity puts it in quintile bands instead, few enough for k-anonymity to cover.
_STATISTICAL_SIGNIFICANT_FIGURES = 3
_PRIVACY_BINS = 5
# The unit a fingerprint timestamp keeps.
_DATE_UNIT = {Fidelity.STATISTICAL: "day", Fidelity.PRIVACY: "year"}


def treat_untreated_columns(plan: AnonymizationPlan, frame: WorkingFrame) -> AnonymizationPlan:
    """Complete an anonymization plan with the columns only the values show to be risky."""
    if plan.protection is not Protection.ANONYMIZATION:
        return plan
    plan = _replace_untreated_keys(plan, frame)
    plan = _generalize_fingerprints(plan, frame)
    return _cover_discrete_columns(plan, frame)


def _replace_untreated_keys(plan: AnonymizationPlan, frame: WorkingFrame) -> AnonymizationPlan:
    # Nobody can tell an account key from a product code by its statistics. Fresh codes keep
    # the grouping for a model while nobody can look up their own value. A whole-number column
    # may be a measurement whose distribution must survive, so it is left to the report.
    schema = frame.collect_schema()
    columns = [
        column
        for column in untreated_discrete_columns(plan, frame)
        if plan.action_for(column) is None and _holds_text(schema[column])
    ]
    actions = [
        ColumnAction(
            column=column,
            technique="scramble_categories",
            params={"prefix": "KEY"},
            reason="untreated discrete column: fresh code per release; rows stay grouped",
        )
        for column in columns
    ]
    return _with_actions(plan, actions)


def _generalize_fingerprints(plan: AnonymizationPlan, frame: WorkingFrame) -> AnonymizationPlan:
    schema = frame.collect_schema()
    candidates = [
        column
        for column, dtype in schema.items()
        if plan.action_for(column) is None
        and column not in plan.direct_identifiers
        and (dtype.is_numeric() or dtype in (nw.Datetime, nw.Date))
    ]
    if not candidates:
        return plan
    present = first_row(frame.select(*[nw.col(c).count().alias(c) for c in candidates]))
    distinct = first_row(frame.select(*[distinct_present(c).alias(c) for c in candidates]))
    actions = []
    for column in candidates:
        rows = present[column]
        if rows < FINGERPRINT_MIN_ROWS or distinct[column] / rows < FINGERPRINT_SHARE:
            continue
        actions.append(_generalizing_action(plan, frame, column, schema[column]))
    generalized = [action for action in actions if action is not None]
    plan = _with_actions(plan, generalized)
    if plan.fidelity is not Fidelity.PRIVACY:
        return plan
    # Years and quintile bands leave few distinct values, so these columns now single
    # out rows together with the other quasi-identifiers and k-anonymity must cover them.
    added = [action.column for action in generalized]
    return plan.model_copy(update={"quasi_identifiers": [*plan.quasi_identifiers, *added]})


def _generalizing_action(
    plan: AnonymizationPlan, frame: WorkingFrame, column: str, dtype: Any
) -> ColumnAction | None:
    if dtype in (nw.Datetime, nw.Date):
        unit = _DATE_UNIT[plan.fidelity]
        return ColumnAction(
            column=column,
            technique="truncate_date",
            params={"unit": unit},
            reason=f"distinct in nearly every row: cut to the {unit}",
        )
    if plan.fidelity is Fidelity.PRIVACY:
        return ColumnAction(
            column=column,
            technique="bin_numeric",
            params={"bins": _PRIVACY_BINS, "strategy": "quantile"},
            reason="distinct in nearly every row: quintile bands",
        )
    typical = first_row(frame.select(nw.col(column).abs().median().alias("typical")))["typical"]
    if typical is None or typical == 0:
        return None
    figures = _STATISTICAL_SIGNIFICANT_FIGURES
    decimals = figures - 1 - math.floor(math.log10(typical))
    return ColumnAction(
        column=column,
        technique="round_numeric",
        params={"decimals": decimals},
        reason=f"distinct in nearly every row: {figures} significant figures kept",
    )


def _cover_discrete_columns(plan: AnonymizationPlan, frame: WorkingFrame) -> AnonymizationPlan:
    excluded = {*plan.direct_identifiers, *plan.sensitive_attributes, *plan.quasi_identifiers}
    if plan.entity_column is not None:
        excluded.add(plan.entity_column)
    candidates = [column for column in frame.collect_schema().names() if column not in excluded]
    if not candidates:
        return plan
    distinct = first_row(frame.select(*[nw.col(c).n_unique().alias(c) for c in candidates]))
    discrete = [column for column in candidates if distinct[column] <= DISCRETE_LIMIT]
    if not discrete:
        return plan
    return plan.model_copy(update={"quasi_identifiers": [*plan.quasi_identifiers, *discrete]})


def _holds_text(dtype: Any) -> bool:
    return dtype == nw.String or dtype == nw.Categorical


def _with_actions(plan: AnonymizationPlan, actions: list[ColumnAction]) -> AnonymizationPlan:
    if not actions:
        return plan
    return plan.model_copy(update={"actions": [*plan.actions, *actions]})
