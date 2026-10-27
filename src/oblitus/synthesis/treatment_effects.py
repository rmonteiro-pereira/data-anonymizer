"""The `causal` section of a challenge's answer key: the true effects of the treatment.

The effects come from replaying the build twice, once with every row control and once with every
row treated, so they hold exactly for the rows that were generated. The naive difference is what
a participant sees by comparing the groups in the published training rows, and its bias is what
confounding put there.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from oblitus.synthesis.causal_graph import CausalGraph
from oblitus.synthesis.draw_replay import ReplayableTable
from oblitus.synthesis.interventions import intervene
from oblitus.synthesis.schema import (
    ColumnSchema,
    DatasetSchema,
    Question,
    treatment_levels,
)
from oblitus.taxonomy import NUMERIC_KINDS

# Numeric effect modifiers are reported by quartile, unless they have few distinct values.
_QUANTILE_BINS = 4
_MAX_VALUES_REPORTED_ONE_BY_ONE = 10

OutcomeAsNumbers = Callable[[pd.Series], np.ndarray]


def causal_answer(
    schema: DatasetSchema,
    question: Question,
    built: ReplayableTable,
    train: pd.DataFrame,
    outcome_as_numbers: OutcomeAsNumbers,
) -> dict[str, Any]:
    """Return the true effects, the naive estimate and its bias, and the role of each column."""
    assert question.treatment is not None
    treatment = schema.column(question.treatment)
    control, treated = treatment_levels(question, treatment)
    under_control = intervene(schema, built, treatment.name, control)
    under_treatment = intervene(schema, built, treatment.name, treated)
    effects = outcome_as_numbers(under_treatment[question.target]) - outcome_as_numbers(
        under_control[question.target]
    )
    was_treated = (built.structural[treatment.name] == treated).to_numpy(dtype=bool)

    answer: dict[str, Any] = {
        "treatment": treatment.name,
        "treatment_values": {"control": control, "treated": treated},
        "outcome": question.target,
        "ate": float(effects.mean()),
        "att": _mean_or_none(effects[was_treated]),
    }
    if question.truth_column is not None:
        answer["effect_on_truth_column"] = _truth_effects(
            question.truth_column, under_control, under_treatment, was_treated
        )
    answer["cate"] = {
        name: _conditional_effects(schema.column(name), built.structural[name], effects)
        for name in question.effect_modifiers
    }
    naive = _naive_difference(train, question, (control, treated), outcome_as_numbers)
    answer["naive_difference"] = naive
    answer["naive_bias"] = None if naive is None else naive - answer["ate"]
    answer.update(_roles(schema, question, train))
    return answer


def _truth_effects(
    truth_column: str,
    under_control: pd.DataFrame,
    under_treatment: pd.DataFrame,
    was_treated: np.ndarray,
) -> dict[str, Any]:
    difference = under_treatment[truth_column].to_numpy(dtype=float) - under_control[
        truth_column
    ].to_numpy(dtype=float)
    return {
        "column": truth_column,
        "ate": float(difference.mean()),
        "att": _mean_or_none(difference[was_treated]),
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def _conditional_effects(
    column: ColumnSchema, values: pd.Series, effects: np.ndarray
) -> list[dict[str, Any]]:
    """Mean effect per level of a modifier, per quartile of a numeric one with many values."""
    levels: Any
    distinct = values.nunique()
    if column.kind in NUMERIC_KINDS and distinct > _MAX_VALUES_REPORTED_ONE_BY_ONE:
        levels = pd.qcut(values, _QUANTILE_BINS, duplicates="drop")
    elif column.kind in NUMERIC_KINDS:
        levels = pd.Categorical(values, categories=sorted(values.dropna().unique()))
    else:
        labels = [str(choice) for choice in column.choices or sorted(values.astype(str).unique())]
        levels = pd.Categorical(values.astype(str), categories=labels)
    grouped = pd.DataFrame({"level": levels, "effect": effects}).groupby("level", observed=True)
    return [
        {"level": str(level), "rows": int(len(group)), "effect": float(group["effect"].mean())}
        for level, group in grouped
    ]


def _naive_difference(
    train: pd.DataFrame,
    question: Question,
    levels: tuple[Any, Any],
    outcome_as_numbers: OutcomeAsNumbers,
) -> float | None:
    """Mean outcome of the treated minus the control, over the published training rows."""
    assert question.treatment is not None
    control, treated = levels
    outcome = outcome_as_numbers(train[question.target])
    has_outcome = ~np.isnan(outcome)
    given = train[question.treatment]
    in_treated = given.eq(treated).fillna(False).to_numpy(dtype=bool) & has_outcome
    in_control = given.eq(control).fillna(False).to_numpy(dtype=bool) & has_outcome
    if not in_treated.any() or not in_control.any():
        return None
    return float(outcome[in_treated].mean() - outcome[in_control].mean())


def _roles(schema: DatasetSchema, question: Question, train: pd.DataFrame) -> dict[str, Any]:
    assert question.treatment is not None
    graph = CausalGraph.of(schema)
    roles = graph.roles(question.treatment, question.target)
    adjustment = graph.adjustment_set(question.treatment, question.target, list(train.columns))
    return {
        "roles": {
            "confounders": roles.confounders,
            "mediators": roles.mediators,
            "colliders": roles.colliders,
            "instruments": roles.instruments,
            "effect_modifiers": list(question.effect_modifiers),
        },
        "adjustment_set": adjustment.columns,
        "adjustment_set_reason": adjustment.reason,
    }
