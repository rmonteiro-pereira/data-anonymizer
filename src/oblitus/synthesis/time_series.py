"""The `time_series` section of a challenge's answer key: how the panel was laid out and built.

It names the entity and time columns, how many periods each split holds, which columns read
earlier periods of which others, and the conditions that change a column at some period.
"""

from typing import Any

import numpy as np

from oblitus.synthesis.expressions import referenced_columns
from oblitus.synthesis.panel import PERIOD_INDEX
from oblitus.synthesis.schema import DatasetSchema, Question, Split


def describe_time_series(
    schema: DatasetSchema, question: Question, in_test: np.ndarray
) -> dict[str, Any]:
    """Return the answer key's time series section for a schema with a panel."""
    panel = schema.panel
    assert panel is not None
    period_of_row = panel.row_period_index()
    train_periods = np.unique(period_of_row[~in_test])
    test_periods = np.unique(period_of_row[in_test])
    dates = panel.dates()
    return {
        "entity": panel.entity,
        "time": panel.time,
        "entities": panel.entities,
        "periods": panel.periods,
        "start": panel.start.isoformat(),
        "frequency": panel.frequency,
        "split": question.split.value,
        "periods_per_split": {"train": len(train_periods), "test": len(test_periods)},
        "first_test_period": _first_test_period(question, test_periods, dates),
        "lags": _lags_used(schema),
        "regimes": _regimes(schema),
    }


def _first_test_period(question: Question, test_periods: np.ndarray, dates: Any) -> Any:
    """The first held-out period of a time split, or None when the test rows are random."""
    if question.split is not Split.TIME:
        return None
    first = int(test_periods.min())
    return {"period": first, "date": dates[first].isoformat()}


def _lags_used(schema: DatasetSchema) -> dict[str, dict[str, list[int]]]:
    """Per column, the columns its formula reads from earlier periods and by how many."""
    return {
        column.name: {name: sorted(steps) for name, steps in sorted(column.lags().items())}
        for column in schema.columns
        if column.lags()
    }


def _regimes(schema: DatasetSchema) -> list[dict[str, Any]]:
    """The conditions that test the period or the date, which switch a column's behaviour."""
    panel = schema.panel
    assert panel is not None
    by_time = {PERIOD_INDEX, panel.time}
    regimes = []
    for column in schema.columns:
        for override in column.conditions:
            if referenced_columns(override.when) & by_time:
                changes = override.model_dump(mode="json", exclude_none=True, exclude={"when"})
                regimes.append({"column": column.name, "when": override.when, "changes": changes})
    return regimes
