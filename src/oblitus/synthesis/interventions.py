"""Set a column for every row and replay the rest of the build: the do-operator.

`intervene` keeps every column the treatment does not cause, sets the treatment, and
recomputes its descendants in generation order from the random numbers stored with the rows.
A row therefore keeps its own noise, and the difference between two interventions is the
exact effect on that row.

The result describes the structural model:
- outliers and blanked cells are not applied again;
- a descendant that is neither drawn nor computed, such as a fake name, keeps its value;
- `exact` statistics are not restored on the recomputed columns.
"""

from typing import Any

import pandas as pd

from oblitus.errors import SchemaError
from oblitus.synthesis.causal_graph import CausalGraph
from oblitus.synthesis.draw_replay import ColumnDraws, ReplayableTable
from oblitus.synthesis.patterns import (
    apply_conditions,
    compute_formula,
    conditional_uniform,
    draw,
)
from oblitus.synthesis.schema import ColumnSchema, DatasetSchema


def intervene(
    schema: DatasetSchema, built: ReplayableTable, treatment: str, value: Any
) -> pd.DataFrame:
    """Return every row as it would be if `treatment` were `value` for all of them."""
    frame = built.structural.copy()
    frame[treatment] = pd.Series(value, index=frame.index).astype(frame[treatment].dtype)
    ledger = built.ledger()
    descendants = CausalGraph.of(schema).descendants(treatment)
    for name in schema.generation_order():
        if name not in descendants:
            continue
        column = schema.column(name)
        draws = ledger.column(name)
        if column.formula is None and draws.base() is None:
            continue
        frame[name] = _replayed_values(column, frame, draws).to_numpy()
    return frame


def _replayed_values(column: ColumnSchema, frame: pd.DataFrame, draws: ColumnDraws) -> pd.Series:
    if column.formula is not None:
        values = compute_formula(column, frame, draws)
    else:
        base_uniform = draws.base()
        assert base_uniform is not None
        values = draw(column, base_uniform)
    uniform = conditional_uniform(column, draws)
    return apply_conditions(column, values, frame, uniform, _cannot_regenerate)


def _cannot_regenerate(settings: ColumnSchema, count: int) -> pd.Series:
    raise SchemaError(
        f"{settings.name}: a condition on a computed column of this kind has no stored "
        "random numbers to replay"
    )
