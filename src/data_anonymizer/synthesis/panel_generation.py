"""Build a panel table one period at a time.

A column that reads `lag(...)` needs the earlier periods of its own entity, and an
autoregressive column needs its own earlier value. So the rows are built period by period,
with all entities of a period computed together as one vectorised table:
1. entity-level columns are drawn once, for every entity;
2. for each period, period-level columns are drawn or computed for that one period, and then
   row-level columns for every entity, in dependency order;
3. the periods are stacked into a table ordered by entity and then by period, and cells are
   blanked at each column's null fraction.

The random numbers of each period are the window of one stored array per slot onto the
period's rows, the same way `draw_replay` hands them out for a table built in one pass.
A panel keeps no stored numbers in its result: causal questions are refused on panels, and
nothing else replays a build.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_anonymizer.errors import SchemaError
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.column_values import build_column, draw_uniforms
from data_anonymizer.synthesis.draw_replay import DrawLedger, ReplayableTable
from data_anonymizer.synthesis.expressions import LagSource
from data_anonymizer.synthesis.panel import ENTITY_INDEX, PERIOD_INDEX, Level
from data_anonymizer.synthesis.patterns import apply_gaps
from data_anonymizer.synthesis.schema import ColumnSchema, DatasetSchema


@dataclass(frozen=True, slots=True)
class SchemaSwitch:
    """A schema that takes over from one period on, such as the shifted test periods.

    Entities and their series carry on across the switch, so an autoregressive column keeps
    the value it had in the last period before it.
    """

    schema: DatasetSchema
    from_period: int


def generate_panel(
    schema: DatasetSchema,
    rows: int,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
    switch: SchemaSwitch | None = None,
) -> ReplayableTable:
    """Generate the `entities × periods` rows of a schema's panel."""
    panel = schema.panel
    assert panel is not None
    if rows != panel.rows:
        raise SchemaError(
            f"the panel has {panel.entities} entities × {panel.periods} periods = {panel.rows} "
            f"rows, so {rows} rows cannot be generated; leave the row count out"
        )
    structural = _PanelBuilder(schema, rng, surrogates).build(switch)
    table = _published_table(schema, structural, switch, rng)
    return ReplayableTable(table=table, structural=structural, draws={})


class _PanelBuilder:
    """The state of one panel build: stored numbers, entity values and the history of columns."""

    def __init__(
        self, schema: DatasetSchema, rng: np.random.Generator, surrogates: SurrogateFactory
    ) -> None:
        assert schema.panel is not None
        self._schema = schema
        self._panel = schema.panel
        self._rng = rng
        self._surrogates = surrogates
        self._dates = schema.panel.dates().to_numpy()
        self._entity_values = self._build_entity_level()
        self._period_ledger = self._ledger(Level.PERIOD, self._panel.periods)
        self._row_ledger = self._ledger(Level.ROW, self._panel.rows)
        # Each column's values per period, in period order, for lags and for stacking.
        self._history: dict[str, list[np.ndarray]] = {}

    def build(self, switch: SchemaSwitch | None) -> pd.DataFrame:
        """Build every period and return the stacked table, with the helper columns."""
        for period in range(self._panel.periods):
            in_force = self._schema
            if switch is not None and period >= switch.from_period:
                in_force = switch.schema
            self._build_period(period, in_force)
        return self._stacked()

    def _ledger(self, level: Level, count: int) -> DrawLedger:
        columns = [column for column in self._schema.columns if column.level is level]
        names = {column.name for column in columns}
        correlations = [
            correlation
            for correlation in self._schema.correlations
            if correlation.columns[0] in names
        ]
        return DrawLedger.recording(
            count, self._rng, draw_uniforms(columns, correlations, count, self._rng)
        )

    def _build_entity_level(self) -> dict[str, np.ndarray]:
        entities = self._panel.entities
        frame = pd.DataFrame(
            {self._panel.entity: self._panel.entity_ids(), ENTITY_INDEX: np.arange(entities)}
        )
        ledger = self._ledger(Level.ENTITY, entities)
        for name in self._schema.generation_order():
            column = self._schema.column(name)
            if column.level is Level.ENTITY:
                values = build_column(
                    column, frame, ledger.column(name), self._rng, self._surrogates
                )
                frame[name] = values.to_numpy()
        names = [column.name for column in self._schema.columns if column.level is Level.ENTITY]
        return {name: frame[name].to_numpy() for name in names}

    def _build_period(self, period: int, in_force: DatasetSchema) -> None:
        order = in_force.generation_order()
        period_values = self._build_period_level(period, in_force, order)
        frame = self._period_frame(period, period_values)
        window = self._row_ledger.subset(self._panel.positions_in_period(period))
        lags = self._lags(period, self._panel.entities)
        for name in order:
            column = in_force.column(name)
            if column.level is Level.ROW:
                values = build_column(
                    column, frame, window.column(name), self._rng, self._surrogates, lags=lags
                )
                frame[name] = values.to_numpy()
                self._history.setdefault(name, []).append(frame[name].to_numpy())

    def _build_period_level(
        self, period: int, in_force: DatasetSchema, order: list[str]
    ) -> dict[str, np.ndarray]:
        """Build the period-level columns on a table of the period's single row."""
        frame = pd.DataFrame({self._panel.time: self._dates[period : period + 1]})
        frame[PERIOD_INDEX] = period
        window = self._period_ledger.subset(np.array([period]))
        lags = self._lags(period, 1)
        values: dict[str, np.ndarray] = {}
        for name in order:
            column = in_force.column(name)
            if column.level is Level.PERIOD:
                built = build_column(
                    column, frame, window.column(name), self._rng, self._surrogates, lags=lags
                )
                frame[name] = built.to_numpy()
                values[name] = frame[name].to_numpy()
                self._history.setdefault(name, []).append(values[name])
        return values

    def _period_frame(self, period: int, period_values: dict[str, np.ndarray]) -> pd.DataFrame:
        """The table of one period: a row per entity, holding everything but row-level columns."""
        entities = self._panel.entities
        columns: dict[str, np.ndarray] = {
            self._panel.entity: self._panel.entity_ids(),
            self._panel.time: np.repeat(self._dates[period : period + 1], entities),
            PERIOD_INDEX: np.full(entities, period),
            ENTITY_INDEX: np.arange(entities),
            **self._entity_values,
        }
        for name, values in period_values.items():
            columns[name] = np.repeat(values, entities)
        return pd.DataFrame(columns)

    def _lags(self, period: int, rows: int) -> LagSource:
        def lagged(name: str, steps: int) -> np.ndarray:
            earlier = period - steps
            if earlier < 0:
                return np.full(rows, np.nan)
            return np.broadcast_to(self._history[name][earlier], (rows,))

        return lagged

    def _stacked(self) -> pd.DataFrame:
        """Stack the periods into rows ordered by entity and then by period."""
        panel = self._panel
        stacked: dict[str, np.ndarray] = {
            panel.entity: np.repeat(panel.entity_ids(), panel.periods),
            panel.time: np.tile(self._dates, panel.entities),
            PERIOD_INDEX: panel.row_period_index(),
            ENTITY_INDEX: panel.row_entity_index(),
        }
        for column in self._schema.columns:
            stacked[column.name] = self._over_rows(column)
        return pd.DataFrame(stacked)

    def _over_rows(self, column: ColumnSchema) -> np.ndarray:
        panel = self._panel
        if column.level is Level.ENTITY:
            return np.repeat(self._entity_values[column.name], panel.periods)
        periods = self._history[column.name]
        if column.level is Level.PERIOD:
            return np.tile(np.concatenate(periods), panel.entities)
        return np.column_stack(periods).reshape(-1)


def _published_table(
    schema: DatasetSchema,
    structural: pd.DataFrame,
    switch: SchemaSwitch | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Blank cells at each column's null fraction and drop the helper columns."""
    panel = schema.panel
    assert panel is not None
    gapped = {panel.entity: structural[panel.entity], panel.time: structural[panel.time]}
    for column in schema.columns:
        values = structural[column.name]
        later = None if switch is None else switch.schema.column(column.name)
        if later is None or later == column:
            gapped[column.name] = apply_gaps(column, values, structural, rng)
        else:
            assert switch is not None
            after = (structural[PERIOD_INDEX] >= switch.from_period).to_numpy()
            gapped[column.name] = _gapped_before_and_after(
                (column, later), values, structural, after, rng
            )
    return pd.DataFrame(gapped)


def _gapped_before_and_after(
    settings: tuple[ColumnSchema, ColumnSchema],
    values: pd.Series,
    structural: pd.DataFrame,
    after: np.ndarray,
    rng: np.random.Generator,
) -> pd.Series:
    """Blank cells with the first settings before the switch and the second after it."""
    parts = [
        apply_gaps(column, values[rows], structural[rows], rng)
        for column, rows in zip(settings, (~after, after), strict=True)
    ]
    return pd.concat(parts).sort_index()
