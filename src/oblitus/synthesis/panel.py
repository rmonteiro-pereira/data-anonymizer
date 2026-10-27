"""Panels: the same entities observed over many periods, such as 40 stores over 36 months.

A schema with a `panel` generates `entities × periods` rows, ordered by entity and then by
period. The panel creates two columns of its own, the entity id (1 to n) and the period's date,
and every other column has a `level`:
- "row" columns are drawn or computed for each entity and period;
- "entity" columns are drawn once per entity and repeated over its periods, such as store size;
- "period" columns are drawn once per period and shared by every entity, such as a market shock.

This module holds the panel's settings, its row layout, the rules that keep levels consistent,
and the lag of a column in a finished table. Building the rows period by period is in
`panel_generation.py`.
"""

from collections.abc import Collection, Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oblitus.errors import SchemaError
from oblitus.synthesis.expressions import LagSource

if TYPE_CHECKING:
    from oblitus.synthesis.schema import ColumnSchema, Correlation

# The 0-based position of a row's period and entity, readable in formulas and conditions.
PERIOD_INDEX = "period"
ENTITY_INDEX = "entity_index"


class Level(StrEnum):
    """How often a column's value is drawn in a panel."""

    ROW = "row"
    ENTITY = "entity"
    PERIOD = "period"


class Panel(BaseModel):
    """The entities and periods of a panel.

    `entity` and `time` name the two columns the panel creates. `start` is the first period's
    date and `frequency` a pandas offset alias such as "MS" (month start), "W" or "D".
    """

    model_config = ConfigDict(extra="forbid")

    entity: str
    entities: int = Field(gt=0)
    time: str
    start: datetime
    periods: int = Field(gt=0)
    frequency: str = "MS"

    @field_validator("frequency")
    @classmethod
    def _known_frequency(cls, text: str) -> str:
        try:
            to_offset(text)
        except ValueError as error:
            raise ValueError(f"frequency {text!r} is not a pandas offset alias: {error}") from error
        return text

    @model_validator(mode="after")
    def _distinct_names(self) -> "Panel":
        names = [self.entity, self.time, PERIOD_INDEX, ENTITY_INDEX]
        if len(set(names)) != len(names):
            raise ValueError(
                f"the panel's entity and time columns need names different from each other "
                f"and from {PERIOD_INDEX!r} and {ENTITY_INDEX!r}, which formulas can read"
            )
        return self

    @property
    def rows(self) -> int:
        """The number of rows the panel makes."""
        return self.entities * self.periods

    def readable_names(self) -> set[str]:
        """Names formulas can read without a column of the schema."""
        return {self.entity, self.time, PERIOD_INDEX, ENTITY_INDEX}

    def dates(self) -> pd.DatetimeIndex:
        """The date of each period, in order."""
        return pd.date_range(self.start, periods=self.periods, freq=self.frequency)

    def entity_ids(self) -> np.ndarray:
        """The id of each entity, 1 to n."""
        return np.arange(1, self.entities + 1)

    def row_entity_index(self) -> np.ndarray:
        """The 0-based entity of every row of the table."""
        return np.repeat(np.arange(self.entities), self.periods)

    def row_period_index(self) -> np.ndarray:
        """The 0-based period of every row of the table."""
        return np.tile(np.arange(self.periods), self.entities)

    def positions_in_period(self, period: int) -> np.ndarray:
        """The row positions of one period, one per entity in entity order."""
        return np.arange(self.entities) * self.periods + period

    def first_test_period(self, test_share: float) -> int:
        """The first period of the test set when the last `test_share` of periods are held out."""
        test_periods = max(1, round(test_share * self.periods))
        if test_periods >= self.periods:
            raise SchemaError(
                f"a time split of {self.periods} periods with test_share {test_share} leaves "
                "no training period; add periods or lower the question's test_share"
            )
        return self.periods - test_periods

    def with_index_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return the table with the `period` and `entity_index` columns formulas can read."""
        return table.assign(
            **{PERIOD_INDEX: self.row_period_index(), ENTITY_INDEX: self.row_entity_index()}
        )

    def table_lags(self, table: pd.DataFrame) -> LagSource:
        """Return the lag of any column of a finished table, shifted within each entity."""

        def lagged(name: str, steps: int) -> np.ndarray:
            by_entity = table[name].to_numpy().reshape(self.entities, self.periods)
            shifted = blank_like(by_entity.dtype, by_entity.shape)
            if steps < self.periods:
                shifted[:, steps:] = by_entity[:, : self.periods - steps]
            return shifted.reshape(-1)

        return lagged


def blank_like(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Return an array of missing values that can hold values of `dtype`."""
    if dtype.kind in "biuf":
        return np.full(shape, np.nan)
    return np.full(shape, None, dtype=object)


def check_panel_schema(
    panel: Panel | None,
    columns: Sequence["ColumnSchema"],
    correlations: Sequence["Correlation"],
    constraints: Sequence[str],
) -> None:
    """Raise ValueError when levels, lags or names do not fit the schema's panel (or lack one)."""
    levels = {column.name: column.level for column in columns}
    if panel is None:
        _check_no_panel_features(columns)
        return
    taken = [name for name in levels if name in panel.readable_names()]
    if taken:
        raise ValueError(f"column names {taken} are made or reserved by the panel")
    if constraints:
        raise ValueError("a panel cannot have constraints: dropping rows would break its periods")
    for column in columns:
        _check_column_in_panel(column, levels, panel)
    for correlation in correlations:
        first, second = (levels[name] for name in correlation.columns)
        if first is not second:
            raise ValueError(
                f"correlation {list(correlation.columns)} joins columns of different levels"
            )


def _check_no_panel_features(columns: Sequence["ColumnSchema"]) -> None:
    for column in columns:
        if column.level is not Level.ROW:
            raise ValueError(f"{column.name}: level {column.level.value!r} needs a panel")
        if column.lagged_dependencies():
            raise ValueError(f"{column.name}: lag needs a panel")


def _check_column_in_panel(column: "ColumnSchema", levels: dict[str, Level], panel: Panel) -> None:
    if column.unique and column.level is not Level.ENTITY:
        raise ValueError(f"{column.name}: only entity-level columns can be unique in a panel")
    if column.level is Level.PERIOD and (column.exact or column.outlier_fraction > 0):
        raise ValueError(f"{column.name}: a period-level column has one value per period")
    if column.level is Level.ROW and column.exact:
        raise ValueError(
            f"{column.name}: exact would hold in every period, not over the whole panel; "
            "make the column entity-level or drop exact"
        )
    own_level = column.level
    allowed = {name for name, level in levels.items() if level is own_level}
    if own_level is Level.ENTITY:
        allowed |= {panel.entity, ENTITY_INDEX}
    elif own_level is Level.PERIOD:
        allowed |= {panel.time, PERIOD_INDEX}
    if own_level is not Level.ROW:
        _check_reads(column, allowed, own_level)
    _check_lagged_columns(column, levels)


def _check_reads(column: "ColumnSchema", allowed: Collection[str], level: Level) -> None:
    outside = sorted(column.dependencies() - set(allowed))
    if outside:
        raise ValueError(
            f"{column.name}: a {level.value}-level column can only read {level.value}-level "
            f"columns, but it reads {outside}"
        )


def _check_lagged_columns(column: "ColumnSchema", levels: dict[str, Level]) -> None:
    for name in sorted(column.lagged_dependencies()):
        lagged_level = levels.get(name)
        if lagged_level is None or lagged_level is Level.ENTITY:
            raise ValueError(
                f"{column.name}: lag({name}) needs a row-level or period-level column "
                "of the schema; an entity-level column never changes"
            )
        if column.level is Level.PERIOD and lagged_level is not Level.PERIOD:
            raise ValueError(f"{column.name}: a period-level column can only lag period-level ones")
        if column.level is Level.ENTITY:
            raise ValueError(f"{column.name}: an entity-level column cannot use lag")
