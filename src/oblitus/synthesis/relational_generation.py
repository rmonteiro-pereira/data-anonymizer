"""Build the tables of a relational schema, parents first, with every foreign key valid.

For one table the steps are:
1. build the columns that read nothing from the children, given the parent's values per row;
2. for each child table, draw how many rows every row gets, build the child with its parent's
   values repeated per child row, and summarise the child rows into the declared aggregates;
3. build the columns that read an aggregate.

Child counts come from stored-style uniform numbers through the count's inverse distribution
function, and each table's columns go through `generate_with_replay`, so every random number
of a table is a uniform number the generator could store. Replaying a build across tables is
not supported, and the schema refuses the questions that would need it.

Formulas and aggregates see the true values, before blank cells are cut, as in a single table.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oblitus.errors import SchemaError
from oblitus.surrogates import SurrogateFactory
from oblitus.synthesis.expressions import evaluate
from oblitus.synthesis.generator import generate_with_replay
from oblitus.synthesis.marginals import quantile_function
from oblitus.synthesis.relational import (
    AggregateFunction,
    ChildAggregate,
    RelatedTable,
    RelationalSchema,
)


@dataclass(frozen=True, slots=True)
class GeneratedTables:
    """The published tables, and each table's aggregates of its children, one row per row."""

    tables: dict[str, pd.DataFrame]
    aggregates: dict[str, pd.DataFrame]


@dataclass(slots=True)
class _Results:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    aggregates: dict[str, pd.DataFrame] = field(default_factory=dict)


def generate_tables(
    schema: RelationalSchema,
    rows: int,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> GeneratedTables:
    """Generate every table; the root has `rows` rows and the others follow from it."""
    results = _Results()
    builder = _TableBuilder(schema, rng, surrogates, results)
    builder.build(schema.tables[0], rows, parent_context=None, parent_keys=None)
    return GeneratedTables(results.tables, results.aggregates)


class _TableBuilder:
    def __init__(
        self,
        schema: RelationalSchema,
        rng: np.random.Generator,
        surrogates: SurrogateFactory,
        results: _Results,
    ) -> None:
        self._schema = schema
        self._rng = rng
        self._surrogates = surrogates
        self._results = results

    def build(
        self,
        table: RelatedTable,
        rows: int,
        parent_context: pd.DataFrame | None,
        parent_keys: np.ndarray | None,
    ) -> pd.DataFrame:
        """Build one table and its children; return its true values for the parent's aggregates."""
        schema = self._schema
        first = generate_with_replay(
            schema.first_schema(table),
            rows,
            rng=self._rng,
            surrogates=self._surrogates,
            context=parent_context,
        )
        first_names = [column.name for column in schema.first_schema(table).columns]
        first_structural = first.structural[first_names]
        aggregates = self._build_children(table, rows, first_structural)
        later_schema = schema.later_schema(table)
        parts = [(first.table, first_structural)]
        if later_schema is not None:
            context = _later_context(parent_context, first_structural, aggregates)
            later = generate_with_replay(
                later_schema, rows, rng=self._rng, surrogates=self._surrogates, context=context
            )
            later_names = [column.name for column in later_schema.columns]
            parts.append((later.table, later.structural[later_names]))
        published, structural = _joined(table, parts, parent_keys)
        self._results.tables[table.name] = published
        self._results.aggregates[table.name] = aggregates
        return structural

    def _build_children(
        self, table: RelatedTable, rows: int, structural: pd.DataFrame
    ) -> pd.DataFrame:
        aggregates = pd.DataFrame(index=range(rows))
        keys = np.arange(1, rows + 1)
        for child in self._schema.children_of(table.name):
            counts = self._child_counts(child, rows)
            positions = np.repeat(np.arange(rows), counts)
            if len(positions) == 0:
                raise SchemaError(
                    f"{child.name}: rows_per_parent gave no rows at all for {rows} "
                    f"{table.name} rows; raise its mean or ask for more rows"
                )
            context = self._parent_context(table, structural, positions)
            child_structural = self.build(child, len(positions), context, keys[positions])
            for aggregate in table.aggregates:
                if aggregate.table == child.name:
                    aggregates[aggregate.name] = _aggregate(
                        aggregate, child_structural, positions, rows
                    )
        return aggregates

    def _child_counts(self, child: RelatedTable, parents: int) -> np.ndarray:
        assert child.rows_per_parent is not None
        inverse = quantile_function(child.rows_per_parent.as_column())
        return inverse(self._rng.random(parents)).to_numpy(dtype=np.int64)

    def _parent_context(
        self, table: RelatedTable, structural: pd.DataFrame, positions: np.ndarray
    ) -> pd.DataFrame:
        """Repeat the parent's true values once per child row, under the names children read."""
        later = table.later_columns()
        repeated = {
            f"{table.name}_{column.name}": structural[column.name].to_numpy()[positions]
            for column in table.columns
            if column.name not in later
        }
        return pd.DataFrame(repeated)


def _later_context(
    parent_context: pd.DataFrame | None, structural: pd.DataFrame, aggregates: pd.DataFrame
) -> pd.DataFrame:
    parts = [structural.reset_index(drop=True), aggregates.reset_index(drop=True)]
    if parent_context is not None:
        parts.insert(0, parent_context.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def _joined(
    table: RelatedTable,
    parts: list[tuple[pd.DataFrame, pd.DataFrame]],
    parent_keys: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put the built parts together in the declared column order, with the keys in front."""
    names = [column.name for column in table.columns]
    published_by_name = {name: frame[name] for frame, _ in parts for name in frame.columns}
    structural_by_name = {name: frame[name] for _, frame in parts for name in frame.columns}
    rows = len(parts[0][0])
    keys = {table.key: np.arange(1, rows + 1)}
    if table.foreign_key is not None:
        assert parent_keys is not None
        keys[table.foreign_key] = parent_keys
    published = pd.DataFrame({**keys, **{name: published_by_name[name] for name in names}})
    structural = pd.DataFrame({name: structural_by_name[name] for name in names})
    return published, structural


def _aggregate(
    aggregate: ChildAggregate,
    child_structural: pd.DataFrame,
    positions: np.ndarray,
    parent_rows: int,
) -> np.ndarray:
    """Summarise the child rows per parent row; parents with no matching child rows get 0."""
    keep = np.ones(len(child_structural), dtype=bool)
    if aggregate.when is not None:
        keep = np.asarray(evaluate(aggregate.when, child_structural), dtype=bool)
    if aggregate.column is None:
        values = pd.Series(np.ones(int(keep.sum())))
    else:
        values = child_structural[aggregate.column][keep].astype(float).reset_index(drop=True)
    grouped = values.groupby(positions[keep])
    # A count is the sum of one per child row.
    method = "sum" if aggregate.function is AggregateFunction.COUNT else aggregate.function.value
    summary = getattr(grouped, method)()
    return summary.reindex(range(parent_rows), fill_value=0.0).to_numpy(dtype=float)
