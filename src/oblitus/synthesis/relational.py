"""A schema of several related tables joined by foreign keys, such as customers and their orders.

The first table is the root: it has as many rows as are asked for. Every other table names its
`parent` table, the `foreign_key` column that holds the parent's key, and `rows_per_parent`, how
many rows each parent row gets. Keys are the whole numbers 1, 2, 3, ... so every foreign key
points at a parent row that exists.

A child column may read its parent's columns: the column `tier` of table `customers` is read as
`customers_tier`. A parent column may read an aggregate of its children, declared in
`aggregates`, such as `sum(orders.amount)` named `lifetime_spend`. Such a column, and every
column that reads it, is built after the children, so a child can only read parent columns that
do not themselves depend on an aggregate. Aggregates are withheld from the published tables, so
a challenge whose target depends on one can only be solved by aggregating the child table.
"""

import json
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oblitus.errors import SchemaError
from oblitus.synthesis.expressions import parse, referenced_columns, uses_randomness
from oblitus.synthesis.marginals import quantile_function
from oblitus.synthesis.schema import (
    ColumnSchema,
    Correlation,
    DatasetSchema,
    Distribution,
    Question,
)
from oblitus.taxonomy import ColumnKind


class AggregateFunction(StrEnum):
    """How the rows of a child table are summarised per parent row."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


class ChildAggregate(BaseModel):
    """A summary of one child table's rows per parent row, such as the total of `amount`.

    `column` is the child column to summarise, left out for `count`. `when` keeps only the
    child rows where the expression over the child's columns is true. A parent with no
    matching child rows gets 0.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    table: str
    function: AggregateFunction
    column: str | None = None
    when: str | None = None

    @field_validator("when")
    @classmethod
    def _parseable_condition(cls, text: str | None) -> str | None:
        if text is not None:
            parse(text)
        return text

    @model_validator(mode="after")
    def _column_matches_function(self) -> "ChildAggregate":
        counts = self.function is AggregateFunction.COUNT
        if counts and self.column is not None:
            raise ValueError(f"{self.name}: a count aggregate takes no column")
        if not counts and self.column is None:
            raise ValueError(f"{self.name}: a {self.function.value} aggregate needs a column")
        return self


class ChildCount(BaseModel):
    """How many rows of a child table each parent row gets, drawn per parent.

    It takes the settings of a numeric column's distribution; the default is a Poisson count,
    which needs a `mean`. `minimum` is 0 or more, so a parent may have no children.
    """

    model_config = ConfigDict(extra="forbid")

    distribution: Distribution = Distribution.POISSON
    mean: float | None = None
    std: float | None = Field(default=None, ge=0)
    minimum: int = Field(default=0, ge=0)
    maximum: int | None = None

    def as_column(self) -> ColumnSchema:
        """Return the integer column whose distribution gives the counts."""
        return ColumnSchema(
            name="rows_per_parent",
            kind=ColumnKind.INTEGER,
            distribution=self.distribution,
            mean=self.mean,
            std=self.std,
            minimum=self.minimum,
            maximum=self.maximum,
        )

    @model_validator(mode="after")
    def _distribution_is_complete(self) -> "ChildCount":
        try:
            quantile_function(self.as_column())(np.array([0.5]))
        except SchemaError as error:
            raise ValueError(f"rows_per_parent: {error}") from error
        return self


class RelatedTable(BaseModel):
    """One table of a relational schema.

    `key` names the generated primary key, 1 to the number of rows. A child table also has
    `parent`, `foreign_key` and `rows_per_parent`. Neither key is listed among `columns`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    key: str
    columns: list[ColumnSchema] = Field(min_length=1)
    correlations: list[Correlation] = Field(default_factory=list)
    parent: str | None = None
    foreign_key: str | None = None
    rows_per_parent: ChildCount | None = None
    aggregates: list[ChildAggregate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names_and_link_are_consistent(self) -> "RelatedTable":
        link = (self.parent, self.foreign_key, self.rows_per_parent)
        if any(part is None for part in link) and any(part is not None for part in link):
            raise ValueError(
                f"{self.name}: parent, foreign_key and rows_per_parent go together; "
                "a root table has none of them"
            )
        taken = [column.name for column in self.columns]
        generated = [self.key, *([self.foreign_key] if self.foreign_key else [])]
        generated += [aggregate.name for aggregate in self.aggregates]
        names = taken + generated
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"{self.name}: names used twice: {', '.join(repeated)}")
        return self

    def aggregate_names(self) -> set[str]:
        """Return the names of the aggregates this table reads from its children."""
        return {aggregate.name for aggregate in self.aggregates}

    def later_columns(self) -> set[str]:
        """Return the columns built after the children: those that read an aggregate.

        A column reads one directly, or through another column that does.
        """
        reads_children = self.aggregate_names()
        later: set[str] = set()
        found_more = True
        while found_more:
            found_more = False
            for column in self.columns:
                if column.name not in later and column.dependencies() & (reads_children | later):
                    later.add(column.name)
                    found_more = True
        return later


class RelationalSchema(BaseModel):
    """Related tables, and optionally a question whose target is a column of the root table.

    A causal question or a test shift is refused: both regenerate rows from stored random
    numbers, which this schema keeps per table, not across tables.
    """

    model_config = ConfigDict(extra="forbid")

    tables: list[RelatedTable] = Field(min_length=2)
    question: Question | None = None

    @model_validator(mode="after")
    def _tables_form_a_tree(self) -> "RelationalSchema":
        names = [table.name for table in self.tables]
        if len(set(names)) != len(names):
            raise ValueError("table names must be unique")
        if self.tables[0].parent is not None:
            raise ValueError(f"the first table {names[0]!r} is the root and has no parent")
        for index, table in enumerate(self.tables[1:], start=1):
            if table.parent not in names[:index]:
                raise ValueError(
                    f"{table.name}: parent {table.parent!r} must be a table listed before it"
                )
        return self

    @model_validator(mode="after")
    def _aggregates_read_child_columns(self) -> "RelationalSchema":
        for table in self.tables:
            for aggregate in table.aggregates:
                self._check_aggregate(table, aggregate)
        return self

    def _check_aggregate(self, table: RelatedTable, aggregate: ChildAggregate) -> None:
        child = next((item for item in self.tables if item.name == aggregate.table), None)
        if child is None or child.parent != table.name:
            raise ValueError(
                f"{aggregate.name}: {aggregate.table!r} is not a child table of {table.name!r}"
            )
        names = {column.name for column in child.columns}
        read = {aggregate.column} if aggregate.column else set()
        read |= referenced_columns(aggregate.when) if aggregate.when else set()
        if aggregate.when and uses_randomness(aggregate.when):
            raise ValueError(f"{aggregate.name}: random functions are not allowed in when")
        unknown = read - names
        if unknown:
            raise ValueError(
                f"{aggregate.name}: {aggregate.table} has no columns {', '.join(sorted(unknown))}"
            )

    @model_validator(mode="after")
    def _every_table_builds(self) -> "RelationalSchema":
        for table in self.tables:
            if len(table.later_columns()) == len(table.columns):
                raise ValueError(
                    f"{table.name}: every column reads an aggregate of its children, so none "
                    "can be built first; add a column that does not"
                )
            self._check_columns_read_the_right_parent_columns(table)
            self.table_schema(table)
            self.first_schema(table)
            self.later_schema(table)
        return self

    @model_validator(mode="after")
    def _question_fits(self) -> "RelationalSchema":
        question = self.question
        if question is None:
            return self
        if question.treatment is not None:
            raise SchemaError(
                "a causal question (treatment) is not supported with related tables: an "
                "intervention replays one table from its stored random numbers, and a treatment "
                "on a parent would also change its children's rows and the aggregates built "
                "from them. Remove the treatment"
            )
        if question.test_shift:
            raise SchemaError(
                "test_shift is not supported with related tables, because a shifted test set "
                "is generated again from the schema, and the child rows of a parent cannot "
                "be generated apart from it"
            )
        self.root_schema()
        return self

    def _check_columns_read_the_right_parent_columns(self, table: RelatedTable) -> None:
        if table.parent is None:
            return
        parent = self.table(table.parent)
        built_after_children = {f"{parent.name}_{name}" for name in parent.later_columns()}
        for column in table.columns:
            wrongly_read = column.dependencies() & built_after_children
            if wrongly_read:
                raise ValueError(
                    f"{table.name}.{column.name} reads {', '.join(sorted(wrongly_read))}, which "
                    f"depends on an aggregate of {table.name}; a child can only read parent "
                    "columns built before its rows"
                )

    def table(self, name: str) -> RelatedTable:
        """Return one table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)

    def children_of(self, name: str) -> list[RelatedTable]:
        """Return the tables whose parent is the named table, in schema order."""
        return [table for table in self.tables if table.parent == name]

    def parent_context_names(self, table: RelatedTable) -> list[str]:
        """Return the names under which a table's columns read its parent's columns."""
        if table.parent is None:
            return []
        parent = self.table(table.parent)
        later = parent.later_columns()
        return [
            f"{parent.name}_{column.name}" for column in parent.columns if column.name not in later
        ]

    def table_schema(self, table: RelatedTable, question: Question | None = None) -> DatasetSchema:
        """Return the whole table as one schema; aggregates and parent columns are context."""
        context = self.parent_context_names(table) + sorted(table.aggregate_names())
        return DatasetSchema(
            columns=table.columns,
            correlations=table.correlations,
            context_columns=context,
            question=question,
        )

    def first_schema(self, table: RelatedTable) -> DatasetSchema:
        """Return the columns built before the children, which read only the parent's columns."""
        later = table.later_columns()
        columns = [column for column in table.columns if column.name not in later]
        names = {column.name for column in columns}
        for correlation in table.correlations:
            if not set(correlation.columns) <= names:
                raise ValueError(f"{table.name}: a correlation names a column built after children")
        return DatasetSchema(
            columns=columns,
            correlations=table.correlations,
            context_columns=self.parent_context_names(table),
        )

    def later_schema(self, table: RelatedTable) -> DatasetSchema | None:
        """Return the columns built after the children, or None when there are none."""
        later = table.later_columns()
        if not later:
            return None
        first_names = [column.name for column in table.columns if column.name not in later]
        context = self.parent_context_names(table) + first_names + sorted(table.aggregate_names())
        return DatasetSchema(
            columns=[column for column in table.columns if column.name in later],
            context_columns=context,
        )

    def root_schema(self) -> DatasetSchema:
        """Return the root table as one schema carrying the question, to validate and to key."""
        return self.table_schema(self.tables[0], self.question)

    def with_root(self, root: DatasetSchema) -> "RelationalSchema":
        """Return the schema with the root table's columns and the question taken from `root`."""
        tables = [self.tables[0].model_copy(update={"columns": root.columns}), *self.tables[1:]]
        return RelationalSchema(tables=tables, question=root.question)


def load_schema(path: Path) -> DatasetSchema | RelationalSchema:
    """Read a schema file: related tables when it has a `tables` list, else a single table."""
    text = path.read_text(encoding="utf-8")
    if "tables" in json.loads(text):
        return RelationalSchema.model_validate_json(text)
    return DatasetSchema.from_json(text)
