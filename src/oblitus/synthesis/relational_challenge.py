"""A data-science challenge over related tables, with the target in the root table.

The root table is split into train and test rows like a single table. Every child table is
published whole, because the features that matter are summaries of a parent's child rows. The
aggregates the schema declares are withheld, and the answer key says which of them the target
depends on.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from oblitus.errors import SchemaError
from oblitus.surrogates import SurrogateFactory
from oblitus.synthesis.challenge import Challenge, SplitTable, finish_challenge, split_rows
from oblitus.synthesis.class_balance import with_positive_share
from oblitus.synthesis.difficulty import apply_difficulty
from oblitus.synthesis.missingness import describe_missingness
from oblitus.synthesis.relational import ChildAggregate, RelatedTable, RelationalSchema
from oblitus.synthesis.relational_generation import GeneratedTables, generate_tables
from oblitus.synthesis.schema import DatasetSchema, columns_feeding


def build_relational_challenge(
    schema: RelationalSchema,
    rows: int | None,
    *,
    rng: np.random.Generator,
    surrogates: SurrogateFactory,
) -> Challenge:
    """Generate the tables, split the root table and write the answer key."""
    if schema.question is None:
        raise SchemaError("the schema has no question to build a challenge from")
    if rows is None:
        raise SchemaError("rows is required: it is the number of rows of the root table")
    root = schema.root_schema()
    difficulty = None
    if schema.question.difficulty is not None:
        difficulty = apply_difficulty(root, supports_drift=False)
        root = difficulty.schema
    root = with_positive_share(root)
    schema = schema.with_root(root)
    generated = generate_tables(schema, rows, rng=rng, surrogates=surrogates)

    root_table = schema.tables[0]
    aggregates = generated.aggregates[root_table.name]
    full = pd.concat([generated.tables[root_table.name], aggregates], axis=1)
    question = root.question
    assert question is not None
    withheld = question.model_copy(
        update={"hidden_columns": [*question.hidden_columns, *aggregates.columns]}
    )
    keyed_root = root.model_copy(update={"question": withheld})
    in_test = split_rows(withheld, rows, rng)
    split = SplitTable(built=None, full=full, in_test=in_test, test_schema=keyed_root)
    challenge = finish_challenge(keyed_root, split, rng, difficulty)

    challenge.answer_key["tables"] = _describe_tables(schema, generated)
    challenge.answer_key["child_aggregates"] = _describe_aggregates(root_table, keyed_root)
    published = {
        table.name: generated.tables[table.name]
        for table in schema.tables
        if table.name != root_table.name
    }
    return replace(challenge, related_tables=published)


def _describe_tables(schema: RelationalSchema, generated: GeneratedTables) -> dict[str, Any]:
    """Return rows, links, child counts and blank cells of every table."""
    described: dict[str, Any] = {}
    for table in schema.tables:
        frame = generated.tables[table.name]
        entry: dict[str, Any] = {"rows": len(frame), "key": table.key}
        if table.parent is not None:
            entry.update(_describe_link(table, frame, generated.tables[table.parent]))
        blank_cells = describe_missingness(table.columns, {"all_rows": frame})
        if blank_cells:
            entry["missingness"] = blank_cells
        described[table.name] = entry
    return described


def _describe_link(
    table: RelatedTable, frame: pd.DataFrame, parent: pd.DataFrame
) -> dict[str, Any]:
    assert table.foreign_key is not None
    assert table.rows_per_parent is not None
    per_parent = frame[table.foreign_key].value_counts().reindex(parent.iloc[:, 0], fill_value=0)
    return {
        "parent": table.parent,
        "foreign_key": table.foreign_key,
        "rows_per_parent": {
            "declared": table.rows_per_parent.model_dump(mode="json", exclude_none=True),
            "observed_mean": float(per_parent.mean()),
            "observed_max": int(per_parent.max()),
        },
    }


def _describe_aggregates(root_table: RelatedTable, root: DatasetSchema) -> dict[str, Any]:
    """Say which aggregates of child rows the target depends on, and which it does not."""
    question = root.question
    assert question is not None
    by_name = {column.name: column for column in root.columns}
    feeding = columns_feeding(by_name, question.target) | {question.target}
    read = set().union(*(by_name[name].dependencies() for name in feeding))
    matter = [item for item in root_table.aggregates if item.name in read]
    unused = [item for item in root_table.aggregates if item.name not in read]
    return {
        "withheld_from_published_tables": [item.name for item in root_table.aggregates],
        "the_target_depends_on": [_aggregate_entry(item) for item in matter],
        "not_used_by_the_target": [_aggregate_entry(item) for item in unused],
    }


def _aggregate_entry(aggregate: ChildAggregate) -> dict[str, Any]:
    return aggregate.model_dump(mode="json", exclude_none=True)
