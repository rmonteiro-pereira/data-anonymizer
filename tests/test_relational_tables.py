"""Related tables: foreign keys that always resolve, and a target that needs the child rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from oblitus import Anonymizer
from oblitus.cli import main
from oblitus.errors import SchemaError
from oblitus.synthesis import RelationalSchema, load_schema

EXAMPLE = Path(__file__).parent.parent / "examples" / "customers_orders_challenge.json"
ROWS = 4_000


def example_spec(**question) -> dict:
    spec = json.loads(EXAMPLE.read_text())
    spec["question"].update(question)
    return spec


@pytest.fixture(scope="module")
def challenge():
    return Anonymizer(seed=3).challenge(load_schema(EXAMPLE), ROWS)


def test_every_foreign_key_points_at_a_parent_row_and_keys_are_unique():
    generated = Anonymizer(seed=3).synthesize_tables(load_schema(EXAMPLE), ROWS)
    customers = generated.tables["customers"]
    orders = generated.tables["orders"]

    assert len(customers) == ROWS
    assert customers["customer_id"].is_unique
    assert orders["order_id"].is_unique
    assert orders["customer_id"].isin(customers["customer_id"]).all()
    assert orders.groupby("customer_id").size().mean() == pytest.approx(5.0, abs=0.3)
    assert "order_count" not in customers.columns


def test_child_columns_can_depend_on_parent_columns():
    generated = Anonymizer(seed=3).synthesize_tables(load_schema(EXAMPLE), ROWS)
    tier = generated.tables["customers"].set_index("customer_id")["loyalty_tier"]
    orders = generated.tables["orders"]

    amounts = orders.groupby(orders["customer_id"].map(tier))["amount"].mean()

    assert amounts["gold"] > 1.5 * amounts["basic"]


def test_the_same_seed_gives_the_same_tables():
    schema = load_schema(EXAMPLE)

    first = Anonymizer(seed=9).synthesize_tables(schema, 500)
    second = Anonymizer(seed=9).synthesize_tables(schema, 500)

    pd.testing.assert_frame_equal(first.tables["orders"], second.tables["orders"])


def test_the_target_lives_in_the_parent_table_and_children_are_published_whole(challenge):
    parents = set(challenge.train["customer_id"]) | set(challenge.test_features["customer_id"])
    orders = challenge.related_tables["orders"]

    assert "churned" in challenge.train.columns
    assert "churned" not in challenge.test_features.columns
    assert {"order_count", "total_spend", "churn_probability"}.isdisjoint(challenge.train.columns)
    assert set(orders["customer_id"]) <= parents


def test_only_the_aggregated_child_rows_explain_the_target(challenge):
    train = challenge.train.set_index("customer_id")
    orders = challenge.related_tables["orders"]
    order_count = orders.groupby("customer_id").size().reindex(train.index, fill_value=0)

    correlation = np.corrcoef(order_count, train["churned"].astype(float))[0, 1]
    own_columns = train[["age"]].corrwith(train["churned"].astype(float)).abs().max()

    assert correlation < -0.1
    assert own_columns < 0.05


def test_the_key_says_which_aggregates_matter(challenge):
    aggregates = challenge.answer_key["child_aggregates"]

    matter = [item["name"] for item in aggregates["the_target_depends_on"]]
    assert matter == ["order_count", "total_spend", "refund_count"]
    assert [item["name"] for item in aggregates["not_used_by_the_target"]] == ["average_items"]
    assert challenge.answer_key["tables"]["orders"]["parent"] == "customers"
    assert challenge.answer_key["tables"]["orders"]["rows_per_parent"]["observed_mean"] > 4


def test_a_chain_of_three_tables_keeps_integrity():
    spec = {
        "tables": [
            {
                "name": "customers",
                "key": "customer_id",
                "columns": [{"name": "age", "kind": "integer", "mean": 40, "std": 10}],
            },
            {
                "name": "orders",
                "key": "order_id",
                "parent": "customers",
                "foreign_key": "customer_id",
                "rows_per_parent": {"mean": 3},
                "columns": [{"name": "total", "kind": "float", "mean": 10, "std": 2}],
            },
            {
                "name": "items",
                "key": "item_id",
                "parent": "orders",
                "foreign_key": "order_id",
                "rows_per_parent": {"mean": 2},
                "columns": [{"name": "price", "kind": "float", "formula": "orders_total / 2"}],
            },
        ]
    }

    tables = Anonymizer(seed=1).synthesize_tables(RelationalSchema.model_validate(spec), 300).tables

    assert tables["items"]["order_id"].isin(tables["orders"]["order_id"]).all()
    merged = tables["items"].merge(tables["orders"], on="order_id")
    np.testing.assert_allclose(merged["price"], merged["total"] / 2)


def test_a_treatment_is_refused_with_a_clear_error():
    spec = example_spec(treatment="loyalty_tier", treatment_values=["basic", "gold"])

    with pytest.raises(SchemaError, match="not supported with related tables"):
        RelationalSchema.model_validate(spec)


def test_a_test_shift_is_refused_with_a_clear_error():
    spec = example_spec(test_shift=[{"name": "age", "mean": 50}])

    with pytest.raises(SchemaError, match="test_shift is not supported"):
        RelationalSchema.model_validate(spec)


def test_a_child_cannot_read_a_parent_column_built_after_the_children():
    spec = example_spec()
    spec["tables"][1]["columns"].append(
        {"name": "risk", "kind": "float", "formula": "customers_churn_probability"}
    )

    with pytest.raises(ValidationError, match="can only read parent columns built before"):
        RelationalSchema.model_validate(spec)


def test_an_aggregate_must_name_a_child_table_and_its_columns():
    spec = example_spec()
    spec["tables"][0]["aggregates"][1]["column"] = "missing_column"

    with pytest.raises(ValidationError, match="has no columns missing_column"):
        RelationalSchema.model_validate(spec)


def test_the_challenge_command_writes_every_table_and_a_task_text(tmp_path):
    folder = tmp_path / "shop"

    assert (
        main(
            ["--seed", "3", "challenge", "--schema", str(EXAMPLE), "--rows", "600"]
            + ["-o", str(folder)]
        )
        == 0
    )

    assert (folder / "orders.csv").exists()
    page = (folder / "QUESTION.md").read_text()
    assert "`orders.csv`" in page
    assert "order_count" not in page
    answers = (folder / "answers" / "answer_key.json").read_text()
    assert "child_aggregates" in answers


def test_the_synthesize_command_writes_a_folder_of_tables(tmp_path, capsys):
    folder = tmp_path / "tables"

    assert main(["synthesize", "--schema", str(EXAMPLE), "--rows", "200", "-o", str(folder)]) == 0

    assert sorted(path.name for path in folder.iterdir()) == ["customers.csv", "orders.csv"]
    assert main(["synthesize", "--schema", str(EXAMPLE), "-o", str(folder)]) == 2
    assert "need --rows" in capsys.readouterr().err
