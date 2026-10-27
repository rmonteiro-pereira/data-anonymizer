"""The task text participants get is written from the schema and never reveals the answer key."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from oblitus import Anonymizer, DatasetSchema
from oblitus.synthesis import load_schema
from oblitus.synthesis.question_text import describe_task

EXAMPLES = Path(__file__).parent.parent / "examples"

# Words that only the answer key may use about how the data was built.
SECRET_WORDS = [
    "mnar",
    "mar ",
    "mcar",
    "driver",
    "formula",
    "label noise",
    "drift",
    "distractor",
    "best achievable",
    "best_achievable",
    "flip",
    "churn_probability",
    "order_count",
    "total_spend",
    "refund_count",
    "average_items",
]


def imbalanced_churn_spec() -> dict:
    spec = json.loads((EXAMPLES / "churn_challenge.json").read_text())
    spec["question"].update(positive_share=0.05, difficulty="hard")
    spec["columns"][5].update(null_fraction=0.1, missingness={"mechanism": "mnar", "strength": 1.0})
    return spec


def test_the_text_names_the_task_the_files_the_score_and_the_rules():
    schema = DatasetSchema.model_validate(imbalanced_churn_spec())
    challenge = Anonymizer(seed=1).challenge(schema, 10_000)

    page = describe_task(schema, challenge)

    assert page.startswith("# Who is about to churn?")
    assert "Predict `churned`" in page
    assert "`train.csv`" in page
    assert "`test.csv`" in page
    assert "precision-recall" in page
    assert "It is rare" in page
    assert "`monthly_spend`" in page.split("blank cells")[1]
    assert "same order" in page


def test_the_text_never_reveals_how_the_data_was_built():
    schema = DatasetSchema.model_validate(imbalanced_churn_spec())
    challenge = Anonymizer(seed=1).challenge(schema, 10_000)

    page = describe_task(schema, challenge).lower()

    for word in SECRET_WORDS:
        assert word not in page


def test_the_related_tables_text_describes_the_link_but_not_the_aggregates():
    schema = load_schema(EXAMPLES / "customers_orders_challenge.json")
    challenge = Anonymizer(seed=1).challenge(schema, 2_000)

    page = describe_task(schema, challenge).lower()

    assert "`orders.csv`" in page
    assert "through `customer_id`" in page
    for word in SECRET_WORDS:
        assert word not in page


def test_a_question_metric_replaces_the_default():
    spec = json.loads((EXAMPLES / "churn_challenge.json").read_text())
    spec["question"]["metric"] = "log_loss"
    schema = DatasetSchema.model_validate(spec)

    page = describe_task(schema, Anonymizer(seed=1).challenge(schema, 3_000))

    assert "log loss (lower is better)" in page
    assert (
        "metric"
        not in Anonymizer(seed=1)
        .challenge(DatasetSchema.load(EXAMPLES / "churn_challenge.json"), 3_000)
        .answer_key
    )


def test_a_metric_must_suit_the_task():
    spec = json.loads((EXAMPLES / "churn_challenge.json").read_text())
    spec["question"]["metric"] = "rmse"

    with pytest.raises(ValidationError, match="does not score a classification"):
        DatasetSchema.model_validate(spec)
