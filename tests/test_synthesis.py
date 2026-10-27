from pathlib import Path

import pandas as pd
import pytest

from data_anonymizer import Anonymizer, ColumnSchema, DatasetSchema, Fidelity, SchemaError
from data_anonymizer.detection.patterns import is_valid_cpf
from data_anonymizer.synthesis import infer_schema
from data_anonymizer.taxonomy import ColumnKind, DataCategory


def test_schema_only_generation_follows_each_column_description():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="id", kind=ColumnKind.INTEGER, minimum=1, unique=True),
            ColumnSchema(name="cpf", kind=ColumnKind.TEXT, category=DataCategory.CPF),
            ColumnSchema(name="score", kind=ColumnKind.FLOAT, mean=600, std=50, minimum=300),
            ColumnSchema(
                name="plan", kind=ColumnKind.CATEGORY, choices=["free", "pro"], weights=[9, 1]
            ),
            ColumnSchema(
                name="signup",
                kind=ColumnKind.DATETIME,
                start="2024-01-01",
                end="2024-12-31",
            ),
            ColumnSchema(name="order", kind=ColumnKind.TEXT, template="ORD-####"),
            ColumnSchema(name="active", kind=ColumnKind.BOOLEAN, true_share=0.3),
            ColumnSchema(
                name="nickname", kind=ColumnKind.TEXT, choices=["x", "y"], null_fraction=0.5
            ),
        ]
    )

    frame = Anonymizer(seed=3).synthesize(2000, schema=schema)

    assert len(frame) == 2000
    assert frame["id"].is_unique
    assert frame["cpf"].map(is_valid_cpf).all()
    assert frame["score"].min() >= 300
    assert frame["score"].mean() == pytest.approx(600, abs=5)
    assert (frame["plan"] == "free").mean() == pytest.approx(0.9, abs=0.03)
    assert frame["signup"].between("2024-01-01", "2024-12-31").all()
    assert frame["order"].str.fullmatch(r"ORD-\d{4}").all()
    assert frame["active"].mean() == pytest.approx(0.3, abs=0.03)
    assert frame["nickname"].isna().mean() == pytest.approx(0.5, abs=0.04)


def test_schema_round_trips_through_json():
    schema = DatasetSchema(columns=[ColumnSchema(name="a", kind=ColumnKind.FLOAT, minimum=0)])

    assert DatasetSchema.from_json(schema.to_json()) == schema


def test_schema_rejects_inconsistent_fields():
    with pytest.raises(ValueError, match="one entry per choice"):
        ColumnSchema(name="a", kind=ColumnKind.CATEGORY, choices=["x"], weights=[1, 2])


def test_missing_ranges_are_reported_clearly():
    schema = DatasetSchema(columns=[ColumnSchema(name="amount", kind=ColumnKind.FLOAT)])

    with pytest.raises(SchemaError, match="minimum and maximum"):
        Anonymizer().synthesize(10, schema=schema)


def test_inferred_schema_never_copies_identifiers(people):
    schema = infer_schema(people)
    by_name = {column.name: column for column in schema.columns}

    assert by_name["email"].category is DataCategory.EMAIL
    assert by_name["email"].choices is None
    assert by_name["cep"].category is DataCategory.POSTAL_CODE
    assert by_name["city"].choices is not None


def test_statistical_synthesis_keeps_correlations_and_invents_identifiers(people):
    synthetic = Anonymizer(seed=5).synthesize(1000, reference=people)

    original_correlation = people["age"].corr(people["salary"])
    synthetic_correlation = synthetic["age"].corr(synthetic["salary"])
    assert synthetic_correlation == pytest.approx(original_correlation, abs=0.1)
    assert synthetic["age"].mean() == pytest.approx(people["age"].mean(), rel=0.05)
    assert not synthetic["email"].isin(people["email"]).any()
    assert synthetic["customer_id"].is_unique
    assert synthetic["birth_date"].dt.year.between(1940, 2010).all()


def test_privacy_synthesis_drops_correlations(people):
    synthetic = Anonymizer(seed=5).synthesize(1000, reference=people, fidelity=Fidelity.PRIVACY)

    assert abs(synthetic["age"].corr(synthetic["salary"])) < 0.15


def test_privacy_inference_hides_rare_categories():
    frame = pd.DataFrame({"plan": ["basic"] * 20 + ["vip-celebrity"]})

    schema = infer_schema(frame, min_category_count=5)

    assert schema.columns[0].choices == ["basic", "OTHER"]


def test_synthesis_needs_a_schema_or_a_reference():
    with pytest.raises(SchemaError):
        Anonymizer().synthesize(10)


def test_example_schema_generates_a_table():
    schema_path = Path(__file__).parent.parent / "examples" / "customers_schema.json"

    frame = Anonymizer(seed=1, locale="pt_BR").synthesize(
        300, schema=DatasetSchema.load(schema_path)
    )

    assert list(frame.columns) == [
        column.name for column in DatasetSchema.load(schema_path).columns
    ]
    assert frame["email"].is_unique


def test_a_challenge_withholds_the_truth_and_scores_the_optimum(tmp_path):
    schema_path = Path(__file__).parent.parent / "examples" / "churn_challenge.json"
    schema = DatasetSchema.load(schema_path)

    challenge = Anonymizer(seed=4).challenge(schema, 3000)
    key = challenge.answer_key

    assert "churn_probability" not in challenge.train.columns
    assert "churned" not in challenge.test_features.columns
    assert len(challenge.test_features) == len(challenge.test_answers)
    assert set(key["features"]["true_drivers"]) == {
        "plan",
        "tenure_months",
        "monthly_spend",
        "support_tickets",
    }
    assert "age" in key["features"]["correlated_but_not_causal"]
    assert "region" in key["features"]["distractors"]
    assert 0.6 < key["best_achievable"]["roc_auc"] < 0.95


def test_regression_challenge_optimum_matches_the_planted_noise():
    schema = DatasetSchema.model_validate(
        {
            "columns": [
                {"name": "x", "kind": "float", "mean": 0, "std": 1},
                {"name": "noise_col", "kind": "float", "mean": 0, "std": 1},
                {"name": "y", "kind": "float", "formula": "3 * x + 1", "noise_std": 0.5},
            ],
            "question": {
                "title": "Fit y",
                "prompt": "Predict y.",
                "target": "y",
                "task": "regression",
            },
        }
    )

    key = Anonymizer(seed=1).challenge(schema, 5000).answer_key

    assert key["best_achievable"]["rmse"] == pytest.approx(0.5, rel=0.08)
    assert key["features"]["true_drivers"] == ["x"]
    assert key["features"]["distractors"] == ["noise_col"]


def test_constraints_do_not_remove_the_requested_nulls():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="start", kind=ColumnKind.FLOAT, minimum=0, maximum=100),
            ColumnSchema(
                name="end", kind=ColumnKind.FLOAT, minimum=0, maximum=100, null_fraction=0.2
            ),
        ],
        constraints=["end >= start", "end > -5"],
    )

    frame = Anonymizer(seed=2).synthesize(5000, schema=schema)
    present = frame[frame["end"].notna()]

    assert len(frame) == 5000
    assert frame["end"].isna().mean() == pytest.approx(0.2, abs=0.03)
    assert (present["end"] >= present["start"]).all()


def test_unique_keys_stay_consistent_with_constraints_and_formulas():
    schema = DatasetSchema(
        columns=[
            ColumnSchema(name="id", kind=ColumnKind.INTEGER, minimum=1, maximum=40000, unique=True),
            ColumnSchema(name="twice", kind=ColumnKind.INTEGER, formula="id * 2"),
            ColumnSchema(name="v", kind=ColumnKind.FLOAT, mean=0, std=1),
        ],
        constraints=["v > 0", "id > 30000"],
    )

    frame = Anonymizer(seed=3).synthesize(3000, schema=schema)

    assert len(frame) == 3000
    assert frame["id"].is_unique
    assert (frame["id"] > 30000).all()
    assert (frame["twice"] == 2 * frame["id"]).all()
    assert (frame["v"] > 0).all()


def test_constraints_with_about_one_percent_acceptance_succeed():
    schema = DatasetSchema(
        columns=[ColumnSchema(name="a", kind=ColumnKind.FLOAT, mean=0, std=1)],
        constraints=["a > 2.33"],
    )

    frame = Anonymizer(seed=4).synthesize(20000, schema=schema)

    assert len(frame) == 20000
    assert (frame["a"] > 2.33).all()


def test_impossible_constraints_report_the_acceptance_rate():
    schema = DatasetSchema(
        columns=[ColumnSchema(name="a", kind=ColumnKind.FLOAT, minimum=0, maximum=1)],
        constraints=["a > 2"],
    )

    with pytest.raises(SchemaError, match=r"accepted only 0 of \d+ drawn rows \(0\.000%\)"):
        Anonymizer(seed=4).synthesize(10, schema=schema)


def test_a_unique_template_gives_distinct_values_or_fails_clearly():
    def template_schema() -> DatasetSchema:
        column = ColumnSchema(name="code", kind=ColumnKind.TEXT, template="ID-###", unique=True)
        return DatasetSchema(columns=[column])

    full = Anonymizer(seed=5).synthesize(1000, schema=template_schema())["code"]
    with pytest.raises(SchemaError, match="only 1000 possible values"):
        Anonymizer(seed=5).synthesize(1001, schema=template_schema())

    assert full.is_unique
    assert full.str.fullmatch(r"ID-\d{3}").all()


@pytest.mark.parametrize(
    "settings",
    [
        {"kind": ColumnKind.BOOLEAN},
        {"kind": ColumnKind.DATETIME, "start": "2024-01-01", "end": "2024-12-31"},
        {"kind": ColumnKind.CATEGORY, "choices": ["a", "b", "c"]},
        {"kind": ColumnKind.TEXT, "choices": ["a", "b"]},
    ],
)
def test_unique_is_rejected_where_it_cannot_be_honoured(settings):
    with pytest.raises(ValueError, match="unique is not possible"):
        ColumnSchema(name="x", unique=True, **settings)
