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
