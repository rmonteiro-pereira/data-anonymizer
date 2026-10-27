from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_anonymizer import Anonymizer, ColumnSchema, DatasetSchema, Fidelity, SchemaError
from data_anonymizer.detection.patterns import is_valid_cpf
from data_anonymizer.synthesis import infer_schema
from data_anonymizer.synthesis.schema import Distribution
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


def _associated_reference(rows: int = 20000) -> pd.DataFrame:
    generator = np.random.default_rng(0)
    smoker = generator.random(rows) < 0.4
    agrees = generator.random(rows) < 0.8
    high_risk = np.where(agrees, smoker, generator.random(rows) < 0.5)
    return pd.DataFrame(
        {
            "smoker": np.where(smoker, "yes", "no"),
            "risk": np.where(high_risk, "high", "low"),
            "flag": smoker,
            "other_flag": high_risk,
        }
    )


def test_statistical_synthesis_keeps_strong_associations_between_categories_and_booleans():
    reference = _associated_reference()

    synthetic = Anonymizer(seed=1).synthesize(20000, reference=reference)

    def high_share(frame: pd.DataFrame, smoker: str) -> float:
        return float((frame.loc[frame["smoker"] == smoker, "risk"] == "high").mean())

    def phi(frame: pd.DataFrame) -> float:
        return float(np.corrcoef(frame["flag"], frame["other_flag"].astype(float))[0, 1])

    assert high_share(synthetic, "yes") == pytest.approx(high_share(reference, "yes"), abs=0.03)
    assert high_share(synthetic, "no") == pytest.approx(high_share(reference, "no"), abs=0.03)
    assert phi(synthetic) == pytest.approx(phi(reference), abs=0.04)


def _grouped_reference() -> pd.DataFrame:
    generator = np.random.default_rng(1)
    groups = generator.choice(["a", "b", "c", "d"], 6000, p=[0.5, 0.3, 0.2 - 0.002, 0.002])
    level = pd.Series(groups).map({"a": 0.0, "b": 10.0, "c": 1.0, "d": 5.0}).to_numpy()
    return pd.DataFrame({"group": groups, "level": level + generator.normal(0, 0.5, 6000)})


def test_stratifying_keeps_each_levels_own_distribution():
    reference = _grouped_reference()

    synthetic = Anonymizer(seed=2).synthesize(10000, reference=reference, stratify_by=["group"])
    means = synthetic.groupby("group")["level"].mean()

    assert means["a"] == pytest.approx(0.0, abs=0.2)
    assert means["b"] == pytest.approx(10.0, abs=0.2)
    assert means["c"] == pytest.approx(1.0, abs=0.2)
    assert synthetic["group"].value_counts(normalize=True)["b"] == pytest.approx(0.3, abs=0.03)
    # The 12 rows of level d are too few for their own copula, but the level is not lost.
    assert "d" in set(synthetic["group"])


def test_stratify_by_names_must_be_learned_category_columns():
    reference = _grouped_reference()

    with pytest.raises(SchemaError, match="stratify_by"):
        Anonymizer(seed=2).synthesize(100, reference=reference, stratify_by=["level"])
    with pytest.raises(SchemaError, match="reference"):
        Anonymizer(seed=2).synthesize(
            100,
            schema=DatasetSchema(columns=[ColumnSchema(name="a", kind=ColumnKind.FLOAT, mean=0)]),
            stratify_by=["a"],
        )


def test_privacy_synthesis_drops_correlations(people):
    synthetic = Anonymizer(seed=5).synthesize(1000, reference=people, fidelity=Fidelity.PRIVACY)

    assert abs(synthetic["age"].corr(synthetic["salary"])) < 0.15


def test_privacy_synthesis_keeps_the_mean_and_sign_of_zero_inflated_heavy_tailed_columns():
    generator = np.random.default_rng(3)
    gains = np.where(generator.random(20000) < 0.92, 0.0, generator.lognormal(8.5, 1.2, 20000))
    balance = generator.lognormal(1.0, 1.0, 20000) * np.where(generator.random(20000) < 0.02, -1, 1)
    reference = pd.DataFrame({"gain": gains.round(2), "balance": balance.round(2)})

    schema = infer_schema(reference)
    synthetic = Anonymizer(seed=9).synthesize(20000, reference=reference, fidelity=Fidelity.PRIVACY)

    assert all(column.distribution is Distribution.QUANTILES for column in schema.columns)
    assert (synthetic["gain"] >= 0).all()
    assert synthetic["gain"].mean() == pytest.approx(reference["gain"].mean(), rel=0.25)
    assert (synthetic["balance"] < 0).mean() == pytest.approx(0.02, abs=0.01)
    assert synthetic["balance"].mean() == pytest.approx(reference["balance"].mean(), rel=0.3)


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


def _regression_schema(target_settings: dict, driver_settings: dict | None = None) -> dict:
    return {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1, **(driver_settings or {})},
            {"name": "y", "kind": "float", "formula": "2 * x", "noise_std": 2, **target_settings},
        ],
        "question": {"title": "Fit y", "prompt": "Predict y.", "target": "y", "task": "regression"},
    }


@pytest.mark.parametrize(
    "target_settings",
    [
        {"minimum": 0},
        {"exact": True, "mean": 5, "std": 3},
        {"conditions": [{"when": "x > 0", "mean": 50, "std": 1}]},
        {"outlier_fraction": 0.05},
    ],
)
def test_best_achievable_is_not_computable_when_the_target_is_changed_after_its_formula(
    target_settings,
):
    schema = DatasetSchema.model_validate(_regression_schema(target_settings))

    key = Anonymizer(seed=1).challenge(schema, 2000).answer_key

    assert key["best_achievable"].startswith("not computable")


def test_best_achievable_skips_rows_with_a_missing_target_or_driver():
    schema = DatasetSchema.model_validate(
        _regression_schema({"null_fraction": 0.1}, {"null_fraction": 0.1})
    )

    scores = Anonymizer(seed=1).challenge(schema, 4000).answer_key["best_achievable"]

    assert scores["rmse"] == pytest.approx(2.0, rel=0.1)
    assert scores["r2"] == pytest.approx(0.5, abs=0.1)


def _label_schema(target: dict, question: dict | None = None) -> dict:
    return {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "p", "kind": "float", "formula": "sigmoid(2 * x)"},
            {"name": "label", "formula": "bernoulli(p)", **target},
        ],
        "question": {
            "title": "Label",
            "prompt": "Predict the label.",
            "target": "label",
            "task": "classification",
            "hidden_columns": ["p"],
            "truth_column": "p",
            **(question or {}),
        },
    }


def test_classification_scores_ignore_rows_without_a_label():
    schema = DatasetSchema.model_validate(_label_schema({"kind": "boolean", "null_fraction": 0.2}))

    scores = Anonymizer(seed=1).challenge(schema, 4000).answer_key["best_achievable"]

    assert 0.7 < scores["accuracy"] < 0.9
    assert 0.35 < scores["log_loss"] < 0.55
    assert 0.4 < scores["positive_rate"] < 0.6


def test_classification_with_text_labels_needs_and_uses_a_positive_label():
    labels = {
        "kind": "category",
        "choices": ["no", "yes"],
        "formula": "where(bernoulli(p), 'yes', 'no')",
    }

    with pytest.raises(ValueError, match="positive_label"):
        DatasetSchema.model_validate(_label_schema(labels))
    schema = DatasetSchema.model_validate(_label_schema(labels, {"positive_label": "yes"}))
    scores = Anonymizer(seed=1).challenge(schema, 4000).answer_key["best_achievable"]

    assert 0.7 < scores["roc_auc"] < 0.9
    assert 0.4 < scores["positive_rate"] < 0.6


def test_classification_rejects_numeric_labels_that_are_not_zero_or_one():
    target = {"kind": "integer", "formula": "1 + bernoulli(p)"}
    schema = DatasetSchema.model_validate(_label_schema(target))

    with pytest.raises(SchemaError, match="positive_label"):
        Anonymizer(seed=1).challenge(schema, 500)


def test_a_challenge_with_too_few_rows_fails_clearly():
    schema = DatasetSchema.model_validate(_regression_schema({}))

    with pytest.raises(SchemaError, match="test rows"):
        Anonymizer(seed=1).challenge(schema, 1)


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
