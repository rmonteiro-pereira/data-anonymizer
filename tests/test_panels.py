"""Panels: entities over periods, levels, lags, regimes and time splits."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from data_anonymizer import Anonymizer, DatasetSchema
from data_anonymizer.errors import SchemaError
from data_anonymizer.synthesis.expressions import ExpressionError, evaluate, parse

EXAMPLE = Path(__file__).parent.parent / "examples" / "store_sales_panel.json"
STORES = 40
MONTHS = 36
REGIME_PERIOD = 24
AR_COEFFICIENT = 0.7
# The competitor lowers sales by 12 a month, which an AR(1) with 0.7 settles at 12 / (1 - 0.7).
LONG_RUN_DROP = 12 / (1 - AR_COEFFICIENT)


def example_json() -> dict[str, Any]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def small_panel(columns: list[dict[str, Any]], **question: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "panel": {
            "entity": "unit",
            "entities": 5,
            "time": "day",
            "start": "2024-01-01",
            "periods": 10,
            "frequency": "D",
        },
        "columns": columns,
    }
    if question:
        schema["question"] = {
            "title": "t",
            "prompt": "p",
            "target": "y",
            "task": "regression",
            **question,
        }
    return schema


def regress(columns: list[np.ndarray], outcome: np.ndarray) -> np.ndarray:
    """Least squares with an intercept first."""
    design = np.column_stack([np.ones(len(outcome)), *columns])
    return np.linalg.lstsq(design, outcome, rcond=None)[0]


@pytest.fixture(scope="module")
def sales_schema() -> DatasetSchema:
    return DatasetSchema.load(EXAMPLE)


@pytest.fixture(scope="module")
def sales_table(sales_schema) -> pd.DataFrame:
    return Anonymizer(seed=8).synthesize(schema=sales_schema)


@pytest.fixture(scope="module")
def sales_challenge(sales_schema):
    return Anonymizer(seed=8).challenge(sales_schema)


def test_the_ar_coefficient_estimated_from_the_panel_matches_the_formula(sales_table):
    rows = sales_table[sales_table["last_month_sales"].notna()]
    seasonal = np.sin(2 * np.pi * rows["date"].dt.month / 12)
    competitor = rows["new_competitor"].astype(float)
    coefficients = regress(
        [rows["last_month_sales"], rows["store_size"], seasonal, rows["market_shock"], competitor],
        rows["sales"].to_numpy(),
    )

    assert coefficients[1] == pytest.approx(AR_COEFFICIENT, abs=0.02)
    assert coefficients[2] == pytest.approx(0.9, abs=0.1)
    assert coefficients[3] == pytest.approx(6, abs=0.8)
    assert coefficients[4] == pytest.approx(1, abs=0.1)
    assert coefficients[5] == pytest.approx(-12, abs=1.5)


def test_a_self_lagging_formula_recovers_its_coefficient_in_a_small_panel():
    columns = [
        {
            "name": "y",
            "kind": "float",
            "noise_std": 2,
            "formula": "where(period == 0, 50, 10 + 0.8 * lag(y))",
        }
    ]
    schema = DatasetSchema.model_validate(small_panel(columns))
    schema = schema.model_copy(
        update={"panel": schema.panel.model_copy(update={"entities": 2000})}  # type: ignore[union-attr]
    )

    table = Anonymizer(seed=1).synthesize(schema=schema)
    later = table[table["day"] > table["day"].min()].copy()
    previous = table.groupby("unit")["y"].shift(1).loc[later.index]
    intercept, slope = regress([previous.to_numpy()], later["y"].to_numpy())

    assert slope == pytest.approx(0.8, abs=0.02)
    assert intercept == pytest.approx(10, abs=0.8)


def test_the_regime_at_period_24_shifts_the_level(sales_schema):
    quiet = example_json()
    quiet["columns"][2]["std"] = 0.01
    del quiet["question"]
    table = Anonymizer(seed=2).synthesize(schema=DatasetSchema.model_validate(quiet))
    period = (table["date"].dt.year - 2022) * 12 + table["date"].dt.month - 1
    mean_by_period = table.groupby(period)["sales"].mean()

    # The same calendar months a year apart, long enough after the break to have settled.
    before = mean_by_period.loc[18:23].mean()
    after = mean_by_period.loc[30:35].mean()

    assert after - before == pytest.approx(-LONG_RUN_DROP * (1 - AR_COEFFICIENT**7), abs=3)
    # The month of the break also falls 12 below the same month a year earlier.
    assert mean_by_period.loc[REGIME_PERIOD] < mean_by_period.loc[REGIME_PERIOD - 12] - 8


def test_entity_columns_are_constant_per_entity_and_period_columns_per_period(sales_table):
    per_store = sales_table.groupby("store_id")[["store_size", "region"]].nunique()
    per_month = sales_table.groupby("date")["market_shock"].nunique()

    assert (per_store == 1).all().all()
    assert (per_month == 1).all()
    assert sales_table["store_size"].nunique() == STORES
    assert sales_table["market_shock"].nunique() == MONTHS
    assert sales_table.groupby("date")["weather_index"].nunique().min() > 1


def test_the_panel_has_every_entity_in_every_period_ordered_by_entity(sales_table):
    assert len(sales_table) == STORES * MONTHS
    assert list(sales_table.columns[:2]) == ["store_id", "date"]
    assert sales_table["store_id"].is_monotonic_increasing
    assert sales_table["store_id"].unique().tolist() == list(range(1, STORES + 1))
    first_store = sales_table[sales_table["store_id"] == 1]["date"]
    assert first_store.is_monotonic_increasing
    assert first_store.iloc[0] == pd.Timestamp("2022-01-01")
    assert first_store.iloc[-1] == pd.Timestamp("2024-12-01")


def test_lag_is_missing_at_the_start_of_each_entity_and_matches_the_earlier_row(sales_table):
    by_store = sales_table.groupby("store_id")
    first_rows = by_store.head(1)
    later_rows = sales_table.drop(first_rows.index)

    assert first_rows["last_month_sales"].isna().all()
    assert later_rows["last_month_sales"].notna().all()
    expected = by_store["sales"].shift(1).loc[later_rows.index]
    np.testing.assert_allclose(later_rows["last_month_sales"], expected)


def test_a_lag_of_two_periods_is_missing_for_the_first_two_periods():
    columns = [
        {"name": "level", "kind": "float", "mean": 5, "std": 1},
        {"name": "before", "kind": "float", "formula": "lag(level, 2)"},
    ]
    table = Anonymizer(seed=4).synthesize(schema=DatasetSchema.model_validate(small_panel(columns)))

    assert table.groupby("unit")["before"].apply(lambda series: series.iloc[:2].isna().all()).all()
    assert table.groupby("unit")["before"].apply(lambda series: series.iloc[2:].notna().all()).all()
    expected = table.groupby("unit")["level"].shift(2)
    np.testing.assert_allclose(table["before"].iloc[2:], expected.iloc[2:])


def test_the_time_split_holds_out_the_last_periods(sales_challenge):
    train = sales_challenge.train
    test = sales_challenge.test_features
    periods = sales_challenge.answer_key["time_series"]["periods_per_split"]

    assert periods == {"train": 29, "test": 7}
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 7
    assert test["date"].max() == pd.Timestamp("2024-12-01")
    assert len(train) == STORES * 29
    assert len(test) == STORES * 7
    assert sales_challenge.test_answers.index.equals(test.index)


def test_the_answer_key_describes_the_panel_lags_and_regime(sales_challenge):
    series = sales_challenge.answer_key["time_series"]

    assert series["entity"] == "store_id"
    assert series["time"] == "date"
    assert series["split"] == "time"
    assert series["first_test_period"]["period"] == 29
    assert series["lags"] == {"sales": {"sales": [1]}, "last_month_sales": {"sales": [1]}}
    assert series["regimes"] == [
        {"column": "new_competitor", "when": "period >= 24", "changes": {"true_share": 1.0}}
    ]


def test_a_test_row_does_not_publish_the_previous_test_label(sales_challenge):
    test = sales_challenge.test_features.sort_values(["store_id", "date"])
    answers = sales_challenge.test_answers.loc[test.index, "sales"]
    first_date = test["date"].min()
    later = test["date"] > first_date

    assert test.loc[later, "last_month_sales"].isna().all()
    assert test.loc[~later, "last_month_sales"].notna().all()
    previous_label = answers.groupby(test["store_id"]).shift(1)
    assert not np.isclose(test["last_month_sales"], previous_label).any()
    key = sales_challenge.answer_key["time_series"]
    assert key["withheld_test_columns"] == ["last_month_sales"]


def test_the_training_rows_keep_their_lag_of_the_target(sales_challenge):
    train = sales_challenge.train.sort_values(["store_id", "date"])
    previous = train.groupby("store_id")["sales"].shift(1)

    later = previous.notna()
    np.testing.assert_allclose(train.loc[later, "last_month_sales"], previous[later])


def test_the_best_achievable_error_is_the_noise_of_the_one_step_formula(sales_challenge):
    best = sales_challenge.answer_key["best_achievable"]

    assert best["rmse"] == pytest.approx(5, abs=0.5)
    assert best["r2"] > 0.95


def test_the_shifted_test_periods_carry_on_from_the_end_of_the_train_periods(sales_challenge):
    train = sales_challenge.train
    test = sales_challenge.test_features
    last_train = train[train["date"] == train["date"].max()].set_index("store_id")["sales"]
    first_test = test[test["date"] == test["date"].min()].set_index("store_id")["last_month_sales"]

    np.testing.assert_allclose(first_test.loc[last_train.index], last_train)
    assert sales_challenge.answer_key["drift"]["shifted_columns"] == ["market_shock"]
    assert sales_challenge.answer_key["drift"]["kind"] == "covariate"


def test_a_shift_moves_the_test_periods_only():
    shifted = example_json()
    shifted["question"]["test_shift"] = [{"name": "market_shock", "mean": 60, "std": 1}]

    challenge = Anonymizer(seed=5).challenge(DatasetSchema.model_validate(shifted))
    by_period = lambda frame: frame.groupby("date")["market_shock"].first()  # noqa: E731

    assert by_period(challenge.train).abs().max() < 20
    assert by_period(challenge.test_features).min() > 50


def test_a_random_split_on_a_panel_keeps_every_period_in_both_splits(sales_schema):
    spec = example_json()
    spec["question"]["split"] = "random"
    del spec["question"]["test_shift"]
    spec["question"]["hidden_columns"].append("last_month_sales")

    challenge = Anonymizer(seed=6).challenge(DatasetSchema.model_validate(spec))

    assert challenge.answer_key["time_series"]["periods_per_split"] == {"train": 36, "test": 36}
    assert challenge.answer_key["time_series"]["first_test_period"] is None
    assert len(challenge.train) + len(challenge.test_features) == STORES * MONTHS


def test_label_noise_works_on_a_panel():
    spec = example_json()
    spec["question"]["label_noise"] = {"std": 5.0, "split": "train"}

    challenge = Anonymizer(seed=7).challenge(DatasetSchema.model_validate(spec))
    noise = challenge.answer_key["label_noise"]

    assert noise["perturbed"] == {"train": len(challenge.train), "test": 0}
    assert noise["best_achievable_on_clean_test_labels"]["rmse"] == pytest.approx(5, abs=0.5)


def test_a_column_reading_the_lagged_target_is_a_feature_not_a_leak(sales_challenge):
    features = sales_challenge.answer_key["features"]

    assert features["leaks_target"] == []
    assert features["lags_target"] == ["last_month_sales"]
    assert "last_month_sales" not in features["distractors"]
    assert features["hidden_drivers"] == ["new_competitor"]
    assert features["true_drivers"] == ["store_size", "market_shock"]


def test_a_column_reading_the_current_target_is_still_a_leak():
    columns = [
        {"name": "x", "kind": "float", "mean": 0, "std": 1},
        {"name": "y", "kind": "float", "noise_std": 1, "formula": "2 * x + 0.5 * lag(y)"},
        {"name": "echo", "kind": "float", "formula": "y * 3"},
        {"name": "previous", "kind": "float", "formula": "lag(y)"},
        {"name": "echo_of_previous", "kind": "float", "formula": "lag(echo)"},
    ]
    schema = DatasetSchema.model_validate(small_panel(columns, test_share=0.3, split="time"))

    features = Anonymizer(seed=1).challenge(schema).answer_key["features"]

    assert features["leaks_target"] == ["echo"]
    assert features["lags_target"] == ["previous"]
    assert "echo_of_previous" in features["distractors"]


def test_a_treatment_on_a_panel_is_refused_with_a_clear_error():
    columns = [
        {"name": "promo", "kind": "boolean", "true_share": 0.5},
        {"name": "y", "kind": "float", "noise_std": 1, "formula": "3 * promo + 0.5 * lag(y)"},
    ]
    spec = small_panel(columns, treatment="promo")

    with pytest.raises(SchemaError, match="not supported with a panel"):
        DatasetSchema.model_validate(spec)


def test_the_row_count_of_a_panel_is_fixed_by_the_panel(sales_schema):
    anonymizer = Anonymizer(seed=1)

    assert len(anonymizer.synthesize(STORES * MONTHS, schema=sales_schema)) == STORES * MONTHS
    with pytest.raises(SchemaError, match="1440 rows"):
        anonymizer.synthesize(100, schema=sales_schema)
    with pytest.raises(SchemaError, match="1440 rows"):
        anonymizer.challenge(sales_schema, 100)
    with pytest.raises(SchemaError, match="rows is required"):
        anonymizer.synthesize(schema=DatasetSchema.model_validate({"columns": [_float("x")]}))


def test_the_same_seed_builds_the_same_panel(sales_schema):
    first = Anonymizer(seed=9).synthesize(schema=sales_schema)
    second = Anonymizer(seed=9).synthesize(schema=sales_schema)

    pd.testing.assert_frame_equal(first, second)


def test_columns_can_be_correlated_within_a_level():
    columns = [
        {"name": "a", "kind": "float", "level": "entity", "mean": 0, "std": 1},
        {"name": "b", "kind": "float", "level": "entity", "mean": 0, "std": 1},
    ]
    spec = small_panel(columns)
    spec["panel"]["entities"] = 4000
    spec["correlations"] = [{"columns": ["a", "b"], "value": 0.8}]

    table = Anonymizer(seed=1).synthesize(schema=DatasetSchema.model_validate(spec))
    per_entity = table.groupby("unit")[["a", "b"]].first()

    assert per_entity.corr(method="spearman").iloc[0, 1] == pytest.approx(0.8, abs=0.03)


def test_a_gap_in_a_panel_column_blanks_cells():
    columns = [{"name": "x", "kind": "float", "mean": 0, "std": 1, "null_fraction": 0.5}]
    spec = small_panel(columns)
    spec["panel"]["entities"] = 400

    table = Anonymizer(seed=1).synthesize(schema=DatasetSchema.model_validate(spec))

    assert table["x"].isna().mean() == pytest.approx(0.5, abs=0.03)
    assert table["unit"].notna().all()


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            [
                {"name": "x", "kind": "float", "mean": 0, "std": 1, "level": "period"},
                {"name": "y", "kind": "float", "level": "entity", "formula": "x"},
            ],
            "entity-level column can only read entity-level",
        ),
        (
            [
                {"name": "x", "kind": "float", "mean": 0, "std": 1},
                {"name": "y", "kind": "float", "level": "period", "formula": "x"},
            ],
            "period-level column can only read period-level",
        ),
        (
            [
                {"name": "x", "kind": "float", "mean": 0, "std": 1, "level": "entity"},
                {"name": "y", "kind": "float", "formula": "lag(x)"},
            ],
            "entity-level column never changes",
        ),
        (
            [
                {"name": "r", "kind": "float", "mean": 0, "std": 1},
                {"name": "x", "kind": "float", "level": "entity", "formula": "lag(r)"},
            ],
            "entity-level column cannot use lag",
        ),
        ([{"name": "x", "kind": "float", "formula": "lag(missing)"}], "lag of unknown columns"),
        ([{"name": "unit", "kind": "float", "mean": 0, "std": 1}], "made or reserved by the panel"),
        (
            [{"name": "x", "kind": "integer", "unique": True, "minimum": 1}],
            "only entity-level columns can be unique",
        ),
        (
            [{"name": "x", "kind": "float", "mean": 0, "std": 1, "level": "period", "exact": True}],
            "one value per period",
        ),
        (
            [{"name": "x", "kind": "float", "mean": 0, "std": 1, "exact": True}],
            "not over the whole panel",
        ),
        (
            [
                {
                    "name": "x",
                    "kind": "float",
                    "mean": 0,
                    "std": 1,
                    "conditions": [{"when": "lag(x) > 0", "mean": 1}],
                }
            ],
            "lag is only allowed in formulas",
        ),
    ],
)
def test_levels_and_lags_that_do_not_fit_are_rejected(columns, message):
    with pytest.raises(ValidationError, match=message):
        DatasetSchema.model_validate(small_panel(columns))


def test_correlations_must_join_columns_of_the_same_level():
    spec = small_panel(
        [
            {"name": "a", "kind": "float", "level": "entity", "mean": 0, "std": 1},
            {"name": "b", "kind": "float", "mean": 0, "std": 1},
        ]
    )
    spec["correlations"] = [{"columns": ["a", "b"], "value": 0.5}]

    with pytest.raises(ValidationError, match="different levels"):
        DatasetSchema.model_validate(spec)


def test_a_panel_cannot_have_constraints():
    spec = small_panel([_float("x")])
    spec["constraints"] = ["x > 0"]

    with pytest.raises(ValidationError, match="cannot have constraints"):
        DatasetSchema.model_validate(spec)


def test_panel_features_are_rejected_without_a_panel():
    with pytest.raises(ValidationError, match="needs a panel"):
        DatasetSchema.model_validate({"columns": [{**_float("x"), "level": "entity"}]})
    with pytest.raises(ValidationError, match="lag needs a panel"):
        DatasetSchema.model_validate({"columns": [_float("x"), _formula("y", "lag(x)")]})
    with pytest.raises(ValidationError, match="unknown columns period"):
        DatasetSchema.model_validate({"columns": [_float("x"), _formula("y", "x + period")]})


def test_a_time_split_needs_a_panel_and_a_panel_shift_needs_a_time_split():
    plain = {
        "columns": [_float("x"), _formula("y", "2 * x")],
        "question": {"title": "t", "prompt": "p", "target": "y", "task": "regression"},
    }
    with pytest.raises(ValidationError, match="needs a schema with a panel"):
        DatasetSchema.model_validate({**plain, "question": {**plain["question"], "split": "time"}})
    shifted = small_panel(
        [_float("x"), _formula("y", "2 * x")], test_shift=[{"name": "x", "mean": 1}]
    )
    with pytest.raises(ValidationError, match="needs split: 'time'"):
        DatasetSchema.model_validate(shifted)


def test_an_entity_column_cannot_be_shifted_in_the_test_periods():
    columns = [{**_float("x"), "level": "entity"}, _formula("y", "2 * x")]
    spec = small_panel(columns, split="time", test_shift=[{"name": "x", "mean": 1}])

    with pytest.raises(ValidationError, match="cannot change in the test periods"):
        DatasetSchema.model_validate(spec)


def test_a_time_split_needs_a_training_period():
    spec = small_panel([_float("x"), _formula("y", "2 * x")], split="time", test_share=0.99)

    with pytest.raises(SchemaError, match="no training period"):
        Anonymizer(seed=1).challenge(DatasetSchema.model_validate(spec))


def test_a_panel_cannot_be_learned_from_a_reference_table():
    schema = DatasetSchema.model_validate(small_panel([_float("x")]))

    with pytest.raises(SchemaError, match="from the schema alone"):
        Anonymizer(seed=1).synthesize(50, schema=schema, reference=pd.DataFrame({"x": [1.0, 2.0]}))


def test_the_panel_names_must_differ_and_the_frequency_must_exist():
    base = small_panel([_float("x")])
    with pytest.raises(ValidationError, match="different from each other"):
        DatasetSchema.model_validate({**base, "panel": {**base["panel"], "time": "unit"}})
    with pytest.raises(ValidationError, match="not a pandas offset alias"):
        DatasetSchema.model_validate(
            {**base, "panel": {**base["panel"], "frequency": "fortnightly"}}
        )


def test_the_schema_round_trips_through_json(sales_schema):
    again = DatasetSchema.from_json(sales_schema.to_json())

    assert again == sales_schema


def test_lag_expression_rules():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    invalid = ("lag(a, 0)", "lag(a + 1)", "lag(a, 1, 2)", "lag()", "lag(a, 1.5)", "lag(a, j=2)")
    for text in invalid:
        with pytest.raises(ExpressionError, match="lag in"):
            parse(text)
    with pytest.raises(ExpressionError, match="lag needs a panel"):
        evaluate("lag(a)", frame)
    steps_seen = []

    def record_lags(name: str, steps: int) -> np.ndarray:
        steps_seen.append(steps)
        return np.array([np.nan, 5, 6])

    positional = evaluate("lag(a, 2) + 1", frame, lags=record_lags)
    by_keyword = evaluate("lag(a, k=3) + 1", frame, lags=record_lags)
    default = evaluate("lag(a) + 1", frame, lags=record_lags)

    np.testing.assert_array_equal(positional[1:], [6, 7])
    np.testing.assert_array_equal(by_keyword[1:], [6, 7])
    np.testing.assert_array_equal(default[1:], [6, 7])
    assert steps_seen == [2, 3, 1]


def _float(name: str) -> dict[str, Any]:
    return {"name": name, "kind": "float", "mean": 0, "std": 1}


def _formula(name: str, formula: str) -> dict[str, Any]:
    return {"name": name, "kind": "float", "formula": formula}


@pytest.mark.parametrize(
    "example", sorted(EXAMPLE.parent.glob("*.json")), ids=lambda path: path.name
)
def test_every_example_schema_with_a_question_builds_a_challenge(example):
    schema = DatasetSchema.load(example)
    if schema.question is None:
        pytest.skip("this example has no question")

    challenge = Anonymizer(seed=1).challenge(schema, None if schema.panel else 4000)

    assert len(challenge.train) > 0
    assert challenge.answer_key["target"] == schema.question.target


def test_a_random_split_refuses_published_lags_of_the_target():
    spec = example_json()
    spec["question"]["split"] = "random"
    del spec["question"]["test_shift"]

    with pytest.raises(ValueError, match="last_month_sales read the target of earlier periods"):
        DatasetSchema.model_validate(spec)


def _lag_of_a_hidden_copy(published_formula: str) -> dict[str, Any]:
    columns = [
        {"name": "y", "kind": "float", "mean": 100, "std": 20},
        _formula("copy", "y + normal(0, 0.01)"),
        _formula("second_copy", "lag(copy, 1)"),
        _formula("signal", published_formula),
    ]
    return small_panel(
        columns, split="random", hidden_columns=["copy", "second_copy"], test_share=0.3
    )


@pytest.mark.parametrize("published_formula", ["lag(copy, 1)", "lag(second_copy, 1) + 1"])
def test_a_random_split_refuses_a_lag_of_a_column_that_depends_on_the_target(published_formula):
    with pytest.raises(ValueError, match="signal read the target of earlier periods"):
        DatasetSchema.model_validate(_lag_of_a_hidden_copy(published_formula))


def test_a_random_split_allows_a_lag_of_a_column_unrelated_to_the_target():
    spec = _lag_of_a_hidden_copy("lag(noise, 1)")
    spec["columns"].append({"name": "noise", "kind": "float", "mean": 0, "std": 1})

    DatasetSchema.model_validate(spec)
