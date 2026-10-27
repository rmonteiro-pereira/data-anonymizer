"""Causal challenges: the answer key's effects, roles and leakage flags match what was planted."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from data_anonymizer import Anonymizer, DatasetSchema
from data_anonymizer.synthesis.causal_graph import CausalGraph

EXAMPLE = Path(__file__).parent.parent / "examples" / "promo_uplift.json"
ROWS = 40_000

# The planted effect of the promo on one customer in examples/promo_uplift.json: 6 more app
# minutes worth 2 revenue each, plus a direct effect that depends on segment and age.
MEDIATED_EFFECT = 2 * 6


def planted_effect(customers: pd.DataFrame) -> np.ndarray:
    direct = (
        5
        + np.where(customers["segment"] == "loyal", 6, 0)
        - np.where(customers["segment"] == "new", 2, 0)
        + 0.2 * (customers["age"] - 42)
    )
    return MEDIATED_EFFECT + direct.to_numpy()


def regress(columns: list[np.ndarray], outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with an intercept first; returns coefficients and standard errors."""
    design = np.column_stack([np.ones(len(outcome)), *columns])
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    variance = residuals @ residuals / (len(outcome) - design.shape[1])
    errors = np.sqrt(np.diag(variance * np.linalg.inv(design.T @ design)))
    return coefficients, errors


@pytest.fixture(scope="module")
def promo_challenge():
    schema = DatasetSchema.load(EXAMPLE)
    return Anonymizer(seed=11).challenge(schema, ROWS)


@pytest.fixture(scope="module")
def causal(promo_challenge) -> dict[str, Any]:
    return promo_challenge.answer_key["causal"]


@pytest.fixture(scope="module")
def train(promo_challenge) -> pd.DataFrame:
    return promo_challenge.train


def adjusted_effect(train: pd.DataFrame, extra: list[str] = ()) -> tuple[float, float]:
    """The promo coefficient when adjusting for past spend and the modifiers, plus `extra`."""
    promo = train["promo"].to_numpy(dtype=float)
    loyal = (train["segment"] == "loyal").to_numpy(dtype=float)
    new = (train["segment"] == "new").to_numpy(dtype=float)
    age = train["age"].to_numpy(dtype=float)
    centered = [loyal - loyal.mean(), new - new.mean(), age - age.mean()]
    interactions = [promo * column for column in centered]
    others = [train[name].to_numpy(dtype=float) for name in ["past_spend", *extra]]
    design = [promo, *others, *centered, *interactions]
    coefficients, errors = regress(design, train["revenue_next_month"].to_numpy())
    return float(coefficients[1]), float(errors[1])


def test_the_true_average_effect_is_the_planted_one(promo_challenge, causal):
    customers = pd.concat([promo_challenge.train, promo_challenge.test_features])
    treated = customers["promo"].to_numpy(dtype=bool)
    effect = planted_effect(customers)

    assert causal["ate"] == pytest.approx(effect.mean(), abs=1e-6)
    assert causal["att"] == pytest.approx(effect[treated].mean(), abs=1e-6)
    assert causal["treatment_values"] == {"control": False, "treated": True}


def test_conditional_effects_follow_the_planted_modifiers(promo_challenge, causal):
    by_segment = {item["level"]: item["effect"] for item in causal["cate"]["segment"]}
    age_bins = causal["cate"]["age"]

    assert by_segment["loyal"] - by_segment["returning"] == pytest.approx(6, abs=0.5)
    assert by_segment["returning"] - by_segment["new"] == pytest.approx(2, abs=0.5)
    assert len(age_bins) == 4
    assert [item["effect"] for item in age_bins] == sorted(item["effect"] for item in age_bins)
    assert sum(item["rows"] for item in age_bins) == ROWS


def test_regression_with_the_adjustment_set_recovers_the_average_effect(train, causal):
    assert causal["adjustment_set"] == ["past_spend"]

    estimate, error = adjusted_effect(train)

    assert abs(estimate - causal["ate"]) < 4 * error


def test_the_naive_difference_is_biased_by_the_confounder(train, causal):
    treated = train["promo"].to_numpy(dtype=bool)
    revenue = train["revenue_next_month"].to_numpy()
    naive = revenue[treated].mean() - revenue[~treated].mean()

    assert causal["naive_difference"] == pytest.approx(naive)
    assert causal["naive_bias"] == pytest.approx(naive - causal["ate"])
    assert causal["naive_bias"] > 5
    spend = train["past_spend"].to_numpy()
    assert spend[treated].mean() > spend[~treated].mean() + 5


def test_the_confounder_drives_both_promo_and_revenue(train, causal):
    assert causal["roles"]["confounders"] == ["past_spend"]
    promo = train["promo"].to_numpy(dtype=float)
    spend = train["past_spend"].to_numpy()

    first_stage, first_errors = regress([spend], promo)
    assert first_stage[1] > 4 * first_errors[1]
    revenue = train["revenue_next_month"].to_numpy()
    outcome_side, outcome_errors = regress([promo, spend], revenue)
    assert outcome_side[2] > 4 * outcome_errors[2]


def test_adjusting_for_the_mediator_removes_the_mediated_share(train, causal):
    assert causal["roles"]["mediators"] == ["app_minutes"]

    estimate, error = adjusted_effect(train, extra=["app_minutes"])

    direct_effect = causal["ate"] - MEDIATED_EFFECT
    assert abs(estimate - direct_effect) < 4 * error
    assert abs(estimate - causal["ate"]) > 5


def test_adjusting_for_the_collider_biases_the_estimate(train, causal):
    assert causal["roles"]["colliders"] == ["satisfaction_score"]

    estimate, error = adjusted_effect(train, extra=["satisfaction_score"])

    assert abs(estimate - causal["ate"]) > 5
    assert abs(estimate - causal["ate"]) > 10 * error


def test_the_instrument_moves_the_promo_and_nothing_else(train, causal):
    assert causal["roles"]["instruments"] == ["outreach_wave"]
    promo = train["promo"].to_numpy(dtype=float)
    spend = train["past_spend"].to_numpy()
    wave = train["outreach_wave"].to_numpy(dtype=float)
    revenue = train["revenue_next_month"].to_numpy()

    relevance, relevance_errors = regress([spend, wave], promo)
    direct, direct_errors = regress([promo, spend, wave], revenue)

    assert relevance[2] > 10 * relevance_errors[2]
    assert abs(direct[3]) < 4 * direct_errors[3]


def test_the_instrumental_variable_estimate_recovers_the_average_effect(train, causal):
    promo = train["promo"].to_numpy(dtype=float)
    spend = train["past_spend"].to_numpy()
    wave = train["outreach_wave"].to_numpy(dtype=float)
    revenue = train["revenue_next_month"].to_numpy()

    first_stage = np.column_stack([np.ones(len(promo)), spend, wave])
    fitted = first_stage @ np.linalg.lstsq(first_stage, promo, rcond=None)[0]
    stage_two = np.column_stack([np.ones(len(promo)), spend, fitted])
    coefficients = np.linalg.lstsq(stage_two, revenue, rcond=None)[0]
    actual = np.column_stack([np.ones(len(promo)), spend, promo])
    residuals = revenue - actual @ coefficients
    variance = residuals @ residuals / (len(revenue) - 3)
    error = np.sqrt(variance * np.linalg.inv(stage_two.T @ stage_two)[2, 2])

    assert abs(coefficients[2] - causal["ate"]) < 4 * error


def test_effect_modifiers_are_listed_and_a_hidden_node_still_counts_in_the_graph(causal):
    assert causal["roles"]["effect_modifiers"] == ["segment", "age"]
    assert causal["adjustment_set_reason"]


def test_columns_that_read_the_target_are_flagged_as_leaks(promo_challenge, train):
    features = promo_challenge.answer_key["features"]

    assert features["leaks_target"] == ["satisfaction_score", "reward_points_earned"]
    for name in features["leaks_target"]:
        assert name not in features["true_drivers"]
        assert name not in features["distractors"]
    points = train["reward_points_earned"].to_numpy()
    assert np.corrcoef(points, train["revenue_next_month"].to_numpy())[0, 1] > 0.95


def test_the_example_publishes_no_hidden_column_but_keeps_it_in_the_graph(promo_challenge):
    assert "promo_propensity" not in promo_challenge.train.columns
    assert "promo_propensity" in promo_challenge.answer_key["features"]["hidden_drivers"]


def test_the_answer_key_is_json_serialisable(promo_challenge):
    json.dumps(promo_challenge.answer_key)


def graph(parents: dict[str, list[str]], correlated: list[tuple[str, str]] = ()) -> CausalGraph:
    return CausalGraph(parents, correlated)


def test_a_correlated_column_that_causes_the_outcome_is_a_confounder():
    structure = graph({"c": [], "t": [], "y": ["t", "c"]}, [("c", "t")])

    roles = structure.roles("t", "y")
    adjustment = structure.adjustment_set("t", "y", {"c", "t", "y"})

    assert roles.confounders == ["c"]
    assert adjustment.columns == ["c"]


def test_a_hidden_confounder_leaves_no_adjustment_set():
    structure = graph({"u": [], "t": ["u"], "y": ["t", "u"]})

    adjustment = structure.adjustment_set("t", "y", {"t", "y"})

    assert adjustment.columns is None
    assert "u" in adjustment.reason


def test_a_common_cause_shared_by_treatment_and_outcome_blocks_adjustment():
    structure = graph({"t": [], "y": ["t"]}, [("t", "y")])

    adjustment = structure.adjustment_set("t", "y", {"t", "y"})

    assert adjustment.columns is None
    assert "unobserved" in adjustment.reason


def test_an_instrument_connected_to_the_outcome_by_a_correlation_is_not_one():
    structure = graph({"z": [], "w": [], "t": ["z"], "y": ["t", "w"]}, [("z", "w")])

    assert structure.roles("t", "y").instruments == []


def test_a_column_that_only_reads_the_outcome_is_not_a_collider():
    structure = graph({"t": [], "y": ["t"], "after": ["y"], "both": ["t", "y"]})

    roles = structure.roles("t", "y")

    assert roles.colliders == ["both"]
    assert roles.mediators == []


def test_without_confounding_the_adjustment_set_is_empty():
    structure = graph({"t": [], "y": ["t"]})

    adjustment = structure.adjustment_set("t", "y", {"t", "y"})

    assert adjustment.columns == []


def small_schema(**question: Any) -> dict[str, Any]:
    """A confounded binary treatment with a probability behind a binary outcome."""
    return {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "t", "kind": "boolean", "formula": "bernoulli(sigmoid(x))"},
            {"name": "p", "kind": "float", "formula": "sigmoid(-0.5 + 0.8 * x + 1.0 * t)"},
            {"name": "y", "kind": "boolean", "formula": "bernoulli(p)"},
        ],
        "question": {
            "title": "t",
            "prompt": "p",
            "target": "y",
            "task": "classification",
            "hidden_columns": ["p"],
            "truth_column": "p",
            "treatment": "t",
            **question,
        },
    }


def test_a_binary_outcome_effect_matches_the_effect_on_its_probability():
    schema = DatasetSchema.model_validate(small_schema())

    causal = Anonymizer(seed=5).challenge(schema, 60_000).answer_key["causal"]

    on_probability = causal["effect_on_truth_column"]
    assert on_probability["column"] == "p"
    assert abs(causal["ate"] - on_probability["ate"]) < 0.01
    assert on_probability["ate"] > 0.1
    assert on_probability["att"] is not None


def test_a_category_treatment_needs_its_control_and_treated_values():
    schema = small_schema(treatment_values=["no", "yes"])
    schema["columns"][1] = {
        "name": "t",
        "kind": "category",
        "choices": ["no", "yes"],
        "formula": "where(bernoulli(sigmoid(x)), 'yes', 'no')",
    }
    schema["columns"][2]["formula"] = "sigmoid(-0.5 + 0.8 * x + 1.0 * (t == 'yes'))"

    causal = (
        Anonymizer(seed=5)
        .challenge(DatasetSchema.model_validate(schema), 20_000)
        .answer_key["causal"]
    )

    assert causal["treatment_values"] == {"control": "no", "treated": "yes"}
    assert causal["ate"] > 0.1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"treatment": "missing"}, "not a column"),
        ({"treatment": "y"}, "cannot be its target"),
        ({"treatment": "p"}, "cannot be hidden"),
        ({"treatment": "x"}, "treatment_values"),
        ({"treatment_values": ["a", "b", "c"]}, "two values"),
        ({"treatment_values": [True, True]}, "must differ"),
        ({"effect_modifiers": ["y"]}, "must not be"),
        ({"effect_modifiers": ["missing"]}, "not a column"),
    ],
)
def test_invalid_causal_questions_are_rejected(change, message):
    with pytest.raises(ValidationError, match=message):
        DatasetSchema.model_validate(small_schema(**change))


def test_a_treatment_caused_by_the_target_is_rejected():
    schema = small_schema()
    schema["columns"][1]["formula"] = "bernoulli(sigmoid(x)) or y"
    schema["columns"][3]["formula"] = "bernoulli(sigmoid(x))"

    with pytest.raises(ValidationError, match="caused by the target"):
        DatasetSchema.model_validate(schema)


def test_effect_modifiers_need_a_treatment():
    schema = small_schema()
    del schema["question"]["treatment"]
    schema["question"]["effect_modifiers"] = ["x"]

    with pytest.raises(ValidationError, match="need a treatment"):
        DatasetSchema.model_validate(schema)


def test_a_zero_one_integer_treatment_needs_no_values():
    schema = small_schema()
    schema["columns"][1] = {
        "name": "t",
        "kind": "integer",
        "minimum": 0,
        "maximum": 1,
        "formula": "bernoulli(sigmoid(x))",
    }
    schema["columns"][2]["formula"] = "sigmoid(-0.5 + 0.8 * x + 1.0 * t)"

    causal = (
        Anonymizer(seed=5)
        .challenge(DatasetSchema.model_validate(schema), 5_000)
        .answer_key["causal"]
    )

    assert causal["treatment_values"] == {"control": 0, "treated": 1}


def test_planted_leaks_are_flagged_from_a_formula_that_reads_the_target():
    schema = small_schema()
    schema["question"].pop("treatment")
    schema["columns"].append(
        {"name": "refund_issued", "kind": "boolean", "formula": "y and bernoulli(0.7)"}
    )
    schema["columns"].append({"name": "noise", "kind": "float", "mean": 0, "std": 1})

    features = (
        Anonymizer(seed=5)
        .challenge(DatasetSchema.model_validate(schema), 4_000)
        .answer_key["features"]
    )

    assert features["leaks_target"] == ["refund_issued"]
    assert features["distractors"] == ["noise"]
    assert "refund_issued" not in features["true_drivers"]


def test_the_question_page_names_the_treatment(promo_challenge):
    from data_anonymizer.cli import _question_page

    page = _question_page(promo_challenge)

    assert "Treatment column: `promo` (control False, treated True)." in page
