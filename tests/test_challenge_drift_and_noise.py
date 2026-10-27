"""Drift between train and test, and label noise, in generated challenges."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import optimize, special

from oblitus import Anonymizer, DatasetSchema

EXAMPLES = Path(__file__).parent.parent / "examples"
ROWS = 30000


def _load(name: str) -> dict:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _challenge(schema_json: dict, seed: int = 3, rows: int = ROWS):
    schema = DatasetSchema.model_validate(schema_json)
    return Anonymizer(seed=seed).challenge(schema, rows)


def _with_question(name: str, **changes) -> dict:
    schema_json = _load(name)
    schema_json["question"] = {**schema_json["question"], **changes}
    return schema_json


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_a_schema_without_drift_or_noise_gives_the_same_output_as_before():
    challenge = _challenge(_load("churn_challenge.json"), seed=4, rows=3000)

    # The leakage flag list is a newer, additive field, empty when no column leaks.
    assert challenge.answer_key["features"]["leaks_target"] == []
    # Byte-for-byte equality with the earlier output is pinned in test_draw_replay.py.
    assert "drift" not in challenge.answer_key
    assert "label_noise" not in challenge.answer_key


def test_a_covariate_shift_moves_the_test_distribution_as_configured():
    challenge = _challenge(_load("churn_drift.json"))
    key = challenge.answer_key

    assert challenge.train["age"].mean() == pytest.approx(41.5, abs=1.0)
    assert challenge.test_features["age"].mean() == pytest.approx(31, abs=1.0)
    assert challenge.train["support_tickets"].mean() == pytest.approx(1.5, abs=0.05)
    assert challenge.test_features["support_tickets"].mean() == pytest.approx(2.5, abs=0.1)
    free_share = (challenge.test_features["plan"] == "free").mean()
    assert free_share == pytest.approx(0.75, abs=0.03)
    assert key["drift"]["kind"] == "covariate"
    assert key["drift"]["shifted_columns"] == ["age", "support_tickets", "plan"]
    assert key["rows"]["test"] == len(challenge.test_features)


def test_shifting_only_a_distractor_is_still_a_covariate_shift():
    schema_json = _with_question(
        "churn_drift.json", test_shift=[{"name": "region", "weights": [1, 0, 0, 0]}]
    )

    challenge = _challenge(schema_json)

    assert set(challenge.test_features["region"]) == {"north"}
    assert challenge.train["region"].nunique() == 4
    assert challenge.answer_key["drift"]["kind"] == "covariate"


def test_a_new_distribution_for_a_driver_is_not_a_concept_shift():
    schema_json = _with_question(
        "churn_drift.json", test_shift=[{"name": "tenure_months", "mean": 60, "std": 10}]
    )

    assert _challenge(schema_json).answer_key["drift"]["kind"] == "covariate"


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])

    def loss(weights: np.ndarray) -> float:
        logits = design @ weights
        return float(np.mean(np.logaddexp(0, logits) - labels * logits))

    return optimize.minimize(loss, np.zeros(design.shape[1]), method="BFGS").x


def _churn_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["support_tickets"],
            frame["tenure_months"],
            (frame["plan"] == "free").astype(float),
            frame["monthly_spend"],
        ]
    )


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def _train_fitted_gap(schema_json: dict) -> tuple[float, dict]:
    """Return the fitted model's test log loss minus the key's best, and the key."""
    challenge = _challenge(schema_json)
    weights = _fit_logistic(
        _churn_features(challenge.train), challenge.train["churned"].to_numpy(dtype=float)
    )
    design = np.column_stack(
        [np.ones(len(challenge.test_features)), _churn_features(challenge.test_features)]
    )
    predicted = special.expit(design @ weights)
    answers = challenge.test_answers["churned"].to_numpy(dtype=float)
    best = challenge.answer_key["best_achievable"]["log_loss"]
    return _log_loss(answers, predicted) - best, challenge.answer_key


def test_a_concept_shift_hurts_the_train_fitted_model_and_the_key_uses_the_shifted_truth():
    no_shift = _with_question("churn_concept_drift.json", test_shift=[])

    control_gap, _ = _train_fitted_gap(no_shift)
    concept_gap, key = _train_fitted_gap(_load("churn_concept_drift.json"))

    assert key["drift"]["kind"] == "concept"
    assert control_gap < 0.02
    assert concept_gap > 0.05
    assert concept_gap > 4 * control_gap


def test_best_achievable_under_a_concept_shift_is_the_shifted_probability():
    challenge = _challenge(_load("churn_concept_drift.json"))
    test = challenge.test_features
    answers = challenge.test_answers["churned"].to_numpy(dtype=float)
    shifted = special.expit(
        -1.2
        + 1.1 * test["support_tickets"]
        - 0.04 * test["tenure_months"]
        + np.where(test["plan"] == "free", 0.8, 0)
        - 0.004 * test["monthly_spend"]
    )
    base = special.expit(
        -1.2
        + 0.55 * test["support_tickets"]
        - 0.04 * test["tenure_months"]
        + np.where(test["plan"] == "free", 0.8, 0)
        - 0.004 * test["monthly_spend"]
    )

    best = challenge.answer_key["best_achievable"]["log_loss"]

    assert best == pytest.approx(_log_loss(answers, shifted.to_numpy()), rel=1e-6)
    assert _log_loss(answers, base.to_numpy()) > best + 0.05
    assert (
        challenge.answer_key["drift"]["best_achievable"] == challenge.answer_key["best_achievable"]
    )


def test_a_concept_shift_of_a_driver_formula_or_the_target_is_reported_as_concept():
    schema_json = _load("churn_challenge.json")
    schema_json["question"]["test_shift"] = [{"name": "churned", "true_share": 0.5}]
    del schema_json["columns"][-1]["formula"]
    schema_json["columns"][-1]["true_share"] = 0.3

    assert _challenge(schema_json, rows=2000).answer_key["drift"]["kind"] == "concept"


def test_shifts_that_name_unknown_or_repeated_columns_are_rejected():
    unknown = _with_question("churn_drift.json", test_shift=[{"name": "missing", "mean": 1}])
    repeated = _with_question(
        "churn_drift.json", test_shift=[{"name": "age", "mean": 1}, {"name": "age", "mean": 2}]
    )
    row_filter = _with_question(
        "churn_drift.json", test_shift=[{"name": "age", "mean": 1, "when": "age > 3"}]
    )

    for schema_json, message in (
        (unknown, "unknown column"),
        (repeated, "twice"),
        (row_filter, "whole test set"),
    ):
        with pytest.raises(ValidationError, match=message):
            DatasetSchema.model_validate(schema_json)


def test_a_shift_that_makes_formulas_cyclic_is_rejected():
    schema_json = _with_question(
        "churn_drift.json",
        test_shift=[{"name": "monthly_spend", "formula": "churn_probability * 10"}],
    )

    with pytest.raises(ValidationError, match="cycle"):
        DatasetSchema.model_validate(schema_json)


def _without_noise(name: str) -> dict:
    schema_json = _load(name)
    del schema_json["question"]["label_noise"]
    return schema_json


def test_train_label_noise_flips_the_configured_share_and_leaves_test_answers_clean():
    noisy = _challenge(_load("churn_drift.json"))
    clean = _challenge(_without_noise("churn_drift.json"))

    train_rows = len(noisy.train)
    flipped = int((noisy.train["churned"] != clean.train["churned"]).sum())
    spread = np.sqrt(train_rows * 0.1 * 0.9)
    report = noisy.answer_key["label_noise"]

    assert report["flipped"] == {"train": flipped, "test": 0}
    assert abs(flipped - 0.1 * train_rows) < 4 * spread
    assert noisy.test_answers.equals(clean.test_answers)
    assert noisy.test_features.equals(clean.test_features)
    assert noisy.answer_key["best_achievable"] == clean.answer_key["best_achievable"]
    assert report["best_achievable_on_clean_test_labels"] == clean.answer_key["best_achievable"]


@pytest.mark.parametrize("split", ["test", "both"])
def test_test_label_noise_reaches_the_answers_but_the_key_keeps_the_clean_score(split):
    noisy_json = _with_question("churn_drift.json", label_noise={"flip_rate": 0.2, "split": split})
    clean = _challenge(_without_noise("churn_drift.json"))

    noisy = _challenge(noisy_json)

    test_rows = len(noisy.test_answers)
    flipped_in_test = int((noisy.test_answers["churned"] != clean.test_answers["churned"]).sum())
    report = noisy.answer_key["label_noise"]
    spread = np.sqrt(test_rows * 0.2 * 0.8)
    assert report["flipped"]["test"] == flipped_in_test
    assert abs(flipped_in_test - 0.2 * test_rows) < 4 * spread
    assert (report["flipped"]["train"] > 0) == (split == "both")
    assert report["best_achievable_on_clean_test_labels"] == clean.answer_key["best_achievable"]
    assert (
        noisy.answer_key["best_achievable"]["accuracy"]
        < clean.answer_key["best_achievable"]["accuracy"]
    )


def test_regression_label_noise_perturbs_the_train_labels_by_the_configured_std():
    schema_json = {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "float", "formula": "3 * x + 1", "noise_std": 0.5},
        ],
        "question": {"title": "Fit y", "prompt": "Predict y.", "target": "y", "task": "regression"},
    }
    noisy_json = json.loads(json.dumps(schema_json))
    noisy_json["question"]["label_noise"] = {"std": 5.0}

    clean = _challenge(schema_json, rows=20000)
    noisy = _challenge(noisy_json, rows=20000)

    added = noisy.train["y"] - clean.train["y"]
    assert added.std() == pytest.approx(5.0, rel=0.03)
    assert added.mean() == pytest.approx(0.0, abs=0.1)
    assert noisy.answer_key["label_noise"]["perturbed"] == {"train": len(noisy.train), "test": 0}
    assert noisy.test_answers.equals(clean.test_answers)
    assert noisy.answer_key["best_achievable"]["rmse"] == pytest.approx(0.5, rel=0.1)


def test_integer_targets_stay_integer_under_regression_noise():
    schema_json = {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "integer", "formula": "10 * x", "noise_std": 1},
        ],
        "question": {
            "title": "Fit y",
            "prompt": "Predict y.",
            "target": "y",
            "task": "regression",
            "label_noise": {"std": 3.0},
        },
    }

    challenge = _challenge(schema_json, rows=2000)

    assert pd.api.types.is_integer_dtype(challenge.train["y"].dtype)


def test_label_noise_must_match_the_task():
    flips_on_regression = {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "float", "formula": "x"},
        ],
        "question": {
            "title": "t",
            "prompt": "p",
            "target": "y",
            "task": "regression",
            "label_noise": {"flip_rate": 0.1},
        },
    }
    std_on_classification = _with_question("churn_drift.json", label_noise={"std": 1.0})
    both_kinds = _with_question("churn_drift.json", label_noise={"flip_rate": 0.1, "std": 1.0})
    flip_rate_too_high = _with_question("churn_drift.json", label_noise={"flip_rate": 0.8})

    for schema_json in (flips_on_regression, std_on_classification, both_kinds, flip_rate_too_high):
        with pytest.raises(ValidationError):
            DatasetSchema.model_validate(schema_json)


def test_a_two_choice_category_target_flips_to_the_other_choice():
    schema_json = {
        "columns": [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "p", "kind": "float", "formula": "sigmoid(2 * x)"},
            {"name": "answer", "kind": "category", "choices": ["no", "yes"], "weights": [1, 1]},
        ],
        "question": {
            "title": "t",
            "prompt": "p",
            "target": "answer",
            "task": "classification",
            "positive_label": "yes",
            "label_noise": {"flip_rate": 0.25},
        },
    }
    clean_json = json.loads(json.dumps(schema_json))
    del clean_json["question"]["label_noise"]

    noisy = _challenge(schema_json, rows=10000)
    clean = _challenge(clean_json, rows=10000)

    flipped = int((noisy.train["answer"] != clean.train["answer"]).sum())
    assert flipped == noisy.answer_key["label_noise"]["flipped"]["train"]
    assert flipped == pytest.approx(0.25 * len(noisy.train), rel=0.1)
    assert set(noisy.train["answer"]) == {"no", "yes"}
