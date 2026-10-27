"""A question's positive_share sets how rare the positive class is, and the key reports it."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from oblitus import Anonymizer, DatasetSchema
from oblitus.synthesis.class_balance import average_precision
from oblitus.synthesis.logistic_calibration import calibrate_probabilities

CHURN_EXAMPLE = Path(__file__).parent.parent / "examples" / "churn_challenge.json"


def churn_spec(**question) -> dict:
    spec = json.loads(CHURN_EXAMPLE.read_text())
    spec["question"].update(question)
    return spec


def test_the_positive_rate_of_the_published_rows_meets_the_requested_share():
    schema = DatasetSchema.model_validate(churn_spec(positive_share=0.04))

    challenge = Anonymizer(seed=2).challenge(schema, 30_000)

    everything = np.concatenate(
        [challenge.train["churned"].to_numpy(), challenge.test_answers["churned"].to_numpy()]
    )
    assert everything.mean() == pytest.approx(0.04, abs=0.006)


def test_the_key_reports_the_baseline_and_the_best_achievable_pr_auc():
    schema = DatasetSchema.model_validate(churn_spec(positive_share=0.05))

    key = Anonymizer(seed=2).challenge(schema, 30_000).answer_key["class_balance"]

    assert key["requested_positive_share"] == 0.05
    assert key["pr_auc_baseline"] == pytest.approx(key["positive_rate"]["test"])
    assert key["majority_class_accuracy"] == pytest.approx(1 - key["positive_rate"]["test"])
    assert key["best_achievable_pr_auc"] > 2 * key["pr_auc_baseline"]


def test_a_question_without_a_share_has_no_class_balance_section():
    schema = DatasetSchema.model_validate(churn_spec())

    assert "class_balance" not in Anonymizer(seed=2).challenge(schema, 2_000).answer_key


@pytest.mark.parametrize(
    "question, message",
    [
        ({"task": "regression", "positive_share": 0.1}, "classification"),
        ({"truth_column": None, "positive_share": 0.1}, "truth_column"),
        ({"positive_share": 1.5}, "less than 1"),
    ],
)
def test_a_positive_share_needs_a_classification_question_with_a_truth_column(question, message):
    spec = churn_spec(**question)
    if question.get("truth_column", 1) is None:
        del spec["question"]["truth_column"]

    with pytest.raises(ValidationError, match=message):
        DatasetSchema.model_validate(spec)


def test_a_target_that_ignores_the_truth_column_cannot_take_a_share():
    spec = churn_spec(positive_share=0.1)
    spec["columns"][-1]["formula"] = "bernoulli(0.3)"

    with pytest.raises(ValidationError, match="does not read the truth_column"):
        DatasetSchema.model_validate(spec)


def test_calibration_keeps_the_order_of_the_rows():
    probabilities = np.array([0.1, 0.4, 0.5, 0.9])

    calibrated = calibrate_probabilities(probabilities, 0.1)

    assert calibrated.mean() == pytest.approx(0.1, abs=1e-6)
    assert (np.diff(calibrated) > 0).all()


def test_average_precision_is_one_for_a_perfect_ranking_and_ignores_tie_order():
    labels = np.array([1.0, 1.0, 0.0, 0.0])

    assert average_precision(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1.0)
    tied = average_precision(labels, np.array([0.5, 0.5, 0.5, 0.5]))
    assert tied == pytest.approx(0.5)
