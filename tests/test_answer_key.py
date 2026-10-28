"""The answer key names each column's role truly and says what its best score assumes."""

import json
from pathlib import Path

import pytest

from oblitus import Anonymizer, DatasetSchema

CHURN_EXAMPLE = Path(__file__).parent.parent / "examples" / "churn_challenge.json"
QUESTION = {"title": "Fit y", "prompt": "Predict y.", "target": "y", "task": "regression"}
PANEL = {
    "entity": "unit",
    "entities": 5,
    "time": "day",
    "start": "2024-01-01",
    "periods": 10,
    "frequency": "D",
}


def churn_spec(**question) -> dict:
    spec = json.loads(CHURN_EXAMPLE.read_text())
    spec["question"].update(question)
    return spec


def _key(columns: list[dict], **extra) -> dict:
    spec = {"columns": columns, "question": {**QUESTION, **extra.pop("question", {})}, **extra}
    schema = DatasetSchema.model_validate(spec)
    return Anonymizer(seed=3).challenge(schema, 4000).answer_key


def test_blanks_that_follow_the_target_are_a_leak():
    key = _key(
        [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "float", "formula": "2 * x", "noise_std": 1},
            {
                "name": "income",
                "kind": "float",
                "mean": 10,
                "std": 2,
                "null_fraction": 0.3,
                "missingness": {"mechanism": "mar", "depends_on": "y", "strength": 2},
            },
        ]
    )

    assert key["features"]["leaks_target"] == ["income"]
    assert "income" not in key["features"]["distractors"]


def test_a_noisy_proxy_of_a_hidden_driver_moves_with_it_and_its_score_is_unreachable():
    key = _key(
        [
            {"name": "skill", "kind": "float", "mean": 0, "std": 1},
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "proxy", "kind": "float", "formula": "skill", "noise_std": 0.5},
            {"name": "y", "kind": "float", "formula": "3 * skill + 2 * x", "noise_std": 1},
        ],
        question={"hidden_columns": ["skill"]},
    )

    assert key["features"]["correlated_but_not_causal"] == ["proxy"]
    assert any("skill" in caveat for caveat in key["best_achievable_assumes"])


def test_a_column_correlated_with_a_drawn_target_is_not_a_distractor():
    key = _key(
        [
            {"name": "x", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "float", "mean": 0, "std": 1},
        ],
        correlations=[{"columns": ["x", "y"], "value": 0.8}],
    )

    assert key["features"]["correlated_but_not_causal"] == ["x"]
    assert key["features"]["distractors"] == []


def test_a_driver_with_blanks_is_named_in_what_the_best_score_assumes():
    key = _key(
        [
            {"name": "x", "kind": "float", "mean": 0, "std": 1, "null_fraction": 0.3},
            {"name": "y", "kind": "float", "formula": "2 * x", "noise_std": 1},
        ]
    )

    assert any("complete values of x" in caveat for caveat in key["best_achievable_assumes"])


def test_a_random_split_of_a_panel_warns_that_neighbouring_labels_beat_the_best_score():
    columns = [
        {"name": "x", "kind": "float", "mean": 0, "std": 1},
        {"name": "y", "kind": "float", "noise_std": 1, "formula": "0.5 * x + 0.7 * lag(y)"},
    ]
    spec = {"panel": PANEL, "columns": columns, "question": {**QUESTION, "test_share": 0.3}}
    schema = DatasetSchema.model_validate(spec)

    key = Anonymizer(seed=1).challenge(schema).answer_key

    assert any("time split" in caveat for caveat in key["best_achievable_assumes"])


def test_a_hidden_probability_computed_from_published_drivers_adds_no_caveat():
    schema = DatasetSchema.model_validate(churn_spec())

    key = Anonymizer(seed=4).challenge(schema, 3000).answer_key

    assert key["features"]["hidden_drivers"] == ["churn_probability"]
    assert "best_achievable_assumes" not in key


@pytest.mark.parametrize("share", [0.2, 0.8])
def test_majority_class_accuracy_is_the_share_of_the_more_common_class(share):
    schema = DatasetSchema.model_validate(churn_spec(positive_share=share))

    balance = Anonymizer(seed=2).challenge(schema, 20_000).answer_key["class_balance"]

    test_rate = balance["positive_rate"]["test"]
    assert balance["majority_class_accuracy"] == pytest.approx(max(test_rate, 1 - test_rate))
    assert balance["majority_class_accuracy"] > 0.7
