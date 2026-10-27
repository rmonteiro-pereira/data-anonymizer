"""A question's difficulty maps to noise, distractors, label noise and drift."""

import pytest

from oblitus import Anonymizer, DatasetSchema
from oblitus.synthesis.difficulty import PROFILES
from oblitus.synthesis.schema import Difficulty

ROWS = 6_000


def regression_spec(**question) -> dict:
    return {
        "columns": [
            {"name": "x", "kind": "float", "mean": 10, "std": 2},
            {"name": "z", "kind": "float", "mean": 0, "std": 1},
            {"name": "y", "kind": "float", "formula": "3 * x + z", "noise_std": 2.0},
        ],
        "question": {
            "title": "t",
            "prompt": "p",
            "target": "y",
            "task": "regression",
            **question,
        },
    }


def challenge_at(level: str | None, **question):
    spec = regression_spec(**question)
    if level is not None:
        spec["question"]["difficulty"] = level
    return Anonymizer(seed=4).challenge(DatasetSchema.model_validate(spec), ROWS)


@pytest.mark.parametrize("level", list(Difficulty))
def test_each_level_adds_its_distractors_and_scales_the_target_noise(level):
    profile = PROFILES[level]

    challenge = challenge_at(level.value)

    notes = challenge.answer_key["difficulty"]
    assert notes["level"] == level.value
    assert len(notes["distractor_columns"]) == profile.distractors
    assert notes["target_noise_std"]["after"] == pytest.approx(2.0 * profile.noise_scale)
    assert set(notes["distractor_columns"]) <= set(challenge.train.columns)
    assert set(notes["distractor_columns"]) <= set(challenge.answer_key["features"]["distractors"])


def test_a_harder_level_leaves_less_to_explain():
    easy = challenge_at("easy").answer_key["best_achievable"]
    hard = challenge_at("hard").answer_key["best_achievable"]

    assert easy["rmse"] < hard["rmse"]
    assert easy["r2"] > hard["r2"]


def test_hard_adds_label_noise_to_the_train_labels_and_drifts_a_driver():
    key = challenge_at("hard").answer_key

    assert key["label_noise"]["split"] == "train"
    assert key["label_noise"]["std"] > 0
    assert key["drift"]["shifted_columns"] == ["x"]
    assert key["drift"]["kind"] == "covariate"
    assert key["difficulty"]["drift"]["mean_after"] == pytest.approx(11.0)


def test_easy_has_no_label_noise_and_no_drift():
    key = challenge_at("easy").answer_key

    assert "label_noise" not in key
    assert "drift" not in key


def test_settings_the_question_gives_itself_win_over_the_level():
    key = challenge_at(
        "hard",
        label_noise={"std": 0.5, "split": "both"},
        test_shift=[{"name": "z", "mean": 2}],
    ).answer_key

    assert key["label_noise"]["std"] == 0.5
    assert key["drift"]["shifted_columns"] == ["z"]


def test_a_target_without_noise_std_is_left_alone_and_the_key_says_so():
    spec = regression_spec(difficulty="hard")
    del spec["columns"][2]["noise_std"]

    key = Anonymizer(seed=4).challenge(DatasetSchema.model_validate(spec), ROWS).answer_key

    assert key["difficulty"]["target_noise_std"].startswith("not applied")


def test_a_question_without_a_level_is_not_changed():
    challenge = challenge_at(None)

    assert "difficulty" not in challenge.answer_key
    assert list(challenge.train.columns) == ["x", "z", "y"]
