"""Blank cells that follow the data: MNAR on the column itself, MAR on another column."""

import numpy as np
import pytest
from pydantic import ValidationError

from oblitus import Anonymizer, DatasetSchema
from oblitus.surrogates import SurrogateFactory
from oblitus.synthesis.generator import generate_with_replay

ROWS = 20_000


def schema_with(missingness: dict, null_fraction: float = 0.2) -> DatasetSchema:
    return DatasetSchema.model_validate(
        {
            "columns": [
                {"name": "x", "kind": "float", "mean": 0, "std": 1},
                {
                    "name": "y",
                    "kind": "float",
                    "mean": 50,
                    "std": 10,
                    "null_fraction": null_fraction,
                    "missingness": missingness,
                },
            ]
        }
    )


def build(schema: DatasetSchema):
    return generate_with_replay(
        schema, ROWS, rng=np.random.default_rng(5), surrogates=SurrogateFactory("en_US")
    )


def test_mnar_blanks_large_values_more_often_and_keeps_the_overall_share():
    built = build(schema_with({"mechanism": "mnar", "strength": 2.0}))

    blank = built.table["y"].isna()

    assert blank.mean() == pytest.approx(0.2, abs=0.01)
    assert built.structural["y"][blank].mean() > built.structural["y"][~blank].mean() + 5


def test_a_negative_strength_blanks_small_values_more_often():
    built = build(schema_with({"mechanism": "mnar", "strength": -2.0}))

    blank = built.table["y"].isna()

    assert built.structural["y"][blank].mean() < built.structural["y"][~blank].mean() - 5


def test_mar_follows_another_column_and_not_the_cells_own_value():
    built = build(schema_with({"mechanism": "mar", "depends_on": "x", "strength": 2.0}))

    blank = built.table["y"].isna()

    assert blank.mean() == pytest.approx(0.2, abs=0.01)
    assert built.structural["x"][blank].mean() > built.structural["x"][~blank].mean() + 0.5
    assert abs(built.structural["y"][blank].mean() - built.structural["y"][~blank].mean()) < 0.5


def test_a_plain_null_fraction_is_unchanged_by_the_new_setting_being_absent():
    plain = DatasetSchema.model_validate(
        {"columns": [{"name": "y", "kind": "float", "mean": 0, "std": 1, "null_fraction": 0.3}]}
    )

    blank = build(plain).table["y"].isna()

    assert blank.mean() == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize(
    "missingness, null_fraction, message",
    [
        ({"mechanism": "mar"}, 0.2, "needs depends_on"),
        ({"mechanism": "mnar", "depends_on": "x"}, 0.2, "leave depends_on out"),
        ({"mechanism": "mnar"}, 0.0, "null_fraction above zero"),
        ({"mechanism": "mar", "depends_on": "nowhere"}, 0.2, "unknown column"),
    ],
)
def test_a_missingness_setting_must_be_complete(missingness, null_fraction, message):
    with pytest.raises(ValidationError, match=message):
        schema_with(missingness, null_fraction)


def test_a_mnar_column_must_be_ordered():
    with pytest.raises(ValidationError, match="number, a boolean or a date"):
        DatasetSchema.model_validate(
            {
                "columns": [
                    {
                        "name": "plan",
                        "kind": "category",
                        "choices": ["a", "b"],
                        "null_fraction": 0.1,
                        "missingness": {"mechanism": "mnar"},
                    }
                ]
            }
        )


def test_the_answer_key_records_each_mechanism():
    schema = DatasetSchema.model_validate(
        {
            "columns": [
                {"name": "x", "kind": "float", "mean": 0, "std": 1},
                {
                    "name": "a",
                    "kind": "float",
                    "mean": 0,
                    "std": 1,
                    "null_fraction": 0.2,
                    "missingness": {"mechanism": "mnar", "strength": 1.0},
                },
                {
                    "name": "b",
                    "kind": "float",
                    "mean": 0,
                    "std": 1,
                    "null_fraction": 0.1,
                    "missingness": {"mechanism": "mar", "depends_on": "x"},
                },
                {"name": "c", "kind": "float", "mean": 0, "std": 1, "null_fraction": 0.05},
                {"name": "label", "kind": "boolean", "formula": "bernoulli(sigmoid(x))"},
            ],
            "question": {
                "title": "t",
                "prompt": "p",
                "target": "label",
                "task": "classification",
            },
        }
    )

    blanks = Anonymizer(seed=1).challenge(schema, 5_000).answer_key["missingness"]

    assert blanks["a"]["mechanism"] == "mnar"
    assert blanks["a"]["depends_on"] == "a"
    assert blanks["b"]["mechanism"] == "mar"
    assert blanks["b"]["depends_on"] == "x"
    assert blanks["c"] == {
        "mechanism": "mcar",
        "null_fraction": 0.05,
        "observed_null_share": blanks["c"]["observed_null_share"],
    }
    assert blanks["a"]["observed_null_share"]["train"] == pytest.approx(0.2, abs=0.03)
