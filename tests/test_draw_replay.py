"""The stored random numbers behind a table: replaying them reproduces the rows."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_anonymizer import Anonymizer, DatasetSchema
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.synthesis.draw_replay import (
    DrawLedger,
    normal_from_uniform,
    poisson_from_uniform,
)
from data_anonymizer.synthesis.expressions import evaluate
from data_anonymizer.synthesis.generator import generate_with_replay
from data_anonymizer.synthesis.interventions import intervene

CHURN_EXAMPLE = Path(__file__).parent.parent / "examples" / "churn_challenge.json"

# SHA-256 prefixes of the churn challenge for seed 4 and 3000 rows, with floats rounded to 9
# decimals, from output identical to the version before random numbers were stored per row:
# train, test features, test answers and the answer key.
CHURN_DIGESTS = ["b9ad78b476dfea94", "9771f6ab6634e2ed", "aabcef18675ea2eb", "e66dbbc96032b493"]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def table_digest(frame: pd.DataFrame) -> str:
    # Floats are rounded first: numpy's exp and log can differ in the last bit between CPUs,
    # and the pin is about the values, not the bits.
    return digest(frame.round(9).to_csv(index=False))


def test_a_schema_without_new_fields_gives_the_same_challenge_as_before():
    schema = DatasetSchema.load(CHURN_EXAMPLE)

    challenge = Anonymizer(seed=4).challenge(schema, 3000)
    # The only difference in the key is the new, empty list of leaking columns.
    assert challenge.answer_key["features"].pop("leaks_target") == []

    produced = [
        table_digest(challenge.train),
        table_digest(challenge.test_features),
        table_digest(challenge.test_answers),
        digest(json.dumps(challenge.answer_key, sort_keys=True, default=str)),
    ]
    assert produced == CHURN_DIGESTS


def build(columns: list[dict], constraints: list[str] = (), rows: int = 4000):
    schema = DatasetSchema.model_validate({"columns": columns, "constraints": list(constraints)})
    built = generate_with_replay(
        schema, rows, rng=np.random.default_rng(3), surrogates=SurrogateFactory("en_US")
    )
    return schema, built


NOISY_COLUMNS = [
    {"name": "x", "kind": "float", "mean": 0, "std": 1},
    {"name": "t", "kind": "boolean", "formula": "bernoulli(sigmoid(x))"},
    {
        "name": "y",
        "kind": "float",
        "noise_std": 1,
        "formula": "2 * x + 3 * t + normal(0, 1) + poisson(2) + uniform(0, 1)",
    },
]


def test_setting_the_treatment_to_each_rows_own_value_reproduces_its_descendants():
    schema, built = build(NOISY_COLUMNS)
    factual = built.structural

    treated = intervene(schema, built, "t", True)
    control = intervene(schema, built, "t", False)

    on_treated = factual["t"].to_numpy(dtype=bool)
    np.testing.assert_allclose(treated["y"][on_treated], factual["y"][on_treated])
    np.testing.assert_allclose(control["y"][~on_treated], factual["y"][~on_treated])
    np.testing.assert_array_equal(treated["x"], factual["x"])


def test_the_effect_on_each_row_is_the_planted_coefficient_because_noise_is_shared():
    schema, built = build(NOISY_COLUMNS)

    effect = intervene(schema, built, "t", True)["y"] - intervene(schema, built, "t", False)["y"]

    np.testing.assert_allclose(effect, 3.0)


def test_stored_numbers_stay_aligned_with_rows_after_constraint_filtering():
    schema, built = build(NOISY_COLUMNS, constraints=["x > 0.5"])
    factual = built.structural
    ledger = built.ledger()

    replayed = evaluate(
        "bernoulli(sigmoid(x))", factual, uniforms=ledger.column("t").next_call_site
    )

    assert (factual["x"] > 0.5).all()
    np.testing.assert_array_equal(replayed, factual["t"].to_numpy())
    treated = intervene(schema, built, "t", True)
    on_treated = factual["t"].to_numpy(dtype=bool)
    np.testing.assert_allclose(treated["y"][on_treated], factual["y"][on_treated])


def test_a_condition_that_redraws_a_formula_column_replays_too():
    columns = [
        {"name": "x", "kind": "float", "mean": 0, "std": 1},
        {"name": "t", "kind": "boolean", "formula": "bernoulli(0.5)"},
        {
            "name": "y",
            "kind": "float",
            "formula": "x",
            "conditions": [{"when": "t", "mean": 50, "std": 5}],
        },
    ]
    schema, built = build(columns)
    factual = built.structural
    on_treated = factual["t"].to_numpy(dtype=bool)

    treated = intervene(schema, built, "t", True)
    control = intervene(schema, built, "t", False)

    np.testing.assert_allclose(treated["y"][on_treated], factual["y"][on_treated])
    np.testing.assert_allclose(control["y"][~on_treated], factual["y"][~on_treated])
    assert treated["y"].mean() > 45
    np.testing.assert_allclose(control["y"], factual["x"])


def test_the_published_table_has_no_helper_columns():
    _, built = build(NOISY_COLUMNS)

    assert list(built.table.columns) == ["x", "t", "y"]
    assert list(built.structural.columns) == ["x", "t", "y"]


def test_a_ledger_subset_replays_the_numbers_of_the_chosen_rows():
    _, built = build(NOISY_COLUMNS, rows=100)
    ledger = built.ledger()
    positions = np.array([5, 7, 9])

    part = ledger.subset(positions)

    full_numbers = ledger.column("y").noise()
    np.testing.assert_array_equal(part.column("y").noise(), full_numbers[positions])


def test_a_replay_cannot_invent_numbers_it_did_not_store():
    ledger = DrawLedger.replaying({("x", "base"): np.array([0.1, 0.2])})

    with pytest.raises(KeyError, match="no stored random numbers"):
        ledger.column("x").noise()


def test_random_functions_still_follow_their_distributions():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": np.zeros(100_000)})

    normal = evaluate("normal(10, 2)", frame, rng)
    counts = evaluate("poisson(4)", frame, rng)
    spread = evaluate("uniform(2, 6)", frame, rng)
    chance = evaluate("bernoulli(0.3)", frame, rng)

    assert normal.mean() == pytest.approx(10, abs=0.05)
    assert normal.std() == pytest.approx(2, abs=0.05)
    assert counts.mean() == pytest.approx(4, abs=0.05)
    assert counts.var() == pytest.approx(4, abs=0.1)
    assert spread.min() >= 2
    assert spread.max() <= 6
    assert chance.mean() == pytest.approx(0.3, abs=0.01)


def test_inverse_functions_map_uniform_numbers_in_order():
    uniform = np.array([0.0, 0.5, 1.0])

    assert normal_from_uniform(uniform)[1] == pytest.approx(0.0)
    assert np.isfinite(normal_from_uniform(uniform)).all()
    np.testing.assert_array_equal(
        poisson_from_uniform(np.array([0.1, 0.99]), np.array([0.0, 0.0])), [0, 0]
    )
    with pytest.raises(ValueError, match="not negative"):
        poisson_from_uniform(uniform, np.array([-1.0, 1.0, 1.0]))


def test_column_names_cannot_collide_with_the_hidden_helper_columns():
    with pytest.raises(ValueError, match="cannot start with"):
        DatasetSchema.model_validate(
            {"columns": [{"name": "__draw__x", "kind": "float", "mean": 0, "std": 1}]}
        )


def test_a_window_onto_a_recording_ledger_stores_numbers_for_every_row():
    rng = np.random.default_rng(3)
    ledger = DrawLedger.recording(6, rng, {})
    first_rows = ledger.subset(np.array([0, 2, 4]))
    other_rows = ledger.subset(np.array([1, 3, 5]))

    first_noise = first_rows.column("y").noise()
    other_noise = other_rows.column("y").noise()

    stored = ledger.stored(("y", "noise"))
    assert stored is not None
    np.testing.assert_array_equal(stored[[0, 2, 4]], first_noise)
    np.testing.assert_array_equal(stored[[1, 3, 5]], other_noise)
    np.testing.assert_array_equal(first_rows.column("y").noise(), first_noise)
