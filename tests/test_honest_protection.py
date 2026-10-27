import numpy as np
import pandas as pd

from data_anonymizer import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    Protection,
    ProtectionLevel,
)
from data_anonymizer.detection.patterns import TEXT_RECOGNIZERS
from data_anonymizer.detection.text_entities import find_entities
from data_anonymizer.techniques.masking import _redact_text


def _plan(actions, **fields) -> AnonymizationPlan:
    return AnonymizationPlan(fidelity=Fidelity.STATISTICAL, actions=actions, **fields)


def _problems(frame, plan) -> list[str]:
    return Anonymizer(seed=1).apply(frame, plan).protection.problems


def test_an_untreated_account_key_makes_the_release_at_risk():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "card": [f"CARD-{number:05d}" for number in rng.integers(0, 3000, 30000)],
            "mcc": rng.choice(["a", "b", "c"], 30000),
        }
    )

    protection = Anonymizer(seed=1).apply(frame, _plan([])).protection

    assert protection.level is ProtectionLevel.AT_RISK
    assert any(problem.startswith("card was not treated") for problem in protection.problems)
    assert not any("mcc" in problem for problem in protection.problems)


def test_a_code_with_few_distinct_values_is_not_an_account_key():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"product": rng.integers(0, 300, 30000).astype(str)})

    assert not any("account" in problem for problem in _problems(frame, _plan([])))


def test_perturbed_columns_that_link_jointly_are_reported():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["x", "y", "z"])
    actions = [ColumnAction(column=name, technique="gaussian_noise") for name in frame.columns]

    problems = _problems(frame, _plan(actions))

    assert any("together" in problem for problem in problems)


def test_a_cleaned_text_column_is_never_called_anonymized():
    frame = pd.DataFrame({"note": ["Ligar para Maria no email a@b.com"] * 30})
    plan = _plan(
        [ColumnAction(column="note", technique="redact_entities")], direct_identifiers=["note"]
    )

    protection = Anonymizer(seed=1).apply(frame, plan).protection

    assert protection.level is ProtectionLevel.AT_RISK
    assert any("names and other free-text details may remain" in p for p in protection.problems)


def test_a_partly_treated_text_column_that_stays_distinct_is_a_fingerprint():
    frame = pd.DataFrame({"ref": [f"user-{number}-x" for number in range(100)]})
    plan = _plan([ColumnAction(column="ref", technique="partial_mask", params={"keep_last": 4})])

    problems = _problems(frame, plan)

    assert any("ref is still" in problem and "partial_mask" in problem for problem in problems)


def test_randomized_response_without_an_entity_draw_is_reported():
    frame = pd.DataFrame({"patient": ["p1", "p2"] * 20, "dx": ["a", "b", "c", "d"] * 10})
    unpaired = _plan(
        [ColumnAction(column="dx", technique="randomized_response")], entity_column="patient"
    )
    paired = _plan(
        [ColumnAction(column="dx", technique="randomized_response", params={"by": "patient"})],
        entity_column="patient",
    )

    assert any("draws again" in problem for problem in _problems(frame, unpaired))
    assert not any("draws again" in problem for problem in _problems(frame, paired))


def test_randomized_response_by_entity_gives_every_row_of_a_person_one_value():
    frame = pd.DataFrame({"patient": [f"p{n % 50}" for n in range(500)], "dx": ["a", "b"] * 250})
    frame["dx"] = frame["patient"].map(lambda name: "ab"[int(name[1:]) % 2])
    plan = _plan(
        [
            ColumnAction(
                column="dx",
                technique="randomized_response",
                params={"keep_probability": 0.3, "by": "patient"},
            )
        ],
        protection=Protection.PSEUDONYMIZATION,
    )

    released = Anonymizer(seed=1).apply(frame, plan).data

    assert released.groupby("patient")["dx"].nunique().max() == 1


def test_obfuscated_identifiers_are_removed_from_the_original_text():
    forms = [
        "529 982 247 25",
        "52.9982.2472.5",
        "５２９９８２２４７２５",
        "529​982​24725",
        "maria.silva [at] example.test",
        "mаria.silva@exаmple.test",
        "+55 onze 99876 cinco 4321",
    ]
    for form in forms:
        redacted = _redact_text(f"Contato: {form}.", TEXT_RECOGNIZERS)

        assert redacted in {"Contato: <CPF>.", "Contato: <EMAIL>.", "Contato: <PHONE>."}, form


def test_ordinary_prose_and_numbers_are_left_alone():
    text = "Meeting at 5.30pm on 2024-05-12, order 123456, we met at work.Then left"

    assert find_entities(text, TEXT_RECOGNIZERS) == []


def test_a_surviving_obfuscated_identifier_makes_the_release_identifiable():
    frame = pd.DataFrame({"note": ["CPF 529 982 247 25 anotado"] * 30})
    plan = _plan(
        [
            ColumnAction(
                column="note", technique="redact_entities", params={"categories": ["email"]}
            )
        ],
        direct_identifiers=["note"],
    )

    assert Anonymizer(seed=1).apply(frame, plan).protection.level is ProtectionLevel.IDENTIFIABLE
