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


def _repeated_key_frame(distinct: int, rows_each: int, as_float: bool) -> pd.DataFrame:
    keys = np.repeat(np.arange(distinct), rows_each)
    if as_float:
        return pd.DataFrame({"account": keys.astype(float) + 4e15})
    return pd.DataFrame({"account": [f"K{key:06d}" for key in keys]})


def test_a_key_below_a_thousand_accounts_is_still_an_account_key():
    problems = _problems(_repeated_key_frame(999, 5, as_float=False), _plan([]))

    assert any(problem.startswith("account was not treated") for problem in problems)


def test_a_whole_number_float_key_is_an_account_key():
    problems = _problems(_repeated_key_frame(1001, 5, as_float=True), _plan([]))

    assert any(problem.startswith("account was not treated") for problem in problems)


def test_a_fractional_float_repeated_by_chance_is_not_an_account_key():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"price": rng.choice(np.arange(100) + 0.5, 3000)})

    assert not any("price was not treated" in p for p in _problems(frame, _plan([])))


def test_a_column_with_many_rows_per_value_is_an_account_key():
    frame = _repeated_key_frame(60, 100, as_float=False)

    problems = _problems(frame, _plan([]))

    assert any("account was not treated" in p and "knows their own value" in p for p in problems)


def test_a_merchant_card_with_twenty_rows_per_value_is_an_account_key():
    problems = _problems(_repeated_key_frame(2000, 20, as_float=False), _plan([]))

    assert any(problem.startswith("account was not treated") for problem in problems)


def test_few_accounts_and_one_fractional_cell_are_still_account_keys():
    few = _repeated_key_frame(49, 10, as_float=False)
    mostly_whole = _repeated_key_frame(60, 10, as_float=True)
    mostly_whole.loc[0, "account"] += 0.25

    assert any("account was not treated" in p for p in _problems(few, _plan([])))
    assert any("account was not treated" in p for p in _problems(mostly_whole, _plan([])))


def test_fewer_than_twenty_distinct_values_are_not_an_account_key():
    frame = _repeated_key_frame(19, 50, as_float=False)

    assert not any("account was not treated" in p for p in _problems(frame, _plan([])))


def test_two_columns_that_are_a_key_only_together_are_flagged_as_a_pair():
    first = np.repeat(np.arange(60), 60 * 2)
    second = np.tile(np.repeat(np.arange(60), 2), 60)
    frame = pd.DataFrame({"left": first, "right": second})

    problems = _problems(frame, _plan([]))

    assert any(p.startswith("left and right were not treated") for p in problems)


def test_a_listed_column_is_not_flagged_as_a_key():
    frame = _repeated_key_frame(60, 10, as_float=False)
    plan = _plan([], quasi_identifiers=["account"])

    assert not any("account was not treated" in p for p in _problems(frame, plan))


def test_the_proposal_replaces_an_untreated_key_with_fresh_codes():
    engine = Anonymizer(seed=1)
    frame = _repeated_key_frame(49, 11, as_float=False)

    plan = engine.propose(frame)
    result = engine.apply(frame, plan)

    assert plan.action_for("account").technique == "scramble_categories"
    assert not result.data["account"].isin(frame["account"]).any()
    assert result.data["account"].nunique() == 49
    assert result.protection.level is ProtectionLevel.ANONYMIZED


def _composite_key_frame() -> pd.DataFrame:
    first = np.repeat(np.arange(60), 60 * 2)
    second = np.tile(np.repeat(np.arange(60), 2), 60)
    return pd.DataFrame({"left": first, "right": second})


def test_the_proposal_replaces_both_text_columns_of_a_composite_key():
    frame = _composite_key_frame().astype(str)

    plan = Anonymizer(seed=1).propose(frame)

    assert {action.column for action in plan.actions} == {"left", "right"}


def test_a_whole_number_composite_key_is_kept_but_reported():
    engine = Anonymizer(seed=1)
    frame = _composite_key_frame()

    plan = engine.propose(frame)
    result = engine.apply(frame, plan)

    # Whole numbers may be measurements whose distribution must survive, so they stay.
    assert plan.action_for("left") is None
    assert result.protection.level is ProtectionLevel.AT_RISK
    assert any("left and right" in problem for problem in result.protection.problems)


def test_the_proposal_keeps_measurements_and_pseudonymization_plans_unchanged():
    engine = Anonymizer(seed=1)
    rng = np.random.default_rng(0)
    decimals = pd.DataFrame({"price": rng.choice(np.arange(100) + 0.5, 3000)})
    keys = _repeated_key_frame(60, 10, as_float=False)

    assert engine.propose(decimals).actions == []
    assert engine.propose(keys, protection=Protection.PSEUDONYMIZATION).actions == []
