import numpy as np
import pandas as pd
import pytest
from argon2.low_level import Type, hash_secret

from data_anonymizer import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    PlanError,
    Protection,
    ProtectionLevel,
)
from data_anonymizer.cli import main
from data_anonymizer.techniques.noise import laplace_noise
from data_anonymizer.techniques.pseudonymization import (
    argon2_hash,
    keyed_hash,
    scramble_characters,
)

KEY = "a-secret-key-for-tests-0123456789"


def _anonymization(actions, **fields) -> AnonymizationPlan:
    return AnonymizationPlan(fidelity=Fidelity.PRIVACY, actions=actions, **fields)


@pytest.mark.parametrize(
    "action",
    [
        ColumnAction(column="email", technique="truncate_string", params={"keep": 1000}),
        ColumnAction(column="email", technique="redact_entities", params={"categories": ["cpf"]}),
    ],
)
def test_a_technique_that_leaves_identifiers_intact_is_not_anonymized(people, action):
    plan = _anonymization([action], direct_identifiers=["email"])

    protection = Anonymizer().apply(people, plan).protection

    assert protection.level is ProtectionLevel.IDENTIFIABLE


def test_identifiers_replaced_on_the_data_pass_the_identity_check(people):
    plan = _anonymization(
        [ColumnAction(column="email", technique="suppress")], direct_identifiers=["email"]
    )

    protection = Anonymizer().apply(people, plan).protection

    assert not any("email" in problem for problem in protection.problems)


def test_misspelt_quasi_identifiers_are_rejected(people):
    plan = _anonymization([], quasi_identifiers=["Age", "gender"])

    with pytest.raises(PlanError, match="'Age', which is not in the table"):
        Anonymizer().apply(people, plan)


def test_a_unique_column_released_unchanged_is_a_fingerprint(people):
    plan = _anonymization([ColumnAction(column="email", technique="suppress")])

    protection = Anonymizer().apply(people[["email", "basket_value"]], plan).protection

    assert any("basket_value is released unchanged" in problem for problem in protection.problems)
    assert protection.level is ProtectionLevel.AT_RISK


def test_a_fully_suppressed_sensitive_column_is_not_an_inference_risk(people):
    plan = _anonymization(
        [ColumnAction(column="diagnosis", technique="suppress")],
        quasi_identifiers=["gender"],
        sensitive_attributes=["diagnosis"],
    )

    result = Anonymizer().apply(people[["gender", "diagnosis"]], plan)

    assert "diagnosis" not in result.privacy_after.l_diversity
    assert not any("distinct value" in problem for problem in result.protection.problems)


def test_categorical_income_is_randomized_not_suppressed_under_privacy_anonymization():
    frame = pd.DataFrame({"income": np.random.default_rng(0).choice(["<=50K", ">50K"], 200)})

    plan = Anonymizer().propose(frame, fidelity=Fidelity.PRIVACY)

    assert plan.action_for("income").technique == "randomized_response"


def test_anonymization_is_the_default_protection(people):
    assert Anonymizer().propose(people).protection is Protection.ANONYMIZATION
    assert AnonymizationPlan(fidelity=Fidelity.STATISTICAL, actions=[]).protection is (
        Protection.ANONYMIZATION
    )


def test_short_secret_keys_are_refused(tmp_path, monkeypatch, capsys):
    with pytest.raises(ValueError, match="at least 16 bytes"):
        Anonymizer(secret_key="short")
    monkeypatch.setenv("ANONYMIZER_SECRET_KEY", "short")

    assert main(["techniques"]) == 2
    assert "at least 16 bytes" in capsys.readouterr().err


def test_argon2_salt_does_not_depend_on_the_value(run, context):
    hashes = run(
        argon2_hash, pd.Series(["111.444.777-35", "529.982.247-25"]), context, memory_cost=1024
    )
    salts = {encoded.split("$")[4] for encoded in hashes}
    guess_salt = context.keyed_digest("529.982.247-25")[:16]

    assert len(salts) == 1
    expected = hash_secret(
        b"529.982.247-25",
        context.derived_key("password-hash-salt", 16),
        time_cost=2,
        memory_cost=1024,
        parallelism=1,
        hash_len=32,
        type=Type.ID,
    )
    assert hashes.iloc[1] == expected.decode()
    assert guess_salt not in {bytes(salt, "ascii") for salt in salts}


def test_scramble_characters_keeps_distinct_ids_distinct(run, context):
    ids = pd.Series([f"{number:06d}" for number in range(20_000)])

    scrambled = run(scramble_characters, ids, context)

    assert scrambled.nunique() == 20_000
    assert scrambled.str.fullmatch(r"\d{6}").all()


def test_keyed_hash_refuses_short_digests(run, context):
    with pytest.raises(ValueError, match="between 16 and 128"):
        run(keyed_hash, pd.Series(["a"]), context, length=8)


def test_pseudonyms_ignore_the_float_storage_of_integer_ids(run, context):
    clean = pd.Series([101, 102, 103])
    with_gap = pd.Series([101.0, None, 103.0])

    first = run(keyed_hash, clean, context)
    second = run(keyed_hash, with_gap, context)

    assert first.iloc[0] == second.iloc[0]
    assert first.iloc[2] == second.iloc[2]


def test_laplace_noise_needs_bounds_chosen_in_advance(context):
    with pytest.raises(ValueError, match="lower and upper"):
        laplace_noise(pd.Series([1.0, 2.0]), context, epsilon=1)


def test_date_shift_by_entity_moves_every_row_of_a_person_alike(people):
    repeated = pd.concat([people.head(30)] * 4, ignore_index=True)
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[
            ColumnAction(
                column="birth_date",
                technique="date_shift",
                params={"max_days": 30, "by": "customer_id"},
            )
        ],
    )

    shifted = Anonymizer(seed=3).apply(repeated, plan).data
    offsets = (shifted["birth_date"] - repeated["birth_date"]).dt.days

    assert offsets.groupby(repeated["customer_id"]).nunique().max() == 1
    assert offsets.nunique() > 1


def test_proposals_shift_dates_per_entity(people):
    plan = Anonymizer().propose(
        people, fidelity=Fidelity.STATISTICAL, protection=Protection.PSEUDONYMIZATION
    )

    assert plan.entity_column == "customer_id"
    assert plan.action_for("birth_date").params["by"] == "customer_id"
