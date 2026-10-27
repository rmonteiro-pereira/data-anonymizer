import bcrypt
import pandas as pd
import pytest
from argon2 import PasswordHasher

from data_anonymizer import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    PlanError,
    Protection,
)
from data_anonymizer.cli import main
from data_anonymizer.techniques import TechniqueContext
from data_anonymizer.techniques.encryption import (
    decrypt,
    encrypt,
    format_preserving_decrypt,
    format_preserving_encrypt,
)
from data_anonymizer.techniques.ff1 import FF1
from data_anonymizer.techniques.pseudonymization import argon2_hash, bcrypt_hash, keyed_hash

_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_NIST_KEY_128 = "2B7E151628AED2A6ABF7158809CF4F3C"
_NIST_KEY_256 = "2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94"


@pytest.mark.parametrize(
    ("key", "tweak", "radix", "plaintext", "ciphertext"),
    [
        (_NIST_KEY_128, "", 10, "0123456789", "2433477484"),
        (_NIST_KEY_128, "39383736353433323130", 10, "0123456789", "6124200773"),
        (
            _NIST_KEY_128,
            "3737373770717273373737",
            36,
            "0123456789abcdefghi",
            "a9tv40mll9kdu509eum",
        ),
        (_NIST_KEY_256, "", 10, "0123456789", "6657667009"),
    ],
)
def test_ff1_matches_the_nist_sample_vectors(key, tweak, radix, plaintext, ciphertext):
    cipher = FF1(bytes.fromhex(key), radix, bytes.fromhex(tweak))
    numerals = [_BASE36.index(character) for character in plaintext]

    encrypted = cipher.encrypt(numerals)

    assert "".join(_BASE36[numeral] for numeral in encrypted) == ciphertext
    assert cipher.decrypt(encrypted) == numerals


@pytest.mark.parametrize("deterministic", [True, False])
def test_encrypt_round_trips_and_keeps_gaps(context, deterministic):
    values = pd.Series(["ana@example.com", None, "ana@example.com"])

    tokens = encrypt(values, context, deterministic=deterministic)

    assert pd.isna(tokens.iloc[1])
    assert (tokens.iloc[0] == tokens.iloc[2]) is deterministic
    assert decrypt(tokens, context).iloc[[0, 2]].tolist() == ["ana@example.com"] * 2


def test_tokens_do_not_decrypt_under_another_key(context):
    tokens = encrypt(pd.Series(["secret"]), context)
    other = TechniqueContext(context.rng, b"another key", context.surrogates)

    with pytest.raises(ValueError, match="does not decrypt"):
        decrypt(tokens, other)


@pytest.mark.parametrize(
    ("alphabet", "value", "shape"),
    [
        ("digits", "529.982.247-25", r"\d{3}\.\d{3}\.\d{3}-\d{2}"),
        ("alphanumeric", "AB-12cd", r"[0-9A-Za-z]{2}-[0-9A-Za-z]{4}"),
    ],
)
def test_format_preserving_encryption_keeps_the_shape(context, alphabet, value, shape):
    values = pd.Series([value])

    encrypted = format_preserving_encrypt(values, context, alphabet=alphabet, tweak="t1")

    assert encrypted.iloc[0] != value
    assert encrypted.str.fullmatch(shape).all()
    restored = format_preserving_decrypt(encrypted, context, alphabet=alphabet, tweak="t1")
    assert restored.iloc[0] == value


def test_format_preserving_encryption_rejects_too_short_values(context):
    with pytest.raises(ValueError, match="at least 6"):
        format_preserving_encrypt(pd.Series(["1234"]), context)


def test_argon2_hash_verifies_and_is_deterministic_by_default(context):
    values = pd.Series(["529.982.247-25", "529.982.247-25"])

    hashes = argon2_hash(values, context, memory_cost=1024, time_cost=1)
    random_salted = argon2_hash(values.iloc[:1], context, memory_cost=1024, deterministic=False)

    assert hashes.iloc[0] == hashes.iloc[1]
    assert hashes.iloc[0].startswith("$argon2id$")
    assert PasswordHasher().verify(hashes.iloc[0], "529.982.247-25")
    assert random_salted.iloc[0] != hashes.iloc[0]


def test_bcrypt_hash_verifies_and_handles_long_values(context):
    long_value = "x" * 100
    values = pd.Series(["short", "short", long_value])

    hashes = bcrypt_hash(values, context, rounds=4)

    assert hashes.iloc[0] == hashes.iloc[1]
    assert bcrypt.checkpw(b"short", hashes.iloc[0].encode())
    assert hashes.iloc[2].startswith("$2b$04$")


def test_keyed_hash_offers_several_algorithms(context):
    values = pd.Series(["alice"])

    digests = {
        algorithm: keyed_hash(values, context, algorithm=algorithm, length=32).iloc[0]
        for algorithm in ("sha256", "sha3_256", "blake2b")
    }

    assert len(set(digests.values())) == 3
    with pytest.raises(ValueError, match="algorithm"):
        keyed_hash(values, context, algorithm="md5")


def _reversible_plan() -> AnonymizationPlan:
    return AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.PSEUDONYMIZATION,
        actions=[
            ColumnAction(column="email", technique="encrypt"),
            ColumnAction(column="cpf", technique="format_preserving_encrypt"),
            ColumnAction(column="full_name", technique="redact"),
        ],
    )


def test_reveal_recovers_reversible_columns_and_leaves_one_way_ones(people):
    plan = _reversible_plan()
    anonymized = Anonymizer(secret_key="vault-for-tests-0123456789").apply(people, plan).data

    revealed = Anonymizer(secret_key="vault-for-tests-0123456789").reveal(anonymized, plan)

    assert not anonymized["email"].isin(people["email"]).any()
    assert revealed["email"].equals(people["email"].astype(object))
    assert revealed["cpf"].equals(people["cpf"].astype(object))
    assert (revealed["full_name"] == "[REDACTED]").all()


def test_reveal_with_the_wrong_key_fails_clearly(people):
    plan = _reversible_plan()
    anonymized = Anonymizer(secret_key="vault-for-tests-0123456789").apply(people, plan).data

    with pytest.raises(PlanError, match="cannot reveal"):
        Anonymizer(secret_key="guess-for-tests-0123456789").reveal(anonymized, plan)


def test_cli_reveal_needs_the_secret_key(people, tmp_path, monkeypatch, capsys):
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    _reversible_plan().save(plan_path)
    anonymized = tmp_path / "anon.csv"
    revealed = tmp_path / "revealed.csv"

    monkeypatch.setenv("ANONYMIZER_SECRET_KEY", "vault-for-tests-0123456789")
    assert main(["apply", str(source), "--plan", str(plan_path), "-o", str(anonymized)]) == 0
    assert main(["reveal", str(anonymized), "--plan", str(plan_path), "-o", str(revealed)]) == 0
    assert pd.read_csv(revealed)["email"].equals(people["email"])

    monkeypatch.delenv("ANONYMIZER_SECRET_KEY")
    assert main(["reveal", str(anonymized), "--plan", str(plan_path), "-o", str(revealed)]) == 2
    assert "needs ANONYMIZER_SECRET_KEY" in capsys.readouterr().err
