"""Techniques that replace each real value with a consistent stand-in.

By default the same input always maps to the same output under one secret key, so
joins, group counts and distinct counts survive. These outputs are pseudonymous, not
anonymous: whoever holds the key can confirm a guess about a person.
"""

import base64
import hashlib
import os
from collections.abc import Callable

import bcrypt
import narwhals as nw
import pandas as pd
from argon2.low_level import Type, hash_secret

from data_anonymizer.table import Table
from data_anonymizer.taxonomy import DataCategory
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.ff1 import FF1
from data_anonymizer.techniques.value_mapping import (
    canonical_text,
    lookup_distinct,
    map_distinct,
    map_present,
)

_DIGITS = "0123456789"
_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"

KEYED_HASH_ALGORITHMS = ("sha256", "sha512", "sha3_256", "sha3_512", "blake2b", "blake2s")

# bcrypt only reads the first 72 bytes of its input.
_BCRYPT_MAX_BYTES = 72
_STANDARD_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_BCRYPT_BASE64 = "./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# How many reseeds a surrogate may try before accepting a fake value already in use.
_COLLISION_ATTEMPTS = 20


def keyed_hash(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    length: int = 16,
    prefix: str = "",
    algorithm: str = "sha256",
) -> nw.Expr:
    """Replace each value with a truncated HMAC hex digest, optionally prefixed.

    `algorithm` is one of sha256, sha512, sha3_256, sha3_512, blake2b or blake2s. The
    minimum length is 16 hex characters (64 bits): shorter digests start to merge
    different people once a column has millions of distinct values.
    """
    if algorithm not in KEYED_HASH_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(KEYED_HASH_ALGORITHMS)}")
    if not 16 <= length <= 128:
        raise ValueError("length must be between 16 and 128 hex characters")

    def digest(text: str) -> str:
        return prefix + context.keyed_digest(text, algorithm).hex()[:length]

    return lookup_distinct(table, column, digest)


def argon2_hash(
    values: pd.Series,
    context: TechniqueContext,
    *,
    time_cost: int = 2,
    memory_cost: int = 19456,
    parallelism: int = 1,
    hash_length: int = 32,
    deterministic: bool = True,
) -> pd.Series:
    """Replace each value with an Argon2id hash in the standard `$argon2id$...` format.

    Argon2id is deliberately slow and memory-hungry, so every guess at the value
    behind a hash costs a full Argon2 run, even for someone who has the secret key.
    The defaults are the OWASP minimum (19 MiB, two passes): 10 to 20 ms per
    distinct value. With `deterministic`, all values share one salt derived from the
    secret key, so equal values get equal hashes and joins survive. Without it, each
    distinct value gets a random salt and hashes cannot be joined, though anyone can
    still test a guessed value against a published hash.
    """

    def digest(text: str) -> str:
        encoded = hash_secret(
            text.encode("utf-8"),
            _salt(context, deterministic),
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_length,
            type=Type.ID,
        )
        return encoded.decode("ascii")

    return map_distinct(values, digest)


def bcrypt_hash(
    values: pd.Series,
    context: TechniqueContext,
    *,
    rounds: int = 10,
    deterministic: bool = True,
) -> pd.Series:
    """Replace each value with a bcrypt hash in the standard `$2b$...` format.

    Each extra round doubles the cost; 10 rounds take 40 to 60 ms per distinct value.
    `deterministic` works as in `argon2_hash`. bcrypt reads only 72 bytes, so longer
    values are first reduced to their SHA-256 hex digest.
    """
    if not 4 <= rounds <= 31:
        raise ValueError("rounds must be between 4 and 31")

    def digest(text: str) -> str:
        secret = text.encode("utf-8")
        if len(secret) > _BCRYPT_MAX_BYTES:
            secret = hashlib.sha256(secret).hexdigest().encode("ascii")
        if deterministic:
            salt = _bcrypt_salt(_salt(context, deterministic=True), rounds)
        else:
            salt = bcrypt.gensalt(rounds)
        return bcrypt.hashpw(secret, salt).decode("ascii")

    return map_distinct(values, digest)


def _salt(context: TechniqueContext, deterministic: bool) -> bytes:
    # The shared salt does not depend on the value: a salt derived from the value
    # would let the key holder test guesses with one cheap HMAC instead of Argon2.
    if deterministic:
        return context.derived_key("password-hash-salt", 16)
    return os.urandom(16)


def _bcrypt_salt(raw_salt: bytes, rounds: int) -> bytes:
    # bcrypt writes its 16-byte salt as 22 characters of its own base64 alphabet.
    standard = base64.b64encode(raw_salt).decode("ascii")[:22]
    translated = standard.translate(str.maketrans(_STANDARD_BASE64, _BCRYPT_BASE64))
    return f"$2b${rounds:02d}${translated}".encode("ascii")


def scramble_characters(values: pd.Series, context: TechniqueContext) -> pd.Series:
    """Replace each digit with a digit and each letter with a letter, keeping the format.

    `AB-1234567` might become `QZ-8071533`. Digits and letters are each encrypted
    with FF1 under a key derived from the secret key, which is a one-to-one mapping,
    so distinct values stay distinct and joins survive. FF1 needs at least 6 digits
    or 5 letters; shorter runs fall back to a keyed hash per character, where two
    values can occasionally collide.
    """
    digit_cipher = FF1(context.derived_key("scramble-digits", 32), 10)
    letter_cipher = FF1(context.derived_key("scramble-letters", 32), 26)

    def scramble(text: str) -> str:
        characters = list(text)
        _encrypt_class(characters, str.isdigit, _DIGITS, digit_cipher, context, text)
        _encrypt_class(characters, _is_ascii_letter, _LOWERCASE, letter_cipher, context, text)
        return "".join(characters)

    return map_distinct(values, scramble)


def _is_ascii_letter(character: str) -> bool:
    return character.isascii() and character.isalpha()


def _encrypt_class(
    characters: list[str],
    belongs: Callable[[str], bool],
    alphabet: str,
    cipher: FF1,
    context: TechniqueContext,
    text: str,
) -> None:
    positions = [index for index, character in enumerate(characters) if belongs(character)]
    if not positions:
        return
    numerals = [alphabet.index(characters[index].lower()) for index in positions]
    if len(numerals) >= cipher.min_length:
        encrypted = cipher.encrypt(numerals)
    else:
        stream = context.keyed_digest(text + alphabet)
        encrypted = [(stream[index] + index) % len(alphabet) for index in range(len(numerals))]
    for position, numeral in zip(positions, encrypted, strict=True):
        replacement = alphabet[numeral]
        characters[position] = (
            replacement.upper() if characters[position].isupper() else replacement
        )


def surrogate(
    values: pd.Series,
    context: TechniqueContext,
    *,
    category: str,
    unique: bool = True,
    consistent: bool = True,
) -> pd.Series:
    """Replace each value with a realistic fake of the given category, such as an email.

    With `consistent`, the fake is derived from the value and the secret key, so the
    same person gets the same fake everywhere: a pseudonym. Without it, every cell
    gets an unrelated random fake and nothing links back to the person. With
    `unique` and `consistent`, two different real values never share a fake value,
    which keeps distinct counts exact.
    """
    data_category = DataCategory(category)
    if not consistent:
        return map_present(
            values, lambda original: _random_surrogate(context, data_category, original)
        )

    present = canonical_text(values)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for original in present.unique():
        fake = _surrogate_for(context, data_category, original, used if unique else None)
        mapping[original] = fake
        used.add(fake)

    result = values.astype(object).copy()
    result[values.notna()] = present.map(mapping)
    return result


def _random_surrogate(context: TechniqueContext, category: DataCategory, original: str) -> str:
    seed = int(context.rng.integers(0, 2**63 - 1))
    return context.surrogates.value_for(category, seed, like=original)


def _surrogate_for(
    context: TechniqueContext,
    category: DataCategory,
    original: str,
    used: set[str] | None,
) -> str:
    seed = context.keyed_seed(original)
    fake = context.surrogates.value_for(category, seed, like=original)
    if used is None:
        return fake
    attempt = 1
    while fake in used and attempt < _COLLISION_ATTEMPTS:
        fake = context.surrogates.value_for(category, seed + attempt, like=original)
        attempt += 1
    return fake


def scramble_categories(
    values: pd.Series,
    context: TechniqueContext,
    *,
    prefix: str = "C",
    min_count: int = 1,
    keyed: bool = False,
) -> pd.Series:
    """Replace each category with an opaque code such as `C-03`, the same code for the same label.

    Codes are handed out in a random order, so they reveal neither the alphabetical
    order nor the size of the categories. Categories seen fewer than `min_count`
    times are first merged into one `<prefix>-OTHER` code, because a segment with
    one or two members identifies them. With `keyed`, the order comes from the
    secret key instead, so the same label gets the same code in every release: a
    pseudonym that the key holder can link.
    """
    present = canonical_text(values)
    counts = present.value_counts()
    common = [label for label in counts.index if counts[label] >= min_count]
    if keyed:
        ordered = sorted(common, key=lambda label: context.keyed_digest(label))
    else:
        ordered = [common[index] for index in context.rng.permutation(len(common))]
    width = max(2, len(str(len(ordered))))
    codes = {label: f"{prefix}-{number:0{width}d}" for number, label in enumerate(ordered, 1)}
    other = f"{prefix}-OTHER"
    result = values.astype(object).copy()
    result[values.notna()] = present.map(lambda label: codes.get(label, other))
    return result
