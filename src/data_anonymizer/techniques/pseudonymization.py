"""Techniques that replace each real value with a consistent stand-in.

By default the same input always maps to the same output under one secret key, so
joins, group counts and distinct counts survive. These outputs are pseudonymous, not
anonymous: whoever holds the key can confirm a guess about a person.
"""

import base64
import hashlib
import os

import bcrypt
import pandas as pd
from argon2.low_level import Type, hash_secret

from data_anonymizer.taxonomy import DataCategory
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.value_mapping import map_distinct

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
    values: pd.Series,
    context: TechniqueContext,
    *,
    length: int = 16,
    prefix: str = "",
    algorithm: str = "sha256",
) -> pd.Series:
    """Replace each value with a truncated HMAC hex digest, optionally prefixed.

    `algorithm` is one of sha256, sha512, sha3_256, sha3_512, blake2b or blake2s.
    """
    if algorithm not in KEYED_HASH_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(KEYED_HASH_ALGORITHMS)}")
    if not 8 <= length <= 128:
        raise ValueError("length must be between 8 and 128 hex characters")

    def digest(text: str) -> str:
        return prefix + context.keyed_digest(text, algorithm).hex()[:length]

    return map_distinct(values, digest)


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

    Argon2id is deliberately slow and memory-hungry, so guessing the value behind a
    hash stays expensive even for someone who has the secret key. The defaults are
    the OWASP minimum (19 MiB, two passes): 10 to 20 ms per distinct value.
    With `deterministic`, the salt is derived from the value and the secret key, so
    equal values get equal hashes and joins survive. Without it, each distinct value
    gets a random salt and the result cannot be linked to anything.
    """

    def digest(text: str) -> str:
        encoded = hash_secret(
            text.encode("utf-8"),
            _salt(context, text, deterministic),
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
            salt = _bcrypt_salt(_salt(context, text, deterministic=True), rounds)
        else:
            salt = bcrypt.gensalt(rounds)
        return bcrypt.hashpw(secret, salt).decode("ascii")

    return map_distinct(values, digest)


def _salt(context: TechniqueContext, text: str, deterministic: bool) -> bytes:
    if deterministic:
        return context.keyed_digest(text)[:16]
    return os.urandom(16)


def _bcrypt_salt(raw_salt: bytes, rounds: int) -> bytes:
    # bcrypt writes its 16-byte salt as 22 characters of its own base64 alphabet.
    standard = base64.b64encode(raw_salt).decode("ascii")[:22]
    translated = standard.translate(str.maketrans(_STANDARD_BASE64, _BCRYPT_BASE64))
    return f"$2b${rounds:02d}${translated}".encode("ascii")


def scramble_characters(values: pd.Series, context: TechniqueContext) -> pd.Series:
    """Replace each digit with a digit and each letter with a letter, keeping the format.

    `AB-1234` might become `QZ-8071`. The replacement depends on the whole value, so
    two values that share a prefix do not share the scrambled prefix.
    """

    def scramble(text: str) -> str:
        stream = context.keyed_digest(text)
        characters = []
        for index, character in enumerate(text):
            byte = stream[index % len(stream)] + index
            characters.append(_scramble_one(character, byte))
        return "".join(characters)

    return map_distinct(values, scramble)


def _scramble_one(character: str, byte: int) -> str:
    if character.isdigit():
        return _DIGITS[byte % 10]
    if character.isascii() and character.isalpha():
        replacement = _LOWERCASE[byte % 26]
        return replacement.upper() if character.isupper() else replacement
    return character


def surrogate(
    values: pd.Series,
    context: TechniqueContext,
    *,
    category: str,
    unique: bool = True,
) -> pd.Series:
    """Replace each value with a realistic fake of the given category, such as an email.

    With `unique`, two different real values never share a fake value, which keeps
    distinct counts exact.
    """
    data_category = DataCategory(category)
    present = values.dropna().astype(str)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for original in present.unique():
        fake = _surrogate_for(context, data_category, original, used if unique else None)
        mapping[original] = fake
        used.add(fake)

    result = values.astype(object).copy()
    result[values.notna()] = present.map(mapping)
    return result


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
