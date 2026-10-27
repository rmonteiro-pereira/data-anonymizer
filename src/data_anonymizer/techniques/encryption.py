"""Reversible techniques: whoever holds the secret key can recover the original values.

Each technique here has an inverse that `Anonymizer.reveal` runs. Keys are derived
from the run's secret key with HKDF, one per purpose, so the same secret both
pseudonymizes and decrypts. Revealed values come back as text.
"""

import base64
import os

import narwhals as nw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, AESSIV

from data_anonymizer.table import Table
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.ff1 import FF1
from data_anonymizer.techniques.value_mapping import lookup_distinct, lookup_per_row

_DETERMINISTIC_PREFIX = "siv:"
_RANDOMIZED_PREFIX = "gcm:"
_GCM_NONCE_BYTES = 12

_ALPHABETS = {
    "digits": "0123456789",
    "alphanumeric": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
}


def encrypt(
    table: Table, column: str, context: TechniqueContext, *, deterministic: bool = True
) -> nw.Expr:
    """Replace each value with an encrypted, authenticated token.

    Deterministic mode uses AES-SIV (RFC 5297): equal values give equal tokens, so
    joins and counts survive, at the cost of revealing which rows share a value.
    Randomized mode uses AES-256-GCM with a fresh random nonce per cell, so equal
    values give different tokens and nothing can be linked without the key. Randomized
    mode needs one Python call per row, so it collects the column's values.
    """
    if deterministic:
        cipher = AESSIV(context.derived_key("aes-siv", 64))
        return lookup_distinct(table, column, lambda text: _encrypt_deterministic(cipher, text))
    gcm = AESGCM(context.derived_key("aes-gcm", 32))
    return lookup_per_row(table, column, lambda text: _encrypt_randomized(gcm, text))


def decrypt(
    table: Table, column: str, context: TechniqueContext, *, deterministic: bool = True
) -> nw.Expr:
    """Reverse `encrypt`. Tokens made in either mode are recognised by their prefix.

    `deterministic` is accepted, and ignored, so a plan's parameters fit both directions.
    """
    siv = AESSIV(context.derived_key("aes-siv", 64))
    gcm = AESGCM(context.derived_key("aes-gcm", 32))
    return lookup_distinct(table, column, lambda token: _decrypt_token(siv, gcm, token))


def _encrypt_deterministic(cipher: AESSIV, text: str) -> str:
    sealed = cipher.encrypt(text.encode("utf-8"), None)
    return _DETERMINISTIC_PREFIX + _encode(sealed)


def _encrypt_randomized(cipher: AESGCM, text: str) -> str:
    nonce = os.urandom(_GCM_NONCE_BYTES)
    sealed = cipher.encrypt(nonce, text.encode("utf-8"), None)
    return _RANDOMIZED_PREFIX + _encode(nonce + sealed)


def _decrypt_token(siv: AESSIV, gcm: AESGCM, token: str) -> str:
    try:
        if token.startswith(_DETERMINISTIC_PREFIX):
            sealed = _decode(token.removeprefix(_DETERMINISTIC_PREFIX))
            return siv.decrypt(sealed, None).decode("utf-8")
        if token.startswith(_RANDOMIZED_PREFIX):
            raw = _decode(token.removeprefix(_RANDOMIZED_PREFIX))
            nonce = raw[:_GCM_NONCE_BYTES]
            return gcm.decrypt(nonce, raw[_GCM_NONCE_BYTES:], None).decode("utf-8")
    except InvalidTag as error:
        raise ValueError("a token does not decrypt under this secret key") from error
    raise ValueError("a value is not an encryption token made by this library")


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def format_preserving_encrypt(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    alphabet: str = "digits",
    tweak: str = "",
) -> nw.Expr:
    """Encrypt with FF1 so the result keeps the original format and length.

    With `digits`, only digits change and every other character stays in place:
    `529.982.247-25` becomes another `ddd.ddd.ddd-dd`. With `alphanumeric`, letters
    and digits are encrypted together. Equal values give equal results. A value
    needs at least 6 digits (or 4 alphanumeric characters) to encrypt. The `tweak`
    is any public string, such as a table name, that makes the same value encrypt
    differently in different contexts.
    """
    cipher = _ff1(context, alphabet, tweak)
    characters = _ALPHABETS[alphabet]
    return lookup_distinct(
        table, column, lambda text: _ff1_transform(cipher.encrypt, characters, text)
    )


def format_preserving_decrypt(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    alphabet: str = "digits",
    tweak: str = "",
) -> nw.Expr:
    """Reverse `format_preserving_encrypt`; use the same alphabet and tweak."""
    cipher = _ff1(context, alphabet, tweak)
    characters = _ALPHABETS[alphabet]
    return lookup_distinct(
        table, column, lambda text: _ff1_transform(cipher.decrypt, characters, text)
    )


def _ff1(context: TechniqueContext, alphabet: str, tweak: str) -> FF1:
    if alphabet not in _ALPHABETS:
        raise ValueError(f"alphabet must be one of {', '.join(_ALPHABETS)}")
    key = context.derived_key("ff1", 32)
    return FF1(key, len(_ALPHABETS[alphabet]), tweak.encode("utf-8"))


def _ff1_transform(operation, characters: str, text: str) -> str:
    positions = [index for index, character in enumerate(text) if character in characters]
    numerals = [characters.index(text[index]) for index in positions]
    transformed = operation(numerals)
    result = list(text)
    for position, numeral in zip(positions, transformed, strict=True):
        result[position] = characters[numeral]
    return "".join(result)
