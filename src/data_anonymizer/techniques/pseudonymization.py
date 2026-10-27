"""Techniques that replace each real value with a consistent stand-in.

By default the same input always maps to the same output under one secret key, so
joins, group counts and distinct counts survive. These outputs are pseudonymous, not
anonymous: whoever holds the key can confirm a guess about a person.
"""

import base64
import hashlib
import os
from collections import Counter
from collections.abc import Callable
from typing import Any

import bcrypt
import narwhals as nw
from argon2.low_level import Type, hash_secret

from data_anonymizer.table import Table
from data_anonymizer.taxonomy import DataCategory
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.ff1 import FF1
from data_anonymizer.techniques.value_mapping import (
    canonical_value,
    distinct_entities_among,
    distinct_entity_counts,
    lookup_distinct,
    lookup_per_row,
    present_counts,
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

# The most fake values built for random surrogates; rows pick from this pool.
_MAX_SURROGATE_POOL = 100_000


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
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    time_cost: int = 2,
    memory_cost: int = 19456,
    parallelism: int = 1,
    hash_length: int = 32,
    deterministic: bool = True,
) -> nw.Expr:
    """Replace each value with an Argon2id hash in the standard `$argon2id$...` format.

    Argon2id is deliberately slow and memory-hungry, so every guess at the value
    behind a hash costs a full Argon2 run, even for someone who has the secret key.
    The defaults are the OWASP minimum (19 MiB, two passes): 10 to 20 ms per
    distinct value. With `deterministic`, all values share one salt derived from the
    secret key, so equal values get equal hashes and joins survive. Without it, each
    cell gets its own random salt and hashes cannot be joined, though anyone can
    still test a guessed value against a published hash. That mode needs one Python
    call per row, so it collects the column's values.
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

    return _hash_lookup(table, column, digest, deterministic)


def bcrypt_hash(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    rounds: int = 10,
    deterministic: bool = True,
) -> nw.Expr:
    """Replace each value with a bcrypt hash in the standard `$2b$...` format.

    Each extra round doubles the cost; 10 rounds take 40 to 60 ms per hash.
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

    return _hash_lookup(table, column, digest, deterministic)


def _hash_lookup(
    table: Table, column: str, digest: Callable[[str], str], deterministic: bool
) -> nw.Expr:
    # Equal values share a hash only when the salt is shared, so only then is one
    # call per distinct value enough.
    if deterministic:
        return lookup_distinct(table, column, digest)
    return lookup_per_row(table, column, digest)


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


def scramble_characters(table: Table, column: str, context: TechniqueContext) -> nw.Expr:
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

    return lookup_distinct(table, column, scramble)


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
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    category: str,
    unique: bool = True,
    consistent: bool = True,
) -> nw.Expr:
    """Replace each value with a realistic fake of the given category, such as an email.

    With `consistent`, the fake is derived from the value and the secret key, so the
    same person gets the same fake everywhere: a pseudonym. Without it, every cell
    gets an unrelated random fake and nothing links back to the person; the fakes
    are drawn from a pool of at most 100,000, as large as the column has present
    rows. With `unique` and `consistent`, two different real values never share a
    fake value, which keeps distinct counts exact. A fake never equals a real value of
    the column, so no real value is published by chance.
    """
    data_category = DataCategory(category)
    if consistent:
        return _consistent_surrogate(table, column, context, data_category, unique)
    return _random_surrogate(table, column, context, data_category)


def _consistent_surrogate(
    table: Table,
    column: str,
    context: TechniqueContext,
    category: DataCategory,
    unique: bool,
) -> nw.Expr:
    # Values are visited in text order so a collision resolves the same way on every engine.
    by_text = {canonical_value(value): value for value in table.distinct_values(column)}
    mapping: dict[Any, str] = {}
    # A fake that equals a real value of the column would publish that value.
    avoid = set(by_text)
    for original in sorted(by_text):
        seed = context.keyed_seed(original)
        fake = _fake_avoiding(context, category, seed, original, avoid)
        mapping[by_text[original]] = fake
        if unique:
            avoid.add(fake)
    return table.lookup(column, mapping, nw.String)


def _random_surrogate(
    table: Table, column: str, context: TechniqueContext, category: DataCategory
) -> nw.Expr:
    value = nw.col(column)
    digits_only = value.cast(nw.String).str.contains(r"^[0-9]+$").all()
    counts = table.statistics(present=value.count(), digits_only=digits_only)
    pool_size = min(counts["present"], _MAX_SURROGATE_POOL)
    if pool_size == 0:
        return value.cast(nw.String)

    # A fake keeps the punctuation style of the value it replaces, judged on the whole column.
    style_example = "0" if counts["digits_only"] else None
    generator = context.generator(column, "surrogate")
    seeds = generator.integers(0, 2**63 - 1, size=pool_size)
    real_values = {canonical_value(value) for value in table.distinct_values(column)}
    pool = {
        index: _fake_avoiding(context, category, int(seed), style_example, real_values)
        for index, seed in enumerate(seeds)
    }
    (uniform,) = table.uniforms(context.choice_key(column, "surrogate"))
    pick = table.add((uniform * pool_size).floor().cast(nw.Int64))
    fakes = table.lookup(pick, pool, nw.String)
    return nw.when(~value.is_null()).then(fakes)


def _fake_avoiding(
    context: TechniqueContext,
    category: DataCategory,
    seed: int,
    like: str | None,
    avoid: set[str],
) -> str:
    fake = context.surrogates.value_for(category, seed, like=like)
    attempt = 1
    while fake in avoid and attempt < _COLLISION_ATTEMPTS:
        fake = context.surrogates.value_for(category, seed + attempt, like=like)
        attempt += 1
    return fake


def scramble_categories(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    prefix: str = "C",
    min_count: int = 1,
    min_entities: int = 1,
    by: str | None = None,
    keyed: bool = False,
) -> nw.Expr:
    """Replace each category with an opaque code such as `C-03`, the same code for the same label.

    Codes are handed out in a random order, so they reveal neither the alphabetical
    order nor the size of the categories. Categories seen fewer than `min_count`
    times are first merged into one `<prefix>-OTHER` code, because a segment with
    one or two members identifies them. A label with many rows can still belong to only
    one or two entities, such as a segment of two stores with 72 monthly rows between them.
    With `by` (the entity column) and `min_entities`, labels held by fewer distinct entities
    are merged too. With `keyed`, the order comes from the secret key instead, so the same
    label gets the same code in every release: a pseudonym that the key holder can link.
    """
    if min_entities > 1 and by is None:
        raise ValueError("min_entities needs `by`, the column that identifies the entity")
    counts_by_label: Counter[str] = Counter()
    raw_by_label: dict[str, list[Any]] = {}
    for value, count in present_counts(table, column).items():
        label = canonical_value(value)
        counts_by_label[label] += count
        raw_by_label.setdefault(label, []).append(value)

    entities_by_label = _entities_by_label(table, column, by, min_entities)
    common = sorted(
        label
        for label, count in counts_by_label.items()
        if count >= min_count and entities_by_label.get(label, min_entities) >= min_entities
    )
    pool = _OtherPool(table, column, by, min_count, min_entities, raw_by_label, counts_by_label)
    common, other_is_blank = pool.grow_until_large_enough(common, entities_by_label)
    if keyed:
        ordered = sorted(common, key=lambda label: context.keyed_digest(label))
    else:
        permutation = context.generator(column, "scramble_categories").permutation(len(common))
        ordered = [common[index] for index in permutation]
    width = max(2, len(str(len(ordered))))
    codes = {label: f"{prefix}-{number:0{width}d}" for number, label in enumerate(ordered, 1)}
    other = None if other_is_blank else f"{prefix}-OTHER"
    mapping = {
        value: codes.get(label, other) for label, values in raw_by_label.items() for value in values
    }
    return table.lookup(column, mapping, nw.String)


class _OtherPool:
    """The labels merged into `OTHER`, which must hide their members as well as a segment does.

    A pool of leftovers can itself be too small, such as three stores under a minimum of five.
    Then the smallest labels that had a code of their own join it, one at a time, until it is
    large enough. If all labels together are still too few, the whole column is blanked.
    """

    def __init__(
        self,
        table: Table,
        column: str,
        by: str | None,
        min_count: int,
        min_entities: int,
        raw_by_label: dict[str, list[Any]],
        counts_by_label: Counter[str],
    ) -> None:
        self._table = table
        self._column = column
        self._by = by
        self._min_count = min_count
        self._min_entities = min_entities
        self._raw_by_label = raw_by_label
        self._counts_by_label = counts_by_label

    def grow_until_large_enough(
        self, common: list[str], entities_by_label: Counter[str]
    ) -> tuple[list[str], bool]:
        """Return the labels that keep a code, and whether `OTHER` has to be blanked."""
        kept = list(common)
        pooled = [label for label in self._counts_by_label if label not in set(kept)]
        if not pooled:
            return kept, False
        smallest_first = sorted(
            kept,
            key=lambda label: (
                entities_by_label.get(label, 0),
                self._counts_by_label[label],
                label,
            ),
        )
        while not self._is_large_enough(pooled):
            if not smallest_first:
                return [], True
            label = smallest_first.pop(0)
            kept.remove(label)
            pooled.append(label)
        return kept, False

    def _is_large_enough(self, pooled: list[str]) -> bool:
        rows = sum(self._counts_by_label[label] for label in pooled)
        if rows < self._min_count:
            return False
        if self._by is None or self._min_entities <= 1:
            return True
        values = [value for label in pooled for value in self._raw_by_label[label]]
        entities = distinct_entities_among(self._table, self._column, self._by, values)
        return entities >= self._min_entities


def _entities_by_label(
    table: Table, column: str, by: str | None, min_entities: int
) -> Counter[str]:
    if by is None or min_entities <= 1:
        return Counter()
    entities: Counter[str] = Counter()
    for value, count in distinct_entity_counts(table, column, by).items():
        entities[canonical_value(value)] += count
    return entities
