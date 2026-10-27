"""Realistic fake values (names, emails, documents) that stand in for real ones.

Pseudonymization uses `value_for` so the same real value always gets the same fake
value. Synthesis uses `many` to draw fresh values for a generated column.
"""

from collections.abc import Callable

import numpy as np
from faker import Faker

from oblitus.detection.crypto_wallets import base58check_encode, eip55_checksum_case
from oblitus.taxonomy import DataCategory

Generator = Callable[[Faker], str]

# How many times to redraw a value that is already taken before giving up on uniqueness.
_UNIQUE_ATTEMPTS = 20


def _check_digit(digits: list[int], weights: list[int], *, cnpj: bool) -> int:
    total = sum(digit * weight for digit, weight in zip(digits, weights, strict=True))
    if cnpj:
        remainder = total % 11
        return 0 if remainder < 2 else 11 - remainder
    return (total * 10) % 11 % 10


def fake_cpf(faker: Faker) -> str:
    """Return a random CPF whose check digits are valid, formatted as 000.000.000-00."""
    digits = [faker.random_int(0, 9) for _ in range(9)]
    digits.append(_check_digit(digits, list(range(10, 1, -1)), cnpj=False))
    digits.append(_check_digit(digits, list(range(11, 1, -1)), cnpj=False))
    text = "".join(map(str, digits))
    return f"{text[:3]}.{text[3:6]}.{text[6:9]}-{text[9:]}"


def fake_cnpj(faker: Faker) -> str:
    """Return a random CNPJ whose check digits are valid, formatted as 00.000.000/0000-00."""
    digits = [faker.random_int(0, 9) for _ in range(8)] + [0, 0, 0, 1]
    first_weights = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    digits.append(_check_digit(digits, first_weights, cnpj=True))
    digits.append(_check_digit(digits, [6, *first_weights], cnpj=True))
    text = "".join(map(str, digits))
    return f"{text[:2]}.{text[2:5]}.{text[5:8]}/{text[8:12]}-{text[12:]}"


def fake_crypto_wallet(faker: Faker) -> str:
    """Return a random wallet address with a valid checksum: Bitcoin or Ethereum."""
    if faker.random_int(0, 1) == 0:
        # Version byte 0 is a legacy Bitcoin address; the 20 bytes stand in for a key hash.
        key_hash = bytes(faker.random_int(0, 255) for _ in range(20))
        return base58check_encode(b"\x00" + key_hash)
    digits = "".join(f"{faker.random_int(0, 255):02x}" for _ in range(20))
    return "0x" + eip55_checksum_case(digits)


_GENERATORS: dict[DataCategory, Generator] = {
    DataCategory.EMAIL: lambda faker: faker.email(),
    DataCategory.PHONE: lambda faker: faker.phone_number(),
    DataCategory.PERSON_NAME: lambda faker: faker.name(),
    DataCategory.CPF: fake_cpf,
    DataCategory.CNPJ: fake_cnpj,
    DataCategory.SSN: lambda faker: faker.ssn(),
    DataCategory.CREDIT_CARD: lambda faker: faker.credit_card_number(),
    DataCategory.IBAN: lambda faker: faker.iban(),
    DataCategory.IP_ADDRESS: lambda faker: faker.ipv4_public(),
    DataCategory.URL: lambda faker: faker.url(),
    DataCategory.STREET_ADDRESS: lambda faker: faker.street_address(),
    DataCategory.CITY: lambda faker: faker.city(),
    DataCategory.CRYPTO_WALLET: fake_crypto_wallet,
    DataCategory.POSTAL_CODE: lambda faker: faker.postcode(),
    DataCategory.RECORD_ID: lambda faker: faker.uuid4(),
    DataCategory.GENERIC_TEXT: lambda faker: faker.sentence(nb_words=6),
}

SURROGATE_CATEGORIES = frozenset(_GENERATORS)

# Categories whose real values are often stored as bare digits. A fake value keeps
# the punctuation style of the value it replaces, so `52998224725` stays digits only.
_DIGIT_CATEGORIES = {
    DataCategory.CPF,
    DataCategory.CNPJ,
    DataCategory.SSN,
    DataCategory.CREDIT_CARD,
    DataCategory.PHONE,
}


class SurrogateFactory:
    """Draws fake values for a data category in one locale."""

    def __init__(self, locale: str = "en_US") -> None:
        self._faker = Faker(locale)

    def value_for(self, category: DataCategory, seed: int, *, like: str | None = None) -> str:
        """Return the fake value that a seed always maps to.

        When `like` is a real value made only of digits, the fake value drops its
        punctuation too, so downstream parsers keep working.
        """
        generator = _generator(category)
        self._faker.seed_instance(seed)
        fake = generator(self._faker)
        return _match_punctuation(category, fake, like)

    def many(
        self,
        category: DataCategory,
        count: int,
        rng: np.random.Generator,
        *,
        unique: bool = False,
    ) -> list[str]:
        """Return `count` fresh fake values, all different when `unique` is set."""
        generator = _generator(category)
        self._faker.seed_instance(int(rng.integers(0, 2**63 - 1)))
        if not unique:
            return [generator(self._faker) for _ in range(count)]
        return self._many_unique(generator, count)

    def _many_unique(self, generator: Generator, count: int) -> list[str]:
        values: list[str] = []
        taken: set[str] = set()
        for index in range(count):
            value = generator(self._faker)
            attempts = 1
            while value in taken and attempts < _UNIQUE_ATTEMPTS:
                value = generator(self._faker)
                attempts += 1
            if value in taken:
                # The generator ran out of distinct values; a suffix keeps rows distinct.
                value = f"{value}-{index}"
            taken.add(value)
            values.append(value)
        return values


def _generator(category: DataCategory) -> Generator:
    try:
        return _GENERATORS[category]
    except KeyError:
        supported = ", ".join(sorted(item.value for item in SURROGATE_CATEGORIES))
        raise ValueError(
            f"no fake value generator for {category.value!r}; supported: {supported}"
        ) from None


def _match_punctuation(category: DataCategory, fake: str, like: str | None) -> str:
    if like is None or category not in _DIGIT_CATEGORIES:
        return fake
    if like.isdigit():
        return "".join(character for character in fake if character.isdigit())
    return fake
