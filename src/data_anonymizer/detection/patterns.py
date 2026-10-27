"""Recognizers that decide whether a value looks like a known kind of personal data.

Each recognizer pairs a regular expression with an optional checksum validator. The
validator removes most false positives: eleven random digits match the CPF shape, but
only about one in a hundred also passes the CPF check digits.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from data_anonymizer.taxonomy import DataCategory


@dataclass(frozen=True, slots=True)
class ValueRecognizer:
    """A pattern, plus an optional validator, for one category of personal data."""

    category: DataCategory
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None

    def matches(self, value: str) -> bool:
        """Return True when the whole value is an instance of this category."""
        if self.pattern.fullmatch(value) is None:
            return False
        return self._is_valid(value)

    def find_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the start and end of every valid occurrence inside free text."""
        spans = []
        for match in self.pattern.finditer(text):
            if self._is_valid(match.group(0)):
                spans.append(match.span())
        return spans

    def _is_valid(self, value: str) -> bool:
        if self.validator is None:
            return True
        return self.validator(value)


def _digits(value: str) -> list[int]:
    return [int(character) for character in value if character.isdigit()]


def is_valid_cpf(value: str) -> bool:
    """Check the two CPF verification digits (Brazilian individual taxpayer number)."""
    digits = _digits(value)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for position in (9, 10):
        weights = range(position + 1, 1, -1)
        total = sum(
            digit * weight for digit, weight in zip(digits[:position], weights, strict=True)
        )
        expected = (total * 10) % 11 % 10
        if digits[position] != expected:
            return False
    return True


def is_valid_cnpj(value: str) -> bool:
    """Check the two CNPJ verification digits (Brazilian company taxpayer number)."""
    digits = _digits(value)
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    first_weights = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    second_weights = [6, *first_weights]
    for position, weights in ((12, first_weights), (13, second_weights)):
        total = sum(
            digit * weight for digit, weight in zip(digits[:position], weights, strict=True)
        )
        remainder = total % 11
        expected = 0 if remainder < 2 else 11 - remainder
        if digits[position] != expected:
            return False
    return True


def passes_luhn(value: str) -> bool:
    """Check the Luhn checksum that every payment card number carries."""
    digits = _digits(value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def is_valid_iban(value: str) -> bool:
    """Check the ISO 13616 mod-97 checksum of an IBAN."""
    compact = value.replace(" ", "").upper()
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(character, 36)) for character in rearranged)
    return int(numeric) % 97 == 1


def is_plausible_ssn(value: str) -> bool:
    """Reject US Social Security numbers in ranges that are never issued."""
    area, group, serial = value.split("-")
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def has_phone_length(value: str) -> bool:
    """Accept numbers with an area code: 10 to 13 digits including the country code."""
    return 10 <= len(_digits(value)) <= 13


# Order matters when several recognizers match the same value: the earlier, more
# specific one wins. CPF comes before phone because eleven bare digits fit both.
RECOGNIZERS: tuple[ValueRecognizer, ...] = (
    ValueRecognizer(
        DataCategory.EMAIL,
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ),
    ValueRecognizer(
        DataCategory.CNPJ,
        re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
        is_valid_cnpj,
    ),
    ValueRecognizer(
        DataCategory.CPF,
        re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
        is_valid_cpf,
    ),
    ValueRecognizer(
        DataCategory.CREDIT_CARD,
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        passes_luhn,
    ),
    ValueRecognizer(
        DataCategory.SSN,
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        is_plausible_ssn,
    ),
    ValueRecognizer(
        DataCategory.IBAN,
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        is_valid_iban,
    ),
    ValueRecognizer(
        DataCategory.IP_ADDRESS,
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ),
    ValueRecognizer(
        DataCategory.URL,
        re.compile(r"https?://[^\s/$.?#][^\s]*"),
    ),
    ValueRecognizer(
        DataCategory.POSTAL_CODE,
        re.compile(r"\b\d{5}-\d{3}\b"),
    ),
    ValueRecognizer(
        DataCategory.PHONE,
        re.compile(r"(?:\+\d{1,3}[\s.-]?)?\(?\b\d{2,3}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}\b"),
        has_phone_length,
    ),
)


def recognizers_for(categories: Iterable[DataCategory] | None) -> tuple[ValueRecognizer, ...]:
    """Return the recognizers for the given categories, or all of them for None."""
    if categories is None:
        return RECOGNIZERS
    wanted = set(categories)
    return tuple(recognizer for recognizer in RECOGNIZERS if recognizer.category in wanted)
