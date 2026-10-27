"""Recognizers that decide whether a value looks like a known kind of personal data.

Each recognizer pairs a regular expression with an optional checksum from
`checksums.py`. Two flags limit where a recognizer may fire:
- `in_text`: whether it is searched for inside free text. Bare-digit formats such
  as NIS or CNH are left out, because prose is full of numbers that fit them.
- `needs_name_hint`: whether a matching column also needs a column name that
  points to the category, for formats with no checksum that ordinary codes share,
  such as UUIDs, which may be PIX keys or plain record keys.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from data_anonymizer.detection import checksums
from data_anonymizer.taxonomy import DataCategory

# Re-exported so callers keep importing the validators from here.
is_valid_cpf = checksums.is_valid_cpf
is_valid_cnpj = checksums.is_valid_cnpj
is_valid_iban = checksums.is_valid_iban
passes_luhn = checksums.passes_luhn


@dataclass(frozen=True, slots=True)
class ValueRecognizer:
    """A pattern, plus an optional validator, for one kind of personal data."""

    category: DataCategory
    label: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None
    in_text: bool = True
    needs_name_hint: bool = False
    digit_length: int | None = None

    def matches(self, value: str) -> bool:
        """Return True when the whole value is an instance of this kind."""
        if self.pattern.fullmatch(value) is None:
            return False
        return self._is_valid(value)

    def matches_padded(self, value: str) -> bool:
        """Like `matches`, but restores leading zeros lost when numbers were stored as integers."""
        if self.digit_length is not None and value.isdigit() and len(value) < self.digit_length:
            value = value.zfill(self.digit_length)
        return self.matches(value)

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


def _recognizer(
    category: DataCategory,
    label: str,
    pattern: str,
    validator: Callable[[str], bool] | None = None,
    *,
    in_text: bool = True,
    needs_name_hint: bool = False,
    digit_length: int | None = None,
) -> ValueRecognizer:
    return ValueRecognizer(
        category,
        label,
        re.compile(pattern),
        validator,
        in_text=in_text,
        needs_name_hint=needs_name_hint,
        digit_length=digit_length,
    )


_C = DataCategory

# Phone layouts, as alternatives. Each alternative fixes the size of its digit groups, so
# CPF (3-3-3-2), IPv4, version strings and dates never fit. The digit count is left to
# `is_phone_number`.
_PHONE_LAYOUTS = (
    # An international prefix makes any grouping safe: +44 7911 123456, +14155552671.
    r"\+\d{1,3}(?:[\s.-]?\(?\d{1,5}\)?){2,5}",
    # North America: 3-3-4, with or without a leading 1 and brackets around the area code.
    r"(?:1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
    # Brazil, Portugal, the UK and Australia: area code, 4-5 digits, 4 digits.
    r"(?:\+?\d{1,3}[\s.-])?\(?\d{2,4}\)?[\s.-]?\d{4,5}[\s.-]?\d{4}",
    # 4-3-3 and 4-3-4: 0800 123 4567, 0412 345 678.
    r"\d{4}[\s.-]\d{3}[\s.-]\d{3,4}",
    # France: five pairs.
    r"\d{2}(?:[\s.-]\d{2}){4}",
    # UK mobiles and India: 4-5 digits, then 5-6.
    r"\d{4,5}[\s.-]\d{5,6}",
)
# The number must not continue into a longer run of digits on either side.
_PHONE_PATTERN = r"(?<![\w.+-])(?<!\d[\s.-])(?:" + "|".join(_PHONE_LAYOUTS) + r")(?![\w]|[\s.-]\d)"

# Order breaks exact ties between recognizers: earlier and more specific wins.
RECOGNIZERS: tuple[ValueRecognizer, ...] = (
    _recognizer(_C.EMAIL, "email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    _recognizer(
        _C.CNPJ,
        "CNPJ",
        r"\b[0-9A-Z]{2}\.?[0-9A-Z]{3}\.?[0-9A-Z]{3}/?[0-9A-Z]{4}-?\d{2}\b",
        checksums.is_valid_cnpj,
        digit_length=14,
    ),
    _recognizer(
        _C.CPF,
        "CPF",
        r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        checksums.is_valid_cpf,
        digit_length=11,
    ),
    _recognizer(
        _C.CNH, "CNH", r"\b\d{11}\b", checksums.is_valid_cnh, in_text=False, digit_length=11
    ),
    _recognizer(
        _C.NIS,
        "PIS/PASEP/NIS",
        r"\b\d{3}\.?\d{5}\.?\d{2}-?\d\b",
        checksums.is_valid_nis,
        in_text=False,
        digit_length=11,
    ),
    _recognizer(
        _C.VEHICLE_REGISTRATION,
        "RENAVAM",
        r"\b\d{11}\b",
        checksums.is_valid_renavam,
        in_text=False,
        needs_name_hint=True,
        digit_length=11,
    ),
    _recognizer(
        _C.HEALTH_CARD,
        "CNS",
        r"\b\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\b",
        checksums.is_valid_cns,
        in_text=False,
    ),
    _recognizer(
        _C.VOTER_ID,
        "Título de Eleitor",
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        checksums.is_valid_voter_id,
        in_text=False,
        digit_length=12,
    ),
    _recognizer(
        _C.INVOICE_KEY,
        "NF-e access key",
        r"\b(?:\d{4}\s?){10}\d{4}\b",
        checksums.is_valid_invoice_key,
    ),
    _recognizer(
        _C.VEHICLE_PLATE,
        "vehicle plate",
        r"\b[A-Z]{3}-?\d[A-Z0-9]\d{2}\b",
        in_text=False,
        needs_name_hint=True,
    ),
    _recognizer(
        _C.PIX_KEY,
        "PIX random key",
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        in_text=False,
        needs_name_hint=True,
    ),
    _recognizer(_C.IMEI, "IMEI", r"\b\d{2}-?\d{6}-?\d{6}-?\d\b", checksums.is_valid_imei),
    _recognizer(
        _C.CREDIT_CARD, "payment card", r"\b(?:\d[ -]?){12,18}\d\b", checksums.is_valid_card
    ),
    _recognizer(_C.SSN, "US SSN", r"\b\d{3}-\d{2}-\d{4}\b", checksums.is_plausible_ssn),
    _recognizer(_C.NATIONAL_ID, "US ITIN", r"\b9\d{2}-\d{2}-\d{4}\b", checksums.is_plausible_itin),
    _recognizer(
        _C.IBAN,
        "IBAN",
        r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4}\b",
        checksums.is_valid_iban,
    ),
    _recognizer(
        _C.IP_ADDRESS,
        "IPv4 address",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    ),
    _recognizer(
        _C.IP_ADDRESS,
        "IPv6 address",
        r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}",
        checksums.is_ip_address,
    ),
    _recognizer(
        _C.MAC_ADDRESS,
        "MAC address",
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
    ),
    _recognizer(_C.URL, "URL", r"https?://[^\s/$.?#][^\s]*"),
    _recognizer(_C.POSTAL_CODE, "CEP", r"\b\d{5}-\d{3}\b"),
    _recognizer(_C.PHONE, "phone number", _PHONE_PATTERN, checksums.is_phone_number),
    _recognizer(
        _C.NATIONAL_ID,
        "Aadhaar",
        r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b",
        checksums.is_valid_aadhaar,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Spanish DNI/NIE",
        r"\b[0-9XYZ]\d{7}-?[A-Z]\b",
        checksums.is_valid_spanish_id,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Portuguese NIF",
        r"\b[1-9]\d{8}\b",
        checksums.is_valid_portuguese_nif,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Dutch BSN",
        r"\b\d{9}\b",
        checksums.is_valid_dutch_bsn,
        in_text=False,
        digit_length=9,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "German tax ID",
        r"\b[1-9]\d{10}\b",
        checksums.is_valid_german_tax_id,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "UK National Insurance number",
        r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
        checksums.is_valid_nino,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Italian codice fiscale",
        r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b",
        checksums.is_valid_fiscal_code,
    ),
)

TEXT_RECOGNIZERS = tuple(recognizer for recognizer in RECOGNIZERS if recognizer.in_text)


def recognizers_for(categories: Iterable[DataCategory] | None) -> tuple[ValueRecognizer, ...]:
    """Return the free-text recognizers for the given categories, or all of them for None."""
    if categories is None:
        return TEXT_RECOGNIZERS
    wanted = set(categories)
    return tuple(recognizer for recognizer in TEXT_RECOGNIZERS if recognizer.category in wanted)
