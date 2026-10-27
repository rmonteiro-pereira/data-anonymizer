"""Recognizers that decide whether a value looks like a known kind of personal data.

Each recognizer pairs a regular expression with an optional checksum from
`checksums.py`. These settings limit where a recognizer may fire:
- `in_text`: whether it is searched for inside free text. Bare-digit formats such
  as NIS, CNH or IMEI are left out, because prose is full of numbers that fit them.
- `strong`: whether a match is unlikely to be an accident (about 1-3% of random values pass).
  The value evidence trusts a strong recognizer at half the values, and keeps it in
  columns whose names say they hold business codes.
- `chance_rate`: how often a random number of the right shape passes the validator. Free
  text must show clearly more hits than this before it counts as holding personal data.
- `max_lost_zeros`: how many leading zeros may be restored, for checks that random
  numbers often pass.
- `needs_name_hint`: whether a matching column also needs a column name that
  points to the category, for formats with no checksum that ordinary codes share,
  such as UUIDs, which may be PIX keys or plain record keys.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from oblitus.detection import checksums, crypto_wallets, passport_mrz
from oblitus.detection.identifier_boundaries import (
    NOT_AFTER_ALPHANUMERIC,
    NOT_BEFORE_ALPHANUMERIC,
    bounded,
)
from oblitus.detection.international_identifiers import (
    STDNUM_IDENTIFIERS,
    stdnum_is_installed,
    stdnum_validator,
)
from oblitus.detection.place_names import CITY_PATTERN, is_municipality_name
from oblitus.taxonomy import DataCategory

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
    max_lost_zeros: int | None = None
    text_validator: Callable[[str], bool] | None = None
    chance_rate: float = 0.0
    strong: bool = False

    def matches(self, value: str) -> bool:
        """Return True when the whole value is an instance of this kind."""
        if self.pattern.fullmatch(value) is None:
            return False
        return self._is_valid(value)

    def matches_padded(self, value: str) -> bool:
        """Like `matches`, but restores leading zeros lost to integer storage or a spreadsheet."""
        if self._lost_zeros(value):
            value = value.zfill(self.digit_length or len(value))
        return self.matches(value)

    def _lost_zeros(self, value: str) -> bool:
        if self.digit_length is None or not value.isdigit():
            return False
        missing = self.digit_length - len(value)
        if self.max_lost_zeros is None:
            return missing > 0
        return 0 < missing <= self.max_lost_zeros

    def find_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the start and end of every valid occurrence inside free text."""
        spans = []
        for match in self.pattern.finditer(text):
            found = match.group(0)
            if self._is_valid(found) and self._is_valid_in_text(found):
                spans.append(match.span())
        return spans

    def _is_valid(self, value: str) -> bool:
        if self.validator is None:
            return True
        return self.validator(value)

    def _is_valid_in_text(self, value: str) -> bool:
        if self.text_validator is None:
            return True
        return self.text_validator(value)


def _recognizer(
    category: DataCategory,
    label: str,
    pattern: str,
    validator: Callable[[str], bool] | None = None,
    *,
    in_text: bool = True,
    needs_name_hint: bool = False,
    digit_length: int | None = None,
    max_lost_zeros: int | None = None,
    text_validator: Callable[[str], bool] | None = None,
    chance_rate: float = 0.0,
    strong: bool = False,
) -> ValueRecognizer:
    return ValueRecognizer(
        category,
        label,
        re.compile(pattern),
        validator,
        in_text=in_text,
        needs_name_hint=needs_name_hint,
        digit_length=digit_length,
        max_lost_zeros=max_lost_zeros,
        text_validator=text_validator,
        chance_rate=chance_rate,
        strong=strong,
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
_PHONE_PATTERN = (
    NOT_AFTER_ALPHANUMERIC
    + r"(?<![.+-])(?<!\d[\s.-])(?:"
    + "|".join(_PHONE_LAYOUTS)
    + r")"
    + NOT_BEFORE_ALPHANUMERIC
    + r"(?![\s.-]\d)"
)

# The local part is possessive and may only start after a character outside its class. Without
# that, a long cell with no "@" is rescanned from every position, which takes quadratic time.
_EMAIL_PATTERN = r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

# A one-digit check passes one random number in ten, so padding a short number with many
# zeros would make small codes look like IMEIs or BSNs. Real leading zeros are one or two.
_WEAK_CHECK_LOST_ZEROS = 2

_BITCOIN_PATTERN = (
    f"(?:{bounded('bc1[02-9ac-hj-np-z]{11,71}', ignore_case=True)})"
    f"|(?:{bounded('[13][a-km-zA-HJ-NP-Z1-9]{25,34}')})"
)

# Order breaks exact ties between recognizers: earlier and more specific wins.
_BUILT_IN_RECOGNIZERS: tuple[ValueRecognizer, ...] = (
    _recognizer(_C.EMAIL, "email", _EMAIL_PATTERN, strong=True),
    _recognizer(
        _C.CNPJ,
        "CNPJ",
        bounded(r"[0-9A-Z]{2}\.?[0-9A-Z]{3}\.?[0-9A-Z]{3}/?[0-9A-Z]{4}-?\d{2}", ignore_case=True),
        checksums.is_valid_cnpj,
        digit_length=14,
        chance_rate=0.01,
        strong=True,
    ),
    _recognizer(
        _C.CPF,
        "CPF",
        bounded(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),
        checksums.is_valid_cpf,
        digit_length=11,
        chance_rate=0.01,
        strong=True,
    ),
    _recognizer(
        _C.CNH,
        "CNH",
        bounded(r"\d{11}"),
        checksums.is_valid_cnh,
        in_text=False,
        digit_length=11,
        strong=True,
    ),
    _recognizer(
        _C.NIS,
        "PIS/PASEP/NIS",
        bounded(r"\d{3}\.?\d{5}\.?\d{2}-?\d"),
        checksums.is_valid_nis,
        in_text=False,
        digit_length=11,
    ),
    _recognizer(
        _C.VEHICLE_REGISTRATION,
        "RENAVAM",
        bounded(r"\d{11}"),
        checksums.is_valid_renavam,
        in_text=False,
        needs_name_hint=True,
        digit_length=11,
    ),
    _recognizer(
        _C.HEALTH_CARD,
        "CNS",
        bounded(r"\d{3}\s?\d{4}\s?\d{4}\s?\d{4}"),
        checksums.is_valid_cns,
        in_text=False,
        strong=True,
    ),
    _recognizer(
        _C.VOTER_ID,
        "Título de Eleitor",
        bounded(r"\d{4}\s?\d{4}\s?\d{4}"),
        checksums.is_valid_voter_id,
        in_text=False,
        digit_length=12,
        strong=True,
    ),
    _recognizer(
        _C.INVOICE_KEY,
        "NF-e access key",
        bounded(r"(?:\d{4}\s?){10}\d{4}"),
        checksums.is_valid_invoice_key,
        chance_rate=0.09,
    ),
    _recognizer(
        _C.VEHICLE_PLATE,
        "vehicle plate",
        bounded(r"[A-Z]{3}-?\d[A-Z0-9]\d{2}", ignore_case=True),
        in_text=False,
        needs_name_hint=True,
    ),
    _recognizer(
        _C.PIX_KEY,
        "PIX random key",
        bounded(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            ignore_case=True,
        ),
        in_text=False,
        needs_name_hint=True,
    ),
    _recognizer(
        _C.IMEI,
        "IMEI",
        bounded(r"\d{2}-?\d{6}-?\d{6}-?\d"),
        checksums.is_valid_imei,
        in_text=False,
        digit_length=15,
        max_lost_zeros=_WEAK_CHECK_LOST_ZEROS,
    ),
    _recognizer(
        _C.CREDIT_CARD,
        "payment card",
        bounded(r"(?:\d[ -]?){12,18}\d"),
        checksums.is_valid_card,
        chance_rate=0.04,
    ),
    _recognizer(_C.SSN, "US SSN", bounded(r"\d{3}-\d{2}-\d{4}"), checksums.is_plausible_ssn),
    _recognizer(
        _C.NATIONAL_ID, "US ITIN", bounded(r"9\d{2}-\d{2}-\d{4}"), checksums.is_plausible_itin
    ),
    _recognizer(
        _C.IBAN,
        "IBAN",
        bounded(r"[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}\s?[A-Z0-9]{1,4}", ignore_case=True),
        checksums.is_valid_iban,
        chance_rate=0.01,
        strong=True,
    ),
    _recognizer(
        _C.IP_ADDRESS,
        "IPv4 address",
        bounded(r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"),
    ),
    _recognizer(
        _C.IP_ADDRESS,
        "IPv6 address",
        r"(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}",
        checksums.is_ip_address,
        text_validator=checksums.has_enough_ipv6_groups,
    ),
    _recognizer(
        _C.MAC_ADDRESS,
        "MAC address",
        bounded(r"(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}"),
    ),
    _recognizer(_C.URL, "URL", r"https?://[^\s/$.?#][^\s]*", strong=True),
    _recognizer(_C.POSTAL_CODE, "CEP", bounded(r"\d{5}-\d{3}")),
    _recognizer(
        _C.PHONE, "phone number", _PHONE_PATTERN, checksums.is_phone_number, chance_rate=0.08
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Aadhaar",
        bounded(r"[2-9]\d{3}\s?\d{4}\s?\d{4}"),
        checksums.is_valid_aadhaar,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Spanish DNI/NIE",
        bounded(r"[0-9XYZ]\d{7}-?[A-Z]", ignore_case=True),
        checksums.is_valid_spanish_id,
        chance_rate=0.04,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Portuguese NIF",
        bounded(r"[1-9]\d{8}"),
        checksums.is_valid_portuguese_nif,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Dutch BSN",
        bounded(r"\d{9}"),
        checksums.is_valid_dutch_bsn,
        in_text=False,
        digit_length=9,
        max_lost_zeros=_WEAK_CHECK_LOST_ZEROS,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "German tax ID",
        bounded(r"[1-9]\d{10}"),
        checksums.is_valid_german_tax_id,
        in_text=False,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "UK National Insurance number",
        bounded(r"[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]", ignore_case=True),
        checksums.is_valid_nino,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "Italian codice fiscale",
        bounded(
            r"[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]",
            ignore_case=True,
        ),
        checksums.is_valid_fiscal_code,
        chance_rate=0.04,
    ),
    _recognizer(
        _C.NATIONAL_ID,
        "passport machine-readable zone",
        passport_mrz.MRZ_PATTERN,
        passport_mrz.is_valid_passport_mrz,
        strong=True,
    ),
    _recognizer(
        _C.CRYPTO_WALLET,
        "Bitcoin address",
        _BITCOIN_PATTERN,
        crypto_wallets.is_valid_bitcoin_address,
        strong=True,
    ),
    _recognizer(
        _C.CRYPTO_WALLET,
        "Ethereum address",
        bounded(r"0[xX][0-9a-fA-F]{40}"),
        crypto_wallets.is_valid_ethereum_address,
        strong=True,
    ),
    _recognizer(
        _C.CITY, "Brazilian municipality", CITY_PATTERN, is_municipality_name, in_text=False
    ),
)


def _stdnum_recognizers() -> tuple[ValueRecognizer, ...]:
    """National numbers of more countries; empty unless `python-stdnum` is installed."""
    if not stdnum_is_installed():
        return ()
    return tuple(
        _recognizer(
            _C.NATIONAL_ID,
            identifier.label,
            identifier.pattern,
            stdnum_validator(identifier.module),
            chance_rate=0.01,
        )
        for identifier in STDNUM_IDENTIFIERS
    )


RECOGNIZERS = _BUILT_IN_RECOGNIZERS + _stdnum_recognizers()

TEXT_RECOGNIZERS = tuple(recognizer for recognizer in RECOGNIZERS if recognizer.in_text)


def recognizers_for(categories: Iterable[DataCategory] | None) -> tuple[ValueRecognizer, ...]:
    """Return the free-text recognizers for the given categories, or all of them for None."""
    if categories is None:
        return TEXT_RECOGNIZERS
    wanted = set(categories)
    return tuple(recognizer for recognizer in TEXT_RECOGNIZERS if recognizer.category in wanted)
