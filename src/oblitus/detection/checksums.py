"""Check digits and structural rules that tell a real document number from random digits.

A random 11-digit number passes the CPF check about one time in a hundred, so a
column where most values pass is very likely CPFs. Each function takes the raw
value, ignores separators, and returns True only when the value is well formed.
"""

import ipaddress

_BRAZILIAN_AREA_CODES = frozenset(
    {11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 27, 28}
    | set(range(31, 36))
    | {37, 38}
    | set(range(41, 47))
    | {47, 48, 49, 51, 53, 54, 55}
    | set(range(61, 70))
    | {71, 73, 74, 75, 77, 79}
    | set(range(81, 90))
    | set(range(91, 100))
)

_VERHOEFF_MULTIPLY = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_PERMUTE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 8, 6, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

# Codice fiscale: the value of a character in an odd position (1st, 3rd, ...).
_FISCAL_CODE_ODD = dict(
    zip(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21]
        + [2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23],
        strict=True,
    )
)

_NINO_FORBIDDEN_PREFIXES = {"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"}


def digits_of(value: str) -> list[int]:
    """Return the decimal digits in a value, in order."""
    return [int(character) for character in value if character.isdigit()]


def _all_same(digits: list[int]) -> bool:
    return len(set(digits)) == 1


def _mod11_digit(digits: list[int], weights: list[int]) -> int:
    remainder = sum(digit * weight for digit, weight in zip(digits, weights, strict=True)) % 11
    return 0 if remainder < 2 else 11 - remainder


def is_valid_cpf(value: str) -> bool:
    """CPF, the Brazilian individual taxpayer number: two mod-11 check digits."""
    digits = digits_of(value)
    if len(digits) != 11 or _all_same(digits):
        return False
    for position in (9, 10):
        weights = list(range(position + 1, 1, -1))
        total = sum(
            digit * weight for digit, weight in zip(digits[:position], weights, strict=True)
        )
        if digits[position] != (total * 10) % 11 % 10:
            return False
    return True


def is_valid_cnpj(value: str) -> bool:
    """CNPJ, the Brazilian company number, numeric or alphanumeric (from July 2026).

    Each of the first twelve characters counts as its ASCII code minus 48, so
    digits keep their value and letters count from 17 upwards.
    """
    characters = [character for character in value.upper() if character.isalnum()]
    if len(characters) != 14 or not "".join(characters[12:]).isdigit():
        return False
    values = [ord(character) - 48 for character in characters]
    if _all_same(values):
        return False
    first_weights = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    first = _mod11_digit(values[:12], first_weights)
    second = _mod11_digit(values[:12] + [first], [6, *first_weights])
    return values[12] == first and values[13] == second


def is_valid_cnh(value: str) -> bool:
    """CNH, the Brazilian driving licence number: two check digits with an offset rule."""
    digits = digits_of(value)
    if len(digits) != 11 or _all_same(digits):
        return False
    first = sum(digit * (9 - index) for index, digit in enumerate(digits[:9])) % 11
    offset = 0
    if first >= 10:
        first = 0
        offset = 2
    second = sum(digit * (index + 1) for index, digit in enumerate(digits[:9])) % 11 - offset
    if second < 0:
        second += 11
    if second >= 10:
        second = 0
    return digits[9] == first and digits[10] == second


def is_valid_nis(value: str) -> bool:
    """PIS, PASEP, NIS or NIT, the Brazilian social security number.

    RENAVAM vehicle registrations use an equivalent check, so the column name has
    to separate the two.
    """
    digits = digits_of(value)
    if len(digits) != 11 or _all_same(digits):
        return False
    return digits[10] == _mod11_digit(digits[:10], [3, 2, 9, 8, 7, 6, 5, 4, 3, 2])


def is_valid_cns(value: str) -> bool:
    """CNS, the Brazilian national health card number (Cartão Nacional de Saúde)."""
    digits = digits_of(value)
    if len(digits) != 15 or digits[0] not in (1, 2, 7, 8, 9):
        return False
    if digits[0] in (7, 8, 9):
        return sum(digit * (15 - index) for index, digit in enumerate(digits)) % 11 == 0
    total = sum(digit * (15 - index) for index, digit in enumerate(digits[:11]))
    check = 11 - total % 11
    if check == 11:
        check = 0
    if check == 10:
        expected = digits[:11] + [0, 0, 1] + [11 - (total + 2) % 11]
    else:
        expected = digits[:11] + [0, 0, 0, check]
    return digits == expected


def is_valid_voter_id(value: str) -> bool:
    """Título de Eleitor: twelve digits, a state code from 01 to 28, two check digits.

    Some states turn a remainder of 0 into 1; both readings are accepted.
    """
    digits = digits_of(value)
    if len(digits) != 12 or not 1 <= digits[8] * 10 + digits[9] <= 28:
        return False
    first_total = sum(
        digit * weight for digit, weight in zip(digits[:8], range(2, 10), strict=True)
    )
    first = first_total % 11 % 10
    second = (digits[8] * 7 + digits[9] * 8 + first * 9) % 11 % 10
    if digits[10] == first and digits[11] == second:
        return True
    first_alternative = 1 if first_total % 11 == 0 else first
    second_alternative = (digits[8] * 7 + digits[9] * 8 + first_alternative * 9) % 11
    second_alternative = 1 if second_alternative == 0 else second_alternative % 10
    return digits[10] == first_alternative and digits[11] == second_alternative


def is_valid_renavam(value: str) -> bool:
    """RENAVAM, the Brazilian vehicle registration number: one mod-11 check digit."""
    digits = digits_of(value)
    if len(digits) != 11 or _all_same(digits):
        return False
    total = sum(
        digit * weight
        for digit, weight in zip(digits[:10], [3, 2, 9, 8, 7, 6, 5, 4, 3, 2], strict=True)
    )
    return digits[10] == (total * 10) % 11 % 10


def is_valid_invoice_key(value: str) -> bool:
    """The 44-digit access key of a Brazilian electronic invoice (NF-e, NFC-e, CT-e)."""
    digits = digits_of(value)
    if len(digits) != 44:
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9]
    total = sum(digit * weights[index % 8] for index, digit in enumerate(reversed(digits[:43])))
    remainder = total % 11
    expected = 0 if remainder < 2 else 11 - remainder
    return digits[43] == expected


def passes_luhn(value: str) -> bool:
    """The Luhn checksum carried by payment cards and IMEI numbers."""
    digits = digits_of(value)
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return bool(digits) and total % 10 == 0


def _card_lengths(number: str) -> range | tuple[int, ...]:
    """Return the lengths the issuing network of this card prefix uses, or none."""
    if number[:2] in ("34", "37"):
        return (15,)
    if number[0] == "4":
        return (13, 16, 19)
    if number[:2] in ("51", "52", "53", "54", "55") or 2221 <= int(number[:4]) <= 2720:
        return (16,)
    # Diners Club.
    if number[:2] in ("36", "38", "39") or number[:3] in ("300", "301", "302", "303", "304", "305"):
        return range(14, 20)
    # JCB (35), UnionPay, Discover, Hipercard and Elo (6), Maestro and Elo (50, 56-58).
    if number[:2] in ("35", "50", "56", "57", "58") or number[0] == "6":
        return range(16, 20)
    return ()


def is_valid_card(value: str) -> bool:
    """A payment card number: Luhn, a known network prefix and a length that network uses.

    The length rule keeps IMEIs (15 digits, Luhn) and order numbers out: 15-digit
    cards only exist on American Express, whose numbers start with 34 or 37.
    """
    digits = digits_of(value)
    if not 13 <= len(digits) <= 19 or not passes_luhn(value):
        return False
    number = "".join(map(str, digits))
    return len(number) in _card_lengths(number)


def is_valid_imei(value: str) -> bool:
    """IMEI, the 15-digit mobile device identifier, with its Luhn check digit."""
    digits = digits_of(value)
    text = "".join(map(str, digits))
    return len(digits) == 15 and passes_luhn(value) and text[:2] not in ("34", "37")


def is_valid_iban(value: str) -> bool:
    """IBAN: the ISO 13616 mod-97 checksum."""
    compact = value.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(character, 36)) for character in rearranged)
    return int(numeric) % 97 == 1


def is_plausible_ssn(value: str) -> bool:
    """US Social Security number outside the ranges that are never issued."""
    area, group, serial = value.split("-")
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def is_plausible_itin(value: str) -> bool:
    """US ITIN: starts with 9, middle group 70-88 or 90-99 except 93."""
    digits = digits_of(value)
    if len(digits) != 9 or digits[0] != 9:
        return False
    middle = digits[3] * 10 + digits[4]
    return 70 <= middle <= 88 or (90 <= middle <= 99 and middle != 93)


def is_phone_number(value: str) -> bool:
    """A phone number with an area code.

    Formatted numbers (with +, brackets, spaces or dashes) need 10 to 13 digits. Bare
    digit strings must look like a Brazilian mobile, area code then 9, because bare
    10 to 12 digit numbers are far more often timestamps or order numbers.
    """
    digits = digits_of(value)
    if not 10 <= len(digits) <= 13:
        return False
    if any(character in value for character in "+()- ."):
        return True
    if len(digits) == 13 and digits[:2] == [5, 5]:
        digits = digits[2:]
    area_code = digits[0] * 10 + digits[1]
    return len(digits) == 11 and digits[2] == 9 and area_code in _BRAZILIAN_AREA_CODES


def is_valid_aadhaar(value: str) -> bool:
    """Aadhaar, the Indian national ID: 12 digits, not starting with 0 or 1, Verhoeff check."""
    digits = digits_of(value)
    if len(digits) != 12 or digits[0] < 2:
        return False
    check = 0
    for index, digit in enumerate(reversed(digits)):
        check = _VERHOEFF_MULTIPLY[check][_VERHOEFF_PERMUTE[index % 8][digit]]
    return check == 0


def is_valid_spanish_id(value: str) -> bool:
    """Spanish DNI (8 digits and a letter) or NIE (X, Y or Z, 7 digits and a letter)."""
    compact = value.replace("-", "").upper()
    if len(compact) != 9:
        return False
    number = compact[:8]
    if number[0] in "XYZ":
        number = str("XYZ".index(number[0])) + number[1:]
    if not number.isdigit():
        return False
    return compact[8] == _DNI_LETTERS[int(number) % 23]


def is_valid_portuguese_nif(value: str) -> bool:
    """Portuguese NIF: 9 digits, not starting with 0, mod-11 check digit."""
    digits = digits_of(value)
    if len(digits) != 9 or digits[0] == 0:
        return False
    total = sum(digit * (9 - index) for index, digit in enumerate(digits[:8]))
    return digits[8] == (11 - total % 11) % 11 % 10


def is_valid_dutch_bsn(value: str) -> bool:
    """Dutch BSN: 9 digits passing the eleven-test."""
    digits = digits_of(value)
    if len(digits) != 9 or _all_same(digits):
        return False
    total = sum(digit * (9 - index) for index, digit in enumerate(digits[:8])) - digits[8]
    return total % 11 == 0


def is_valid_german_tax_id(value: str) -> bool:
    """German Steuer-IdNr: 11 digits, one digit repeated 2 or 3 times, ISO 7064 check."""
    digits = digits_of(value)
    if len(digits) != 11 or digits[0] == 0:
        return False
    counts = sorted((digits[:10].count(digit) for digit in set(digits[:10])), reverse=True)
    if counts[0] not in (2, 3) or counts[1:].count(1) != len(counts) - 1:
        return False
    product = 10
    for digit in digits[:10]:
        total = (digit + product) % 10 or 10
        product = (total * 2) % 11
    return digits[10] == (11 - product) % 10


def is_valid_nino(value: str) -> bool:
    """UK National Insurance number: allowed two-letter prefix, six digits, suffix A-D."""
    compact = value.replace(" ", "").upper()
    if len(compact) != 9 or not compact[2:8].isdigit() or compact[8] not in "ABCD":
        return False
    first, second = compact[0], compact[1]
    if first in "DFIQUV" or second in "DFIQUVO":
        return False
    return compact[:2] not in _NINO_FORBIDDEN_PREFIXES


def is_valid_fiscal_code(value: str) -> bool:
    """Italian codice fiscale: 16 characters with a check letter."""
    compact = value.upper()
    if len(compact) != 16 or not compact.isalnum():
        return False
    total = 0
    for index, character in enumerate(compact[:15]):
        if index % 2 == 0:
            total += _FISCAL_CODE_ODD[character]
        else:
            total += int(character) if character.isdigit() else ord(character) - ord("A")
    return compact[15] == chr(total % 26 + ord("A"))


def has_enough_ipv6_groups(value: str) -> bool:
    """True when an address-like span has at least three groups of hex digits.

    In prose, `::` and `std::vector` parse as valid IPv6 addresses, so a span found inside
    text needs more than that to count.
    """
    return sum(1 for group in value.split(":") if group) >= 3


def is_ip_address(value: str) -> bool:
    """An IPv4 or IPv6 address, as the standard library parses it."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True
