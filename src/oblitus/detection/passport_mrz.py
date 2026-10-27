"""The machine-readable zone of a passport (ICAO 9303, format TD3) and its check digits.

A TD3 zone is two lines of 44 characters. The second line holds the passport number, the
birth date, the expiry date and an optional personal number, each followed by a check digit,
and a final digit that covers all of them.
"""

_LINE_LENGTH = 44
_WEIGHTS = (7, 3, 1)
_FILLER = "<"

# Positions in the second line: (first character, check digit position).
_PASSPORT_NUMBER = (0, 9)
_BIRTH_DATE = (13, 19)
_EXPIRY_DATE = (21, 27)
_PERSONAL_NUMBER = (28, 42)
_COMPOSITE_PARTS = ((0, 10), (13, 20), (21, 43))
_COMPOSITE_CHECK = 43

# Both lines, with at most two characters (a line break or spaces) between them.
MRZ_PATTERN = (
    r"(?<![A-Z0-9<])P[A-Z<][A-Z<]{3}[A-Z<]{39}\s{0,2}"
    r"[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d[MF<]\d{6}\d[A-Z0-9<]{14}[\d<]\d(?![A-Z0-9<])"
)


def is_valid_passport_mrz(value: str) -> bool:
    """True when the two lines of a TD3 zone have valid check digits."""
    second_line = _second_line(value)
    if second_line is None:
        return False
    fields = (_PASSPORT_NUMBER, _BIRTH_DATE, _EXPIRY_DATE, _PERSONAL_NUMBER)
    if not all(_field_check_matches(second_line, field) for field in fields):
        return False
    composite = "".join(second_line[start:end] for start, end in _COMPOSITE_PARTS)
    return weighted_check_digit(composite) == int(second_line[_COMPOSITE_CHECK])


def _second_line(value: str) -> str | None:
    compact = "".join(value.split())
    if len(compact) != 2 * _LINE_LENGTH:
        return None
    return compact[_LINE_LENGTH:]


def _field_check_matches(line: str, field: tuple[int, int]) -> bool:
    start, check_position = field
    character = line[check_position]
    text = line[start:check_position]
    if character == _FILLER:
        # An empty optional field has a filler instead of a digit.
        return set(text) == {_FILLER}
    return character.isdigit() and weighted_check_digit(text) == int(character)


def weighted_check_digit(text: str) -> int:
    """Return the ICAO 9303 check digit of a field: weights 7, 3, 1 over its characters."""
    total = sum(
        _character_value(character) * _WEIGHTS[index % 3] for index, character in enumerate(text)
    )
    return total % 10


def _character_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if character == _FILLER:
        return 0
    return ord(character) - ord("A") + 10
