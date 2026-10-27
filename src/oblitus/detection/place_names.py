"""Tell whether a value is the name of a Brazilian municipality."""

from oblitus.detection.reference_lists import municipalities, normalize_word

# Municipality names are letters, spaces and a few joiners, such as "Mogi-Mirim" or "D'Oeste".
CITY_PATTERN = r"[^\W\d_](?:[^\W\d_]|[ '’.-]){1,60}"


def is_municipality_name(value: str) -> bool:
    """True when the value, ignoring case and accents, is an IBGE municipality name."""
    return normalize_word(value) in municipalities()
