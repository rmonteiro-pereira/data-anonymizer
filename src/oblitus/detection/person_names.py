"""Tell whether text values look like the names of people.

The shape is two to five alphabetic words, written in title case, with the lowercase
particles that names carry ("da Silva", "van der Berg"). Digits, punctuation and single
words rule a value out, which keeps cities, plans and codes from passing.
"""

import re

import pandas as pd

_MIN_WORDS = 2
_MAX_WORDS = 5

# Share of values that must have the shape for the column to count as names.
_NAME_SHAPED_SHARE = 0.8

_PARTICLES = frozenset(
    {"de", "da", "do", "das", "dos", "e", "di", "du", "del", "van", "von", "der", "la", "le", "y"}
)

# Letters with accents, joined by apostrophes or hyphens, and a final dot for "Jr." or "Dr.".
_NAME_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*\.?")


def looks_like_person_names(sample: pd.Series) -> bool:
    """True when most of the values have the shape of a person's name."""
    if sample.empty:
        return False
    shaped = sample.astype(str).map(_is_name_shaped)
    return float(shaped.mean()) >= _NAME_SHAPED_SHARE


def _is_name_shaped(value: str) -> bool:
    words = value.split()
    if not _MIN_WORDS <= len(words) <= _MAX_WORDS:
        return False
    return all(_is_name_word(word) for word in words)


def _is_name_word(word: str) -> bool:
    if _NAME_WORD.fullmatch(word) is None:
        return False
    return word[0].isupper() or word.lower() in _PARTICLES
