"""Tell whether text values look like the names of people.

The shape is two to five alphabetic words, written in title case, with the lowercase
particles that names carry ("da Silva", "van der Berg"). Digits, punctuation and single
words rule a value out, which keeps cities, plans and codes from passing.
"""

import re

import pandas as pd

from oblitus.detection.reference_lists import first_names, normalize_word, surnames

_MIN_WORDS = 2
_MAX_WORDS = 5

# Share of values that must have the shape for the column to count as names.
_NAME_SHAPED_SHARE = 0.8

# Share of values that must have the shape and carry a known first name or surname, for a column
# whose own name says nothing.
_KNOWN_NAME_SHARE = 0.7

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


def looks_like_known_names(sample: pd.Series) -> bool:
    """True when most values have the shape of a name that starts with a common first name.

    A name also counts when it ends with a common surname. The first-name list comes from
    the IBGE 2010 Census. The lists let a column called `contact` or `col_3` be recognized as
    people from its values alone.
    """
    if sample.empty:
        return False
    texts = sample.astype(str)
    known = texts.map(lambda text: _is_name_shaped(text) and _has_known_name_part(text))
    return float(known.mean()) >= _KNOWN_NAME_SHARE


def _has_known_name_part(value: str) -> bool:
    words = value.split()
    starts_with_first_name = normalize_word(words[0]) in first_names()
    ends_with_surname = normalize_word(words[-1]) in surnames()
    return starts_with_first_name or ends_with_surname
