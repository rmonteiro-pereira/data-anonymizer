"""Lists of Brazilian first names, surnames and municipalities, read from the package data.

The files come from IBGE public data (see `data/README.md`). Words are compared without
accents and in lower case, so `São Paulo`, `SAO PAULO` and `sao paulo` are the same entry.
"""

import unicodedata
from functools import cache
from importlib import resources


def normalize_word(text: str) -> str:
    """Lower-case a text, remove its accents and collapse its spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(plain.lower().split())


@cache
def first_names() -> frozenset[str]:
    """The most frequent Brazilian first names of the 2010 Census."""
    return _read_list("first_names.txt")


@cache
def surnames() -> frozenset[str]:
    """Common Brazilian surnames."""
    return _read_list("surnames.txt")


@cache
def municipalities() -> frozenset[str]:
    """The names of the Brazilian municipalities."""
    return _read_list("municipalities.txt")


def _read_list(filename: str) -> frozenset[str]:
    text = resources.files("oblitus.detection").joinpath("data", filename).read_text("utf-8")
    return frozenset(normalize_word(line) for line in text.splitlines() if line.strip())
