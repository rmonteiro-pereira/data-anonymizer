"""Undo the tricks that hide an identifier from a pattern, and remember where each character was.

Someone who wants an identifier to survive redaction can type it with zero-width characters
inside, with full-width or other-script digits, with Cyrillic letters that look Latin, or
spelled out as `name [at] example [dot] com`. The recognizers only read plain text, so they
read a normalized copy. Every character of the copy remembers its position in the original,
so a match found in the copy can still be cut out of the original.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# Letters of other scripts that look like a Latin letter. Only the common ones are listed.
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "в": "b",
        "с": "c",
        "е": "e",
        "н": "h",
        "і": "i",
        "ј": "j",
        "к": "k",
        "м": "m",
        "о": "o",
        "р": "p",
        "ѕ": "s",
        "т": "t",
        "у": "y",
        "х": "x",
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "І": "I",
        "Ј": "J",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Ѕ": "S",
        "Т": "T",
        "Х": "X",
        "α": "a",
        "ο": "o",
        "ν": "v",
        "ρ": "p",
        "ι": "i",
        "κ": "k",
        "Α": "A",
        "Β": "B",
        "Ε": "E",
        "Ζ": "Z",
        "Η": "H",
        "Ι": "I",
        "Κ": "K",
        "Μ": "M",
        "Ν": "N",
        "Ο": "O",
        "Ρ": "P",
        "Τ": "T",
        "Υ": "Y",
        "Χ": "X",
    }
)

_AT_WORDS = r"(?:at|arroba)"
_DOT_WORDS = r"(?:dot|ponto)"
_BRACKETS_OPEN = r"[\[\(\{<]"
_BRACKETS_CLOSE = r"[\]\)\}>]"

# `[at]`, `(at)` and the like are an at-sign wherever they stand.
_BRACKETED_AT = re.compile(
    rf"\s*{_BRACKETS_OPEN}\s*{_AT_WORDS}\s*{_BRACKETS_CLOSE}\s*", re.IGNORECASE
)
_BRACKETED_DOT = re.compile(
    rf"\s*{_BRACKETS_OPEN}\s*{_DOT_WORDS}\s*{_BRACKETS_CLOSE}\s*", re.IGNORECASE
)
# A bare "at" is the at-sign only when what follows looks like a domain name, which ends in a
# top-level domain in one case. That keeps "we met at work.Then" from reading as an email.
_BARE_AT = re.compile(
    rf"(?<=[\w.-])\s+{_AT_WORDS}\s+(?=[\w-]+(?:\.|\s+{_DOT_WORDS}\s+)(?-i:[a-z]{{2}}|[A-Z]{{2}}))",
    re.IGNORECASE,
)
# A bare "dot" is a full stop only inside the domain after an at-sign.
_DOMAIN_AFTER_AT = re.compile(rf"(?<=@)[\w-]+(?:\s+{_DOT_WORDS}\s+[\w-]+)+", re.IGNORECASE)
_BARE_DOT = re.compile(rf"\s+{_DOT_WORDS}\s+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """A cleaned copy of a text, with the original position of each of its characters."""

    text: str
    origins: list[int]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a span of the copy to the span of the original text that produced it."""
        return self.origins[start], self.origins[end - 1] + 1


def normalize_text(text: str) -> NormalizedText:
    """Return the text with hidden characters removed, look-alikes mapped and `[at]` undone."""
    characters: list[str] = []
    origins: list[int] = []
    for position, character in enumerate(text):
        for replacement in _plain_characters(character):
            characters.append(replacement)
            origins.append(position)
    cleaned = NormalizedText("".join(characters), origins)
    cleaned = _substitute(cleaned, _BRACKETED_AT, lambda _match: "@")
    cleaned = _substitute(cleaned, _BRACKETED_DOT, lambda _match: ".")
    cleaned = _substitute(cleaned, _BARE_AT, lambda _match: "@")
    return _substitute(cleaned, _DOMAIN_AFTER_AT, _dots_in_domain)


def _plain_characters(character: str) -> str:
    if unicodedata.category(character) == "Cf":
        # Zero-width spaces and joiners, byte-order marks and soft hyphens print nothing.
        return ""
    if character.isdecimal():
        return str(unicodedata.decimal(character))
    return unicodedata.normalize("NFKC", character).translate(_HOMOGLYPHS)


def _dots_in_domain(match: re.Match[str]) -> str:
    return _BARE_DOT.sub(".", match.group(0))


def _substitute(
    cleaned: NormalizedText, pattern: re.Pattern[str], replace: Callable[[re.Match[str]], str]
) -> NormalizedText:
    """Replace every match, keeping each new character's origin inside the match it replaces."""
    pieces: list[str] = []
    origins: list[int] = []
    cursor = 0
    for match in pattern.finditer(cleaned.text):
        pieces.append(cleaned.text[cursor : match.start()])
        origins.extend(cleaned.origins[cursor : match.start()])
        replacement = replace(match)
        pieces.append(replacement)
        origins.extend(_replacement_origins(cleaned.origins, match, len(replacement)))
        cursor = match.end()
    pieces.append(cleaned.text[cursor:])
    origins.extend(cleaned.origins[cursor:])
    return NormalizedText("".join(pieces), origins)


def _replacement_origins(origins: list[int], match: re.Match[str], length: int) -> list[int]:
    # The first new character stands for the start of the match, the last for its end.
    if length == 0:
        return []
    first = origins[match.start()]
    last = origins[match.end() - 1]
    return [first] * (length - 1) + [last]
