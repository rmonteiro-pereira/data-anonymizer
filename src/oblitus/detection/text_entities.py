"""Find personal data inside free text, including data typed to dodge a pattern.

The recognizers read a normalized copy of the text (see `text_normalization.py`). Two
disguises need more than a clean copy:
- a document number whose digits are split by spaces or odd punctuation, such as
  `529 982 247 25` or `52.9982.2472.5`, counts when its check digits validate it;
- a phone number whose digit groups are separated by number words, such as
  `+55 onze 99876 cinco 4321`.
Spans are returned in the coordinates of the original text, so redaction removes exactly
what was typed.
"""

import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass

from oblitus.detection import checksums
from oblitus.detection.named_entities import NamedEntityBackend
from oblitus.detection.patterns import ValueRecognizer
from oblitus.detection.text_normalization import NormalizedText, normalize_text
from oblitus.taxonomy import DataCategory

Span = tuple[int, int]
SpanFinder = Callable[[str], list[Span]]

# Digit groups closer together than this many separator characters belong to one number.
_MAX_SEPARATOR = 2

_DIGIT_GROUP = re.compile(r"\d+")

_PHONE_MIN_DIGITS = 10
_PHONE_MAX_DIGITS = 15

_NUMBER_WORDS = {
    "zero": "0",
    "cero": "0",
    "um": "1",
    "uma": "1",
    "one": "1",
    "uno": "1",
    "dois": "2",
    "duas": "2",
    "two": "2",
    "dos": "2",
    "três": "3",
    "tres": "3",
    "three": "3",
    "quatro": "4",
    "four": "4",
    "cuatro": "4",
    "cinco": "5",
    "five": "5",
    "seis": "6",
    "six": "6",
    "sete": "7",
    "seven": "7",
    "siete": "7",
    "oito": "8",
    "eight": "8",
    "ocho": "8",
    "nove": "9",
    "nine": "9",
    "nueve": "9",
    "dez": "10",
    "ten": "10",
    "diez": "10",
    "onze": "11",
    "eleven": "11",
    "once": "11",
    "doze": "12",
    "twelve": "12",
    "doce": "12",
    "treze": "13",
    "thirteen": "13",
    "trece": "13",
    "catorze": "14",
    "quatorze": "14",
    "fourteen": "14",
    "catorce": "14",
    "quinze": "15",
    "fifteen": "15",
    "quince": "15",
    "dezesseis": "16",
    "sixteen": "16",
    "dezessete": "17",
    "seventeen": "17",
    "dezoito": "18",
    "eighteen": "18",
    "dezenove": "19",
    "nineteen": "19",
    "vinte": "20",
    "twenty": "20",
}
_WORD_ALTERNATIVES = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
_PHONE_TOKEN = rf"(?:\d+|(?:{_WORD_ALTERNATIVES})\b)"
_SPELLED_PHONE = re.compile(
    rf"(?<!\w)\+?{_PHONE_TOKEN}(?:[\s.-]{{1,{_MAX_SEPARATOR}}}{_PHONE_TOKEN}){{2,}}(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LabelledSpan:
    """A stretch of the original text that holds one kind of personal data."""

    start: int
    end: int
    category: DataCategory


def find_entities(
    text: str,
    recognizers: Iterable[ValueRecognizer],
    named_entities: NamedEntityBackend | None = None,
    named_categories: Collection[DataCategory] | None = None,
) -> list[LabelledSpan]:
    """Return the personal data in a text, in the order of the text, without overlaps.

    When two recognizers claim the same characters, the one listed first keeps them. A
    named-entity backend reads last, so a name never overrides a document number. With
    `named_categories`, the backend's findings of other categories are dropped.
    """
    normalized = normalize_text(text)
    claimed: list[LabelledSpan] = []
    for recognizer in recognizers:
        spans = recognizer.find_spans(normalized.text)
        spans += _DISGUISED_FINDERS.get(recognizer.category, _no_spans)(normalized.text)
        for start, end in sorted(set(spans)):
            _claim(claimed, normalized, start, end, recognizer.category)
    if named_entities is not None:
        for start, end, category in sorted(named_entities.find(normalized.text)):
            if named_categories is None or category in named_categories:
                _claim(claimed, normalized, start, end, category)
    return sorted(claimed, key=lambda span: span.start)


def holds_entities(
    text: str,
    recognizers: Iterable[ValueRecognizer],
    named_entities: NamedEntityBackend | None = None,
) -> bool:
    """True when the text still holds personal data a recognizer or the backend can find."""
    return bool(find_entities(text, recognizers, named_entities))


def _claim(
    claimed: list[LabelledSpan],
    normalized: NormalizedText,
    start: int,
    end: int,
    category: DataCategory,
) -> None:
    """Add a span, given in the normalized text, unless it overlaps one already claimed."""
    original_start, original_end = normalized.original_span(start, end)
    span = LabelledSpan(original_start, original_end, category)
    if not any(_overlap(span, other) for other in claimed):
        claimed.append(span)


def _overlap(first: LabelledSpan, second: LabelledSpan) -> bool:
    return first.start < second.end and second.start < first.end


def _no_spans(_text: str) -> list[Span]:
    return []


def _split_document_spans(text: str, length: int, is_valid: Callable[[str], bool]) -> list[Span]:
    """Find numbers of `length` digits, split by spaces or punctuation, that pass a check."""
    spans = []
    for chunk in _digit_chunks(text):
        for first in range(len(chunk)):
            digits = ""
            for last in range(first, len(chunk)):
                digits += chunk[last].group(0)
                if len(digits) >= length:
                    break
            if len(digits) == length and is_valid(digits):
                spans.append((chunk[first].start(), chunk[last].end()))
    return spans


def _digit_chunks(text: str) -> list[list[re.Match[str]]]:
    """Group digit runs that sit within a couple of separator characters of each other."""
    chunks: list[list[re.Match[str]]] = []
    for group in _DIGIT_GROUP.finditer(text):
        if chunks and _joins(text, chunks[-1][-1], group):
            chunks[-1].append(group)
        else:
            chunks.append([group])
    return chunks


def _joins(text: str, previous: re.Match[str], following: re.Match[str]) -> bool:
    gap = text[previous.end() : following.start()]
    if not 0 < len(gap) <= _MAX_SEPARATOR:
        return False
    return all(not character.isalnum() and character not in "\r\n" for character in gap)


def _spelled_phone_spans(text: str) -> list[Span]:
    spans = []
    for match in _SPELLED_PHONE.finditer(text):
        tokens = re.findall(_PHONE_TOKEN, match.group(0), re.IGNORECASE)
        numerals = [token for token in tokens if token.isdigit()]
        spelled = [token for token in tokens if not token.isdigit()]
        digits = sum(len(_NUMBER_WORDS.get(token.lower(), token)) for token in tokens)
        if len(numerals) >= 2 and spelled and _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS:
            spans.append(match.span())
    return spans


def _cpf_spans(text: str) -> list[Span]:
    return _split_document_spans(text, 11, checksums.is_valid_cpf)


def _cnpj_spans(text: str) -> list[Span]:
    return _split_document_spans(text, 14, checksums.is_valid_cnpj)


_DISGUISED_FINDERS: dict[DataCategory, SpanFinder] = {
    DataCategory.CPF: _cpf_spans,
    DataCategory.CNPJ: _cnpj_spans,
    DataCategory.PHONE: _spelled_phone_spans,
}
