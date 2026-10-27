"""Find person names and Brazilian cities written inside prose, from the bundled IBGE lists.

Without a named-entity model, a note such as "Atendimento de Ana Silva em Santos" would keep
both the name and the city. These rules read the reference lists instead:
- a person is a capitalized first name from the list followed by one or more capitalized words,
  which may be joined by particles ("Maria da Silva");
- a city is a capitalized municipality name of two or more words ("Rio de Janeiro"), or a
  one-word name after "em" or "para" or after "cidade de" ("em Santos"). One-word names that
  are also everyday nouns ("Saúde", "Esperança") are ignored, so "Plano em Saúde" stays text.

Names written in lowercase are found under stricter rules, because lowercase words are common:
a first name from the list must be followed by a surname from the surname list ("ana silva"),
and a municipality of two or more words must follow "em", "para", "na", "no" or "cidade de".

A first name alone is not a person, and a capitalized word alone is not a surname, which keeps
prose such as "Rose Garden" or "Planejamento Estratégico" from matching much.
"""

import re
from dataclasses import dataclass

from oblitus.detection.reference_lists import (
    first_names,
    municipalities,
    normalize_word,
    surnames,
)
from oblitus.taxonomy import DataCategory

TextSpan = tuple[int, int, DataCategory]

_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")

# Lowercase words that sit inside a name without being part of its meaning.
_NAME_PARTICLES = frozenset({"da", "de", "do", "das", "dos", "di", "du", "del"})
_CITY_PARTICLES = _NAME_PARTICLES | {"e"}

# Words after the first name, and words of a municipality, counted as one name or city.
_MAX_SURNAME_WORDS = 4
_MAX_CITY_WORDS = 5

# Listed as first names, but in prose they are days, months and honorifics.
_NOT_FIRST_NAMES = frozenset({"domingo", "março", "santa", "santo", "são"})

# Words besides the name, so that "Sra. Ana Silva" or "Rua Ana Silva" is not prose.
_MIN_WORDS_AROUND_A_NAME = 2

_CITY_PREPOSITIONS = frozenset({"em", "para"})

# Words that may precede a lowercase multi-word municipality, which has no capital to show it.
_LOWERCASE_CITY_PREPOSITIONS = frozenset({"em", "para", "na", "no"})
_CITY_NOUNS = frozenset({"cidade", "municipio"})

# Municipality names that are also everyday nouns or names of things, so as one word they
# would flag ordinary prose such as "Plano de Saúde" or "Cartão Ouro".
_COMMON_NOUN_MUNICIPALITIES = frozenset(
    normalize_word(name)
    for name in [
        "Saúde",
        "Esperança",
        "União",
        "Liberdade",
        "Progresso",
        "Paraíso",
        "Planalto",
        "Salto",
        "Bonito",
        "Alegria",
        "Aurora",
        "Glória",
        "Jardim",
        "Serra",
        "Barra",
        "Lagoa",
        "Passos",
        "Cruz",
        "Flores",
        "Estrela",
        "Ouro",
        "Prata",
        "Pedra",
        "Porto",
        "Cedro",
        "Pinheiro",
        "Palma",
        "Carmo",
        "Areia",
        "Nobres",
        "Centro",
        "Vitória",
        "Divino",
        "Marco",
        "Vera",
        "Messias",
    ]
)


@dataclass(frozen=True, slots=True)
class _Word:
    start: int
    end: int
    text: str

    @property
    def is_capitalized(self) -> bool:
        return self.text[0].isupper()

    @property
    def lowered(self) -> str:
        return self.text.lower()


def find_name_and_city_spans(text: str) -> list[TextSpan]:
    """Return the person names and cities in a text, without overlaps, in text order."""
    words = [_Word(match.start(), match.end(), match.group(0)) for match in _WORD.finditer(text)]
    people = _person_spans(text, words)
    cities = [
        span for span in _city_spans(text, words) if not _overlaps_any(span[0], span[1], people)
    ]
    return sorted(people + cities)


def has_names_in_prose(text: str) -> bool:
    """True when a name or city sits among other words, as in a note, not alone in a cell.

    A cell that holds only a name, perhaps with a title or a street word, is a column of
    names or addresses and is classified as such; only prose that merely mentions one counts.
    """
    words = [_Word(match.start(), match.end(), match.group(0)) for match in _WORD.finditer(text)]
    spans = find_name_and_city_spans(text)
    outside = [word for word in words if not _overlaps_any(word.start, word.end, spans)]
    return bool(spans) and len(outside) >= _MIN_WORDS_AROUND_A_NAME


def _person_spans(text: str, words: list[_Word]) -> list[TextSpan]:
    spans: list[TextSpan] = []
    index = 0
    while index < len(words):
        last = _last_word_of_name_starting_at(text, words, index)
        if last is None:
            index += 1
            continue
        spans.append((words[index].start, words[last].end, DataCategory.PERSON_NAME))
        index = last + 1
    return spans


def _last_word_of_name_starting_at(text: str, words: list[_Word], index: int) -> int | None:
    first = words[index]
    if first.lowered in _NOT_FIRST_NAMES or normalize_word(first.text) not in first_names():
        return None
    if not first.is_capitalized:
        return _last_word_of_lowercase_name(text, words, index)
    last = None
    position = index + 1
    while position < len(words) and position - index <= _MAX_SURNAME_WORDS:
        if not _follows_by_whitespace(text, words[position - 1], words[position]):
            break
        word = words[position]
        if word.is_capitalized:
            last = position
        elif word.lowered not in _NAME_PARTICLES:
            break
        position += 1
    return last


def _last_word_of_lowercase_name(text: str, words: list[_Word], index: int) -> int | None:
    """End of a lowercase name: a listed first name, then listed names, ending on a surname."""
    last = None
    position = index + 1
    while position < len(words) and position - index <= _MAX_SURNAME_WORDS:
        word = words[position]
        if not _follows_by_whitespace(text, words[position - 1], word) or word.is_capitalized:
            break
        normalized = normalize_word(word.text)
        if normalized in surnames():
            last = position
        elif word.lowered not in _NAME_PARTICLES and normalized not in first_names():
            break
        position += 1
    return last


def _city_spans(text: str, words: list[_Word]) -> list[TextSpan]:
    spans: list[TextSpan] = []
    index = 0
    while index < len(words):
        last = _last_word_of_city_starting_at(text, words, index)
        if last is None:
            index += 1
            continue
        spans.append((words[index].start, words[last].end, DataCategory.CITY))
        index = last + 1
    return spans


def _last_word_of_city_starting_at(text: str, words: list[_Word], index: int) -> int | None:
    if not words[index].is_capitalized:
        return _last_word_of_lowercase_city(text, words, index)
    longest = min(len(words), index + _MAX_CITY_WORDS) - 1
    for last in range(longest, index - 1, -1):
        if not words[last].is_capitalized or not _joined_by_whitespace(text, words, index, last):
            continue
        name = normalize_word(text[words[index].start : words[last].end])
        if name in municipalities() and _may_be_a_city(words, index, last, name):
            return last
    return None


def _last_word_of_lowercase_city(text: str, words: list[_Word], index: int) -> int | None:
    """End of a lowercase municipality of two or more words after a preposition."""
    if index == 0 or not _follows_a_city_preposition(words, index):
        return None
    longest = min(len(words), index + _MAX_CITY_WORDS) - 1
    for last in range(longest, index, -1):
        if any(word.is_capitalized for word in words[index : last + 1]):
            continue
        if not _joined_by_whitespace(text, words, index, last):
            continue
        name = normalize_word(text[words[index].start : words[last].end])
        if name in municipalities():
            return last
    return None


def _follows_a_city_preposition(words: list[_Word], index: int) -> bool:
    previous = words[index - 1].lowered
    if previous in _LOWERCASE_CITY_PREPOSITIONS:
        return True
    return previous == "de" and index >= 2 and normalize_word(words[index - 2].text) in _CITY_NOUNS


def _may_be_a_city(words: list[_Word], first: int, last: int, name: str) -> bool:
    if last > first:
        return True
    if name in _COMMON_NOUN_MUNICIPALITIES or first == 0:
        return False
    previous = words[first - 1].lowered
    if previous in _CITY_PREPOSITIONS:
        return True
    return previous == "de" and first >= 2 and normalize_word(words[first - 2].text) in _CITY_NOUNS


def _joined_by_whitespace(text: str, words: list[_Word], first: int, last: int) -> bool:
    for position in range(first, last):
        if not _follows_by_whitespace(text, words[position], words[position + 1]):
            return False
    inner = words[first + 1 : last]
    return all(word.is_capitalized or word.lowered in _CITY_PARTICLES for word in inner)


def _follows_by_whitespace(text: str, before: _Word, after: _Word) -> bool:
    """True when only spaces, tabs or other whitespace separate the two words.

    Exports often double the space between a first name and a surname, so any run counts.
    """
    gap = text[before.end : after.start]
    return bool(gap) and gap.isspace()


def _overlaps_any(start: int, end: int, spans: list[TextSpan]) -> bool:
    return any(start < other_end and other_start < end for other_start, other_end, _ in spans)
