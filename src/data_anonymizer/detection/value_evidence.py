"""Score how well a column's values fit each kind of personal data, and pick the best fit.

Every recognizer is tried on the sample, so the decision can see runners-up: eleven
digits can be a CPF, a CNH or a phone, and a near tie means the values alone cannot
tell which. Confidence follows three tiers, highest first: values pass a checksum
and the name agrees; values pass a checksum; values only match a format.
"""

from dataclasses import dataclass

import pandas as pd

from data_anonymizer.detection.patterns import RECOGNIZERS, ValueRecognizer
from data_anonymizer.taxonomy import DataCategory

# Share of sampled values that must match for the whole column to count.
_CHECKED_SHARE = 0.8
# Formats without a checksum need more support, since ordinary codes can share them.
_FORMAT_ONLY_SHARE = 0.9
_FORMAT_ONLY_DISCOUNT = 0.8
_NAME_BONUS = 0.15
# Runners-up this close to the winner make the column ambiguous.
_TIE_MARGIN = 0.05
_AMBIGUITY_PENALTY = 0.25


@dataclass(frozen=True, slots=True)
class Candidate:
    """One kind of personal data and the share of sampled values that fit it."""

    recognizer: ValueRecognizer
    share: float
    name_agrees: bool

    @property
    def passes(self) -> bool:
        """True when enough values fit for the column to be this kind."""
        needed = _CHECKED_SHARE if self.recognizer.validator else _FORMAT_ONLY_SHARE
        return self.share >= needed

    @property
    def confidence(self) -> float:
        """The confidence tier this candidate reaches."""
        base = self.share if self.recognizer.validator else self.share * _FORMAT_ONLY_DISCOUNT
        if self.name_agrees:
            base += _NAME_BONUS
        return min(1.0, base)


def value_candidates(
    sample: pd.Series,
    *,
    name_category: DataCategory | None,
    integers: bool,
    ignore_bare_digits: bool,
) -> list[Candidate]:
    """Return every kind with at least one matching value, best fit first.

    `integers` restores leading zeros that integer storage dropped. With
    `ignore_bare_digits`, values made only of digits never count as a match.
    """
    texts = sample.astype(str).str.strip()
    if integers:
        texts = texts.str.replace(r"\.0$", "", regex=True)
    if ignore_bare_digits:
        texts = texts[~texts.str.fullmatch(r"\d+")]
    if texts.empty:
        return []

    best_by_category: dict[DataCategory, Candidate] = {}
    for recognizer in RECOGNIZERS:
        if recognizer.needs_name_hint and recognizer.category != name_category:
            continue
        matcher = recognizer.matches_padded if integers else recognizer.matches
        share = float(texts.map(matcher).sum()) / len(sample)
        current = best_by_category.get(recognizer.category)
        if share > 0 and (current is None or share > current.share):
            candidate = Candidate(recognizer, share, recognizer.category == name_category)
            best_by_category[recognizer.category] = candidate

    # Python's sort is stable, so recognizer order breaks exact ties.
    return sorted(best_by_category.values(), key=lambda item: -item.confidence)


def decide_from_values(candidates: list[Candidate]) -> tuple[DataCategory, float, str] | None:
    """Return the category the values support, its confidence and the evidence, if any."""
    passing = [candidate for candidate in candidates if candidate.passes]
    if not passing:
        return None

    best = passing[0]
    evidence = f"{best.share:.0%} of sampled values are valid {best.recognizer.label} values"
    if best.name_agrees:
        evidence += " and the column name agrees"
    confidence = best.confidence

    rivals = [
        candidate
        for candidate in passing[1:]
        if best.confidence - candidate.confidence < _TIE_MARGIN
    ]
    if rivals and not best.name_agrees:
        labels = ", ".join(candidate.recognizer.label for candidate in rivals)
        evidence += f"; they also fit {labels}, so a clearer column name would settle it"
        confidence = max(0.0, confidence - _AMBIGUITY_PENALTY)
    return best.recognizer.category, confidence, evidence
