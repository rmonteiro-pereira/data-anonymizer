"""Score how well a column's values fit each kind of personal data, and pick the best fit.

Every recognizer is tried on the sample, so the decision can see runners-up: eleven
digits can be a CPF, a CNH or a phone, and a near tie means the values alone cannot
tell which. Confidence follows three tiers, highest first: values pass a checksum
and the name agrees; values pass a checksum; values only match a format.

A recognizer with a strong check (CPF, CNPJ, email and the like) is rarely right by
accident, so half of the values passing is already evidence. That evidence is reported
below the review threshold, because the other half is wrong, mistyped or a different kind.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_anonymizer.detection.patterns import RECOGNIZERS, ValueRecognizer
from data_anonymizer.taxonomy import DataCategory, Sensitivity, sensitivity_of

# Share of sampled values that must match for the whole column to count.
_CHECKED_SHARE = 0.8
# Formats without a checksum need more support, since ordinary codes can share them.
_FORMAT_ONLY_SHARE = 0.9
_FORMAT_ONLY_DISCOUNT = 0.8
_NAME_BONUS = 0.15
# Strong recognizers pass at this share, but never above the partial confidence cap.
_STRONG_SHARE = 0.5
_PARTIAL_CONFIDENCE_CAP = 0.55
# A strong recognizer that matches more than this share is never reported as "nothing found".
_WEAK_MATCH_SHARE = 0.2
_WEAK_MATCH_CONFIDENCE = 0.4
# Identifier types that each cover at least this share of a column make it a mixed column.
_MIXED_MIN_SHARE = 0.1
# Runners-up this close to the winner make the column ambiguous.
_TIE_MARGIN = 0.05
_AMBIGUITY_PENALTY = 0.25


@dataclass(frozen=True, slots=True)
class Candidate:
    """One kind of personal data and the share of sampled values that fit it.

    For a column that mixes several identifier types, `recognizer` is the most common
    type, `share` is the share of values that fit any of them, and `mix` describes the
    split.
    """

    recognizer: ValueRecognizer
    share: float
    name_agrees: bool
    mix: str = ""

    @property
    def passes(self) -> bool:
        """True when enough values fit for the column to be this kind."""
        if self.recognizer.strong:
            return self.share >= _STRONG_SHARE
        return self.share >= self._full_share

    @property
    def is_partial(self) -> bool:
        """True when the candidate passes only because its recognizer is strong."""
        return self.recognizer.strong and self.share < self._full_share

    @property
    def confidence(self) -> float:
        """The confidence tier this candidate reaches."""
        base = self.share if self.recognizer.validator else self.share * _FORMAT_ONLY_DISCOUNT
        if self.name_agrees:
            base += _NAME_BONUS
        base = min(1.0, base)
        if self.is_partial:
            return min(base, _PARTIAL_CONFIDENCE_CAP)
        return base

    @property
    def _full_share(self) -> float:
        return _CHECKED_SHARE if self.recognizer.validator else _FORMAT_ONLY_SHARE


def value_candidates(
    sample: pd.Series,
    *,
    name_category: DataCategory | None,
    integers: bool,
    negative_hint: bool,
) -> list[Candidate]:
    """Return every kind with at least one matching value, best fit first.

    Digit-only values shorter than the identifier get their leading zeros back, because
    spreadsheets and integer storage drop them. `integers` also removes the `.0` that
    whole-number floats leave behind. With `negative_hint`, only recognizers with a strong
    check run, apart from the one the column name itself names.
    """
    texts = sample.astype(str).str.strip()
    if integers:
        texts = texts.str.replace(r"\.0$", "", regex=True)

    matches = [
        (recognizer, _match_flags(texts, recognizer))
        for recognizer in RECOGNIZERS
        if _may_run(recognizer, name_category, negative_hint)
    ]
    candidates = _best_per_category(matches, name_category)
    mixed = _mixed_candidate(matches, name_category)
    if mixed is not None:
        # A mixed column outranks a single type that ties with it, so the evidence names the mix.
        candidates.insert(0, mixed)
    # The sort is stable, so recognizer order breaks exact ties.
    return sorted(candidates, key=lambda item: (item.confidence, item.name_agrees), reverse=True)


def _may_run(
    recognizer: ValueRecognizer, name_category: DataCategory | None, negative_hint: bool
) -> bool:
    named = recognizer.category == name_category
    if recognizer.needs_name_hint and not named:
        return False
    return named or recognizer.strong or not negative_hint


def _match_flags(texts: pd.Series, recognizer: ValueRecognizer) -> np.ndarray:
    return np.fromiter(
        (recognizer.matches_padded(text) for text in texts), dtype=bool, count=len(texts)
    )


def _best_per_category(
    matches: list[tuple[ValueRecognizer, np.ndarray]], name_category: DataCategory | None
) -> list[Candidate]:
    best: dict[DataCategory, Candidate] = {}
    for recognizer, flags in matches:
        share = float(flags.mean())
        current = best.get(recognizer.category)
        if share > 0 and (current is None or share > current.share):
            best[recognizer.category] = Candidate(
                recognizer, share, recognizer.category == name_category
            )
    return list(best.values())


def _mixed_candidate(
    matches: list[tuple[ValueRecognizer, np.ndarray]], name_category: DataCategory | None
) -> Candidate | None:
    """Combine direct identifiers with strong checks when a column holds several types."""
    flags_by_category: dict[DataCategory, tuple[ValueRecognizer, np.ndarray]] = {}
    for recognizer, flags in matches:
        if not _is_strong_identifier(recognizer):
            continue
        current = flags_by_category.get(recognizer.category)
        if current is None or flags.mean() > current[1].mean():
            flags_by_category[recognizer.category] = (recognizer, flags)

    present = [item for item in flags_by_category.values() if item[1].mean() >= _MIXED_MIN_SHARE]
    if len(present) < 2:
        return None
    present.sort(key=lambda item: -item[1].mean())
    union = np.logical_or.reduce([flags for _, flags in present])
    leader, leader_flags = present[0]
    if union.mean() <= leader_flags.mean():
        return None

    split = ", ".join(f"{flags.mean():.0%} {recognizer.label}" for recognizer, flags in present)
    named = name_category in {recognizer.category for recognizer, _ in present}
    return Candidate(leader, float(union.mean()), named, mix=split)


def _is_strong_identifier(recognizer: ValueRecognizer) -> bool:
    direct = sensitivity_of(recognizer.category) is Sensitivity.DIRECT_IDENTIFIER
    return recognizer.strong and direct


def decide_from_values(candidates: list[Candidate]) -> tuple[DataCategory, float, str] | None:
    """Return the category the values support, its confidence and the evidence, if any."""
    passing = [candidate for candidate in candidates if candidate.passes]
    if not passing:
        return None

    best = passing[0]
    evidence = _evidence(best)
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


def _evidence(best: Candidate) -> str:
    if best.mix:
        evidence = (
            f"{best.share:.0%} of sampled values are valid identifiers of mixed types ({best.mix})"
        )
    else:
        evidence = f"{best.share:.0%} of sampled values are valid {best.recognizer.label} values"
    if best.name_agrees:
        evidence += " and the column name agrees"
    if best.is_partial:
        evidence += "; the rest are not, so check the column"
    return evidence


def decide_from_weak_match(
    candidates: list[Candidate],
) -> tuple[DataCategory, float, str] | None:
    """Report a strong recognizer that matched a notable share but too few values to pass."""
    strong = [
        candidate
        for candidate in candidates
        if candidate.recognizer.strong and candidate.share > _WEAK_MATCH_SHARE
    ]
    if not strong:
        return None
    best = max(strong, key=lambda candidate: candidate.share)
    label = best.recognizer.label
    evidence = (
        f"{best.share:.0%} of sampled values are valid {label} values, "
        f"too few to call the column {label}; please check it"
    )
    return best.recognizer.category, _WEAK_MATCH_CONFIDENCE, evidence
