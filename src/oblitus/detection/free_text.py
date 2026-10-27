"""Tell prose that mentions personal data from columns that merely hold numbers.

A note such as "call 11 98765-4321 after 5pm" holds personal data. A column of order
numbers does not, yet some of its values pass a recognizer's check by chance. A recognizer
therefore counts only when it fires clearly more often than chance would explain.
"""

import numpy as np
import pandas as pd

from oblitus.detection.patterns import TEXT_RECOGNIZERS, ValueRecognizer
from oblitus.detection.text_names import has_names_in_prose
from oblitus.taxonomy import DataCategory

# Free text is flagged when at least this share of values contains personal data.
_EMBEDDED_MATCH_SHARE = 0.1

# One identifier per cell is a column of identifiers, not prose that mentions one.
_FREE_TEXT_MIN_WORDS = 2

# A recognizer counts when its hit rate is this many times its chance rate.
_CHANCE_MARGIN = 2.0

_BASE_CONFIDENCE = 0.5


def classify_free_text(sample: pd.Series) -> tuple[DataCategory, float, str] | None:
    """Return the free-text finding for a sample of texts, or None when it is not prose with PII."""
    texts = sample.astype(str)
    if texts.map(lambda text: len(text.split())).mean() < _FREE_TEXT_MIN_WORDS:
        return None

    share = float(_cells_with_personal_data(texts).mean())
    if share < _EMBEDDED_MATCH_SHARE:
        return None
    evidence = (
        f"{share:.0%} of sampled texts contain an email, document number, phone, name or city"
    )
    return DataCategory.FREE_TEXT_WITH_PII, min(1.0, _BASE_CONFIDENCE + share), evidence


def _cells_with_personal_data(texts: pd.Series) -> np.ndarray:
    found = np.zeros(len(texts), dtype=bool)
    for recognizer in TEXT_RECOGNIZERS:
        hits = _cells_with_hits(texts, recognizer)
        if hits.mean() > recognizer.chance_rate * _CHANCE_MARGIN:
            found |= hits
    return found | _cells_with_names(texts)


def _cells_with_names(texts: pd.Series) -> np.ndarray:
    return np.fromiter(map(has_names_in_prose, texts), dtype=bool, count=len(texts))


def _cells_with_hits(texts: pd.Series, recognizer: ValueRecognizer) -> np.ndarray:
    return np.fromiter(
        (bool(recognizer.find_spans(text)) for text in texts), dtype=bool, count=len(texts)
    )
