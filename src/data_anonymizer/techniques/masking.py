"""Techniques that hide values, wholly or partly, without replacing them with look-alikes."""

import pandas as pd

from data_anonymizer.detection.patterns import ValueRecognizer, recognizers_for
from data_anonymizer.taxonomy import DataCategory
from data_anonymizer.techniques.context import TechniqueContext
from data_anonymizer.techniques.value_mapping import map_present


def keep(values: pd.Series, context: TechniqueContext) -> pd.Series:
    """Leave the column unchanged; used to record a reviewed decision in a plan."""
    return values.copy()


def redact(
    values: pd.Series, context: TechniqueContext, *, replacement: str = "[REDACTED]"
) -> pd.Series:
    """Replace every present value with one constant string."""
    return values.astype(object).where(values.isna(), replacement)


def suppress(values: pd.Series, context: TechniqueContext) -> pd.Series:
    """Replace every value with a missing value."""
    return pd.Series([None] * len(values), index=values.index, dtype=object, name=values.name)


def partial_mask(
    values: pd.Series,
    context: TechniqueContext,
    *,
    keep_first: int = 0,
    keep_last: int = 4,
    mask_char: str = "*",
) -> pd.Series:
    """Hide the letters and digits of each value except the first and last few.

    Separators such as dots, dashes and spaces stay visible, so `529.982.247-25`
    becomes `***.***.**7-25` and keeps its familiar shape.
    """
    return map_present(values, lambda text: _mask_text(text, keep_first, keep_last, mask_char))


def _mask_text(text: str, keep_first: int, keep_last: int, mask_char: str) -> str:
    positions = [index for index, character in enumerate(text) if character.isalnum()]
    total = len(positions)
    visible: set[int] = set()
    # When the visible parts would cover everything, hide everything instead.
    if keep_first + keep_last < total:
        visible.update(positions[:keep_first])
        visible.update(positions[total - keep_last :])
    masked = [
        character if not character.isalnum() or index in visible else mask_char
        for index, character in enumerate(text)
    ]
    return "".join(masked)


def redact_entities(
    values: pd.Series,
    context: TechniqueContext,
    *,
    categories: list[str] | None = None,
) -> pd.Series:
    """Replace personal data found inside free text with labels such as `<EMAIL>`."""
    wanted = None if categories is None else [DataCategory(item) for item in categories]
    recognizers = recognizers_for(wanted)
    return map_present(values, lambda text: _redact_text(text, recognizers))


def _redact_text(text: str, recognizers: tuple[ValueRecognizer, ...]) -> str:
    for recognizer in recognizers:
        label = f"<{recognizer.category.value.upper()}>"
        # Replacing from the end keeps the earlier spans' offsets valid.
        for start, end in reversed(recognizer.find_spans(text)):
            text = text[:start] + label + text[end:]
    return text
