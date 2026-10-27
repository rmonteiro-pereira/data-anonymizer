"""Techniques that hide values, wholly or partly, without replacing them with look-alikes."""

import narwhals as nw

from oblitus.detection.patterns import ValueRecognizer, recognizers_for
from oblitus.detection.text_entities import find_entities
from oblitus.table import Table
from oblitus.taxonomy import DataCategory
from oblitus.techniques.context import TechniqueContext
from oblitus.techniques.value_mapping import lookup_distinct


def keep(table: Table, column: str, context: TechniqueContext) -> nw.Expr:
    """Leave the column unchanged; used to record a reviewed decision in a plan."""
    return nw.col(column)


def redact(
    table: Table, column: str, context: TechniqueContext, *, replacement: str = "[REDACTED]"
) -> nw.Expr:
    """Replace every present value with one constant string."""
    return nw.when(~nw.col(column).is_null()).then(nw.lit(replacement, dtype=nw.String))


def suppress(table: Table, column: str, context: TechniqueContext) -> nw.Expr:
    """Replace every value with a missing value."""
    return nw.lit(None, dtype=nw.String)


def partial_mask(
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    keep_first: int = 0,
    keep_last: int = 4,
    mask_char: str = "*",
) -> nw.Expr:
    """Hide the letters and digits of each value except the first and last few.

    Separators such as dots, dashes and spaces stay visible, so `529.982.247-25`
    becomes `***.***.**7-25` and keeps its familiar shape.
    """
    return lookup_distinct(
        table, column, lambda text: _mask_text(text, keep_first, keep_last, mask_char)
    )


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
    table: Table,
    column: str,
    context: TechniqueContext,
    *,
    categories: list[str] | None = None,
) -> nw.Expr:
    """Replace personal data found inside free text with labels such as `<EMAIL>`.

    The text is normalized before it is read, so zero-width characters, look-alike letters
    and digits, `[at]` for an at-sign, and document numbers split by spaces or punctuation
    are found and removed too. Names and other details no pattern recognizes stay in the
    text.
    """
    wanted = None if categories is None else [DataCategory(item) for item in categories]
    recognizers = recognizers_for(wanted)
    return lookup_distinct(table, column, lambda text: _redact_text(text, recognizers))


def _redact_text(text: str, recognizers: tuple[ValueRecognizer, ...]) -> str:
    # Replacing from the end keeps the earlier spans' offsets valid.
    for span in reversed(find_entities(text, recognizers)):
        label = f"<{span.category.value.upper()}>"
        text = text[: span.start] + label + text[span.end :]
    return text
