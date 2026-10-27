"""Find the text cells that stand for "no value", so they do not count as evidence.

Spreadsheets and CSV exports mark missing data with blanks, dashes, "N/D" or a CPF of
zeros. If those cells were kept, a column of valid CPFs with a fifth of them blank would
look like nothing in particular.
"""

import pandas as pd

_PLACEHOLDERS = frozenset(
    {
        "",
        "-",
        "--",
        "---",
        "n/d",
        "na",
        "n/a",
        "null",
        "none",
        "nan",
        "?",
        "sem informação",
        "sem informacao",
    }
)

# Six or more of the same digit, with any separators: 000.000.000-00, 99999999999.
_REPEATED_DIGIT = r"\W*(\d)(?:\W*\1){5,}\W*"


def blank_out_placeholders(values: pd.Series) -> pd.Series:
    """Return the column with placeholder text replaced by missing values.

    Numbers, booleans and dates are returned unchanged, since they have no text placeholders.
    """
    if _is_stored_as_numbers_or_dates(values):
        return values
    text = values.astype(str).str.strip()
    is_placeholder = text.str.casefold().isin(_PLACEHOLDERS)
    is_placeholder |= text.str.fullmatch(_REPEATED_DIGIT)
    return values.mask(is_placeholder)


def _is_stored_as_numbers_or_dates(values: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(values)
        or pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_datetime64_any_dtype(values)
    )
