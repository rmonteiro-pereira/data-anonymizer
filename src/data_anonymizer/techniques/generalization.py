"""Techniques that make values coarser so more records share each value."""

import numpy as np
import pandas as pd

from data_anonymizer.detection.kinds import as_datetime
from data_anonymizer.techniques.context import TechniqueContext

_PERIOD_CODES = {"year": "Y", "month": "M", "week": "W", "day": "D"}


def bin_numeric(
    values: pd.Series,
    context: TechniqueContext,
    *,
    bins: int = 5,
    width: float | None = None,
    strategy: str = "quantile",
) -> pd.Series:
    """Replace numbers with interval labels such as `[20, 30)`.

    With `width`, intervals have that fixed width (ages in decades). Otherwise the
    column is cut into `bins` intervals of equal size (`equal_width`) or with an
    equal number of rows each (`quantile`).
    """
    numbers = pd.to_numeric(values, errors="coerce")
    present = numbers.dropna()
    if present.empty:
        return values.astype(object).copy()

    if width is not None:
        lower = present.min() // width * width
        upper = (present.max() // width + 1) * width
        # Half a width past the top edge so floating-point steps never drop it.
        edges = np.arange(lower, upper + width / 2, width)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(present, np.linspace(0, 1, bins + 1)))
    elif strategy == "equal_width":
        edges = np.linspace(present.min(), present.max(), bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'equal_width'")

    if len(edges) < 2:
        # Every value is the same number, so a single closed interval holds them.
        return _single_interval(values, present.iloc[0])

    # Fixed-width bins are all right-open; data-driven bins close the last one on the maximum.
    labels = _interval_labels(edges, close_last=width is None)
    cut = pd.cut(numbers, bins=list(edges), labels=labels, right=False, include_lowest=True)
    # Values equal to the last edge fall outside right-open bins; they belong to the last.
    cut = cut.where(~(numbers == edges[-1]), labels[-1])
    return cut.astype(object).where(numbers.notna(), None)


def _interval_labels(edges: np.ndarray, *, close_last: bool) -> list[str]:
    labels = []
    for index in range(len(edges) - 1):
        is_last = index == len(edges) - 2
        closing = "]" if close_last and is_last else ")"
        labels.append(
            f"[{_format_number(edges[index])}, {_format_number(edges[index + 1])}{closing}"
        )
    return labels


def _single_interval(values: pd.Series, number: float) -> pd.Series:
    label = f"[{_format_number(number)}, {_format_number(number)}]"
    return values.astype(object).where(values.isna(), label)


def _format_number(number: float) -> str:
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.4g}"


def round_numeric(
    values: pd.Series,
    context: TechniqueContext,
    *,
    decimals: int = 0,
    base: float | None = None,
) -> pd.Series:
    """Round to a number of decimals, or to the nearest multiple of `base` (such as 1000)."""
    numbers = pd.to_numeric(values, errors="coerce")
    if base is not None:
        return (numbers / base).round() * base
    return numbers.round(decimals)


def truncate_date(values: pd.Series, context: TechniqueContext, *, unit: str = "year") -> pd.Series:
    """Cut dates down to the start of their year, month, week or day."""
    if unit not in _PERIOD_CODES:
        raise ValueError(f"unit must be one of {', '.join(_PERIOD_CODES)}")
    dates = as_datetime(values)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.to_period(_PERIOD_CODES[unit]).dt.start_time


def truncate_string(
    values: pd.Series,
    context: TechniqueContext,
    *,
    keep: int = 3,
    pad_char: str | None = None,
) -> pd.Series:
    """Keep the first characters, such as the region digits of a postal code.

    With `pad_char`, the removed characters are replaced so the length stays the same.
    """
    result = values.astype(object).copy()
    present = values.notna()
    texts = values[present].astype(str)
    if pad_char is None:
        result[present] = texts.str[:keep]
    else:
        result[present] = texts.map(lambda text: text[:keep] + pad_char * max(0, len(text) - keep))
    return result


def top_bottom_code(
    values: pd.Series,
    context: TechniqueContext,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.Series:
    """Clip extreme values, which single people out, to a lower and an upper limit."""
    numbers = pd.to_numeric(values, errors="coerce")
    lower_limit = float(numbers.quantile(lower_quantile)) if lower is None else lower
    upper_limit = float(numbers.quantile(upper_quantile)) if upper is None else upper
    clipped = numbers.clip(lower=lower_limit, upper=upper_limit)
    if pd.api.types.is_integer_dtype(values):
        return clipped.round().astype(values.dtype)
    return clipped


def suppress_rare(
    values: pd.Series,
    context: TechniqueContext,
    *,
    min_count: int = 5,
    replacement: str | None = None,
) -> pd.Series:
    """Replace categories seen fewer than `min_count` times, which identify few people."""
    counts = values.value_counts(dropna=True)
    rare = set(counts[counts < min_count].index)
    result = values.astype(object).copy()
    result[values.isin(rare)] = replacement
    return result
