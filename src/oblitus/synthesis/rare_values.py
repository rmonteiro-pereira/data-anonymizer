"""Keep a single record from shaping what a learned distribution can emit.

A category seen once, or one extreme number, would otherwise reappear in the synthetic table
and show that a person was in the reference. Both are held back until enough records stand
behind them: rare categories are merged, and the outermost numbers are pulled in.
"""

import numpy as np
import pandas as pd

from oblitus.plan import Fidelity

# Records that must stand behind a value before statistical fidelity may emit it.
MIN_RECORDS_STATISTICAL = 5

# The stricter count for privacy fidelity, whose point is that no record can be singled out.
MIN_RECORDS_PRIVACY = 20

MERGED_CATEGORY = "OTHER"


def min_records_for(fidelity: Fidelity) -> int:
    """Records that must stand behind a category or a numeric tail under this fidelity."""
    if fidelity is Fidelity.PRIVACY:
        return MIN_RECORDS_PRIVACY
    return MIN_RECORDS_STATISTICAL


def merge_rare_categories(values: np.ndarray, minimum: int) -> np.ndarray:
    """Replace categories seen fewer than `minimum` times by `OTHER`.

    When the rare categories together are still fewer than `minimum` rows, a lone `OTHER`
    level would itself be rare, so they join the most common category instead.
    """
    counts = pd.Series(values).value_counts()
    rare = counts[counts < minimum]
    if rare.empty:
        return values
    replacement = MERGED_CATEGORY if int(rare.sum()) >= minimum else counts.index[0]
    return np.where(pd.Series(values).isin(rare.index), replacement, values).astype(object)


def pull_in_extremes(sorted_values: np.ndarray, minimum: int) -> np.ndarray:
    """Cap both ends of sorted numbers so at least `minimum` records share each end value.

    The largest values are lowered to the value that has `minimum` records above it, and the
    smallest are raised the same way, so no single record decides the range. A short column
    gives up fewer values: at most a tenth of its rows at each end.
    """
    count = len(sorted_values)
    pulled = min(minimum, count // 10)
    if pulled == 0:
        return sorted_values
    capped = sorted_values.copy()
    capped[:pulled] = sorted_values[pulled]
    capped[count - pulled :] = sorted_values[count - pulled - 1]
    return capped
