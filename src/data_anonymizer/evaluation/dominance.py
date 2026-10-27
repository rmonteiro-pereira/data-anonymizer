"""Business-style disclosure: does a released number still reveal one contributor's size?

Statistical agencies protect company data with dominance rules. When one company
makes up most of a segment's total, publishing anything in real units about that
segment gives its size away. Two standard rules are used here:
- (1, k) dominance: the largest contributor holds more than k% of the segment total;
- p% rule: the second-largest contributor could estimate the largest one to within
  p% by subtracting its own value from the total.
Both only matter while the values are still in their original units.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Released values count as being in original units when their spread and centre
# are within this factor of the original ones.
_UNIT_FACTOR = 2.0


@dataclass(frozen=True, slots=True)
class DominantSegment:
    """A group whose total is dominated by one contributor."""

    group_column: str
    group: str
    largest_share: float
    rule: str


def keeps_original_units(original: pd.Series, released: pd.Series) -> bool:
    """True when released numbers are still on the original scale, give or take noise."""
    before = pd.to_numeric(original, errors="coerce").dropna()
    after = pd.to_numeric(released, errors="coerce").dropna()
    if before.empty or after.empty:
        return False
    before_spread = _spread(before)
    after_spread = _spread(after)
    if before_spread == 0 or after_spread == 0:
        return bool(np.isclose(before.median(), after.median()))
    ratio = after_spread / before_spread
    centre_moved = abs(after.median() - before.median()) / before_spread
    return 1 / _UNIT_FACTOR <= ratio <= _UNIT_FACTOR and centre_moved <= 1.0


def _spread(values: pd.Series) -> float:
    quartile_range = float(values.quantile(0.75) - values.quantile(0.25))
    return quartile_range or float(values.std(ddof=0))


def dominant_segments(
    values: pd.Series,
    groups: pd.Series,
    contributors: pd.Series,
    *,
    group_column: str,
    dominance_share: float = 0.5,
    p_percent: float = 0.1,
) -> list[DominantSegment]:
    """Return the groups where one contributor dominates the total of `values`.

    `contributors` says whose value each row is (a store, or the row itself), so a
    store with many monthly rows counts once with its full total.
    """
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce").abs(),
            "group": groups.astype(str),
            "contributor": contributors.astype(str),
        }
    ).dropna(subset=["value"])
    totals = frame.groupby(["group", "contributor"])["value"].sum()
    found = []
    for group, contributions in totals.groupby(level="group"):
        ranked = np.sort(contributions.to_numpy())[::-1]
        total = float(ranked.sum())
        if total == 0 or len(ranked) == 0:
            continue
        largest = float(ranked[0])
        second = float(ranked[1]) if len(ranked) > 1 else 0.0
        if largest / total > dominance_share:
            found.append(DominantSegment(group_column, str(group), largest / total, "dominance"))
        elif total - largest - second < p_percent * largest:
            found.append(DominantSegment(group_column, str(group), largest / total, "p% rule"))
    return found
