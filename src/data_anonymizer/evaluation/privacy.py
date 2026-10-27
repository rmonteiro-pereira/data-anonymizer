"""Measure how easily records in a table can be singled out.

k-anonymity: every combination of quasi-identifier values is shared by at least k
rows. l-diversity: inside each such group, the sensitive attribute takes at least
l different values, so belonging to a group does not reveal the attribute.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class PrivacyReport:
    """k-anonymity and l-diversity of a table over its quasi-identifiers."""

    quasi_identifiers: list[str]
    k_anonymity: int
    k_target: int
    rows_below_target: int
    share_below_target: float
    l_diversity: dict[str, int] = field(default_factory=dict)

    @property
    def meets_target(self) -> bool:
        """True when every quasi-identifier group has at least k_target rows."""
        return self.k_anonymity >= self.k_target


def assess_privacy(
    frame: pd.DataFrame,
    quasi_identifiers: Sequence[str],
    sensitive_attributes: Sequence[str] = (),
    *,
    k_target: int = 5,
) -> PrivacyReport:
    """Compute k-anonymity and l-diversity of a table."""
    columns = [column for column in quasi_identifiers if column in frame.columns]
    if not columns or frame.empty:
        return PrivacyReport(
            quasi_identifiers=columns,
            k_anonymity=len(frame),
            k_target=k_target,
            rows_below_target=0,
            share_below_target=0.0,
        )

    keys = frame[columns].astype(str)
    grouped = keys.assign(_row=1).groupby(columns, dropna=False)
    group_sizes = grouped["_row"].transform("size")
    below = int((group_sizes < k_target).sum())
    diversity = {
        attribute: _l_diversity(frame, keys, columns, attribute)
        for attribute in sensitive_attributes
        if attribute in frame.columns
    }
    return PrivacyReport(
        quasi_identifiers=columns,
        k_anonymity=int(group_sizes.min()),
        k_target=k_target,
        rows_below_target=below,
        share_below_target=below / len(frame),
        l_diversity=diversity,
    )


def _l_diversity(
    frame: pd.DataFrame, keys: pd.DataFrame, columns: list[str], attribute: str
) -> int:
    grouped = keys.assign(_attribute=frame[attribute].astype(str)).groupby(columns, dropna=False)
    return int(grouped["_attribute"].nunique().min())
