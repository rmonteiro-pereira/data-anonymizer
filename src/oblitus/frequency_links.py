"""Find columns of two released tables that code the same entities under different names.

`apply_many` links tables through a column with the same name and consistent codes. An
attacker needs no shared name or shared codes when each entity occurs a distinctive number of
times in both tables: the account seen in 57 rows of one table is the patient seen in 57 rows
of the other. Counting the rows per value in each column and pairing the values whose counts
are unique in both columns recovers the mapping.

Two columns count as the same entity under this heuristic when
- both hold at least `MIN_VALUES` distinct values,
- at least `MIN_SHARED_PROFILE` of their count vectors agree (equal counts, compared as
  multisets), and
- at least `MIN_PAIRED` of the values of the smaller column can be paired by a count that no
  other value of either column has.

Columns whose values all occur equally often, or a few times each, have no distinctive counts
and are never paired. The limit of the heuristic: it finds only entities that differ in how
often they occur, and it misses tables that were filtered, sampled or noised enough to change
the counts.
"""

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

MIN_VALUES = 5
MIN_SHARED_PROFILE = 0.8
MIN_PAIRED = 0.5


@dataclass(frozen=True, slots=True)
class FrequencyLink:
    """Two columns of two tables that code the same entities, and which value matches which.

    `pairing` maps a value of `right_column` to the value of `left_column` with the same count.
    """

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    pairing: dict[Hashable, Hashable]


def frequency_links(
    tables: Mapping[str, pd.DataFrame], candidates: Mapping[str, Sequence[str]]
) -> list[FrequencyLink]:
    """The columns of different tables that code the same entities by how often they occur.

    `candidates` lists, per table, the columns worth comparing.
    """
    names = list(tables)
    links = []
    for position, left_table in enumerate(names):
        for right_table in names[position + 1 :]:
            for left_column in candidates.get(left_table, []):
                for right_column in candidates.get(right_table, []):
                    if left_column == right_column:
                        continue
                    pairing = pair_by_frequency(
                        tables[left_table][left_column], tables[right_table][right_column]
                    )
                    if pairing:
                        links.append(
                            FrequencyLink(
                                left_table, left_column, right_table, right_column, pairing
                            )
                        )
    return links


def pair_by_frequency(left: pd.Series, right: pd.Series) -> dict[Hashable, Hashable]:
    """Map each value of `right` to the value of `left` with the same unique count, or {}."""
    left_counts = left.dropna().value_counts()
    right_counts = right.dropna().value_counts()
    smaller = min(len(left_counts), len(right_counts))
    if smaller < MIN_VALUES:
        return {}
    shared = Counter(left_counts.tolist()) & Counter(right_counts.tolist())
    if sum(shared.values()) < MIN_SHARED_PROFILE * max(len(left_counts), len(right_counts)):
        return {}
    left_by_count = _values_by_unique_count(left_counts)
    right_by_count = _values_by_unique_count(right_counts)
    pairing = {
        right_by_count[count]: left_by_count[count]
        for count in left_by_count.keys() & right_by_count.keys()
    }
    if len(pairing) < MIN_PAIRED * smaller:
        return {}
    return pairing


def _values_by_unique_count(counts: pd.Series) -> dict[int, Hashable]:
    """The value for every count that exactly one value has."""
    holders = counts.groupby(counts).size()
    unique = set(holders[holders == 1].index)
    return {int(count): value for value, count in counts.items() if count in unique}
