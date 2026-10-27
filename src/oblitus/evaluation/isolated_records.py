"""Copies of the training records that stand out from all the others.

The whole-table tests average over every record, so a generator that appends a handful of rare
training records to thousands of fresh rows passes them, while anyone who knows one of those
people can tell they were in the data. A record far from every other record of its table is such
a person.

A record is isolated when its distance to the nearest other record of its table is among the
largest `ISOLATED_SHARE` of the table and is not zero. A record with an identical twin is never
isolated, so repeated rows elsewhere in the table do not change what counts as a copy.

A synthetic row copies an isolated record when it is within `COPY_FRACTION` of the record's own
isolation distance on every column (a copy, exact or jittered), or on more than half of the
columns (a partial copy). A fresh draw from the population is almost never that close to a
record that is far from all its neighbours. The same is counted for the isolated holdout
records, whose isolation is measured inside the holdout, so the counts are exchangeable when
the generator never saw the training table.

How many copies are tolerated is measured, not assumed:
- When no real holdout row is that close to a training record either, chance does not produce
  such matches, and any excess of copies of training records over holdout records is a finding.
- When some are (categories, rounded or clipped numbers), chance matches happen. The excess
  then counts only when a one-sided sign test on the two counts rejects chance at
  `SIGN_TEST_LEVEL`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Share of the records that count as isolated: those with the largest distance to the nearest
# other record of their own table.
ISOLATED_SHARE = 0.10

# A synthetic row within this fraction of a record's isolation distance, column by column,
# matches the record.
COPY_FRACTION = 0.01

# Level of the sign test used when real rows also match training records by chance.
SIGN_TEST_LEVEL = 0.01


@dataclass(frozen=True, slots=True)
class RecordMatches:
    """Which records of one table a set of rows matches, and how isolated each record is.

    `isolation` is the distance to the nearest other record of the same table.
    `matched_columns` is the most columns on which one row is within the copy tolerance.
    """

    isolation: np.ndarray
    matched_columns: np.ndarray
    n_columns: int

    @property
    def matches_every_column(self) -> np.ndarray:
        return (self.isolation > 0) & (self.matched_columns == self.n_columns)

    @property
    def matches_most_columns(self) -> np.ndarray:
        # A record with an identical twin in its table has isolation 0 and cannot be a copy.
        return (self.isolation > 0) & (2 * self.matched_columns > self.n_columns)


@dataclass(frozen=True, slots=True)
class CopyCount:
    """Copies of the isolated training records by one rule, and what chance produces.

    `chance_matches` is how many training records real holdout rows match, which is the rate
    at which rows that never saw the training table come this close. It is None without a
    holdout.
    """

    train: int
    holdout: int | None
    chance_matches: int | None

    @property
    def is_excessive(self) -> bool:
        """True when training records are copied more often than chance explains."""
        holdout = self.holdout or 0
        if self.train <= holdout:
            return False
        if not self.chance_matches:
            return True
        sign_test = stats.binomtest(self.train, self.train + holdout, 0.5, alternative="greater")
        return bool(sign_test.pvalue <= SIGN_TEST_LEVEL)


@dataclass(frozen=True, slots=True)
class IsolatedRecordCopies:
    """How many isolated training records, and as many isolated holdout records, are copied."""

    cohort_size: int
    every_column: CopyCount
    most_columns: CopyCount

    @property
    def train_copies(self) -> int:
        return self.most_columns.train

    @property
    def holdout_copies(self) -> int | None:
        return self.most_columns.holdout

    @property
    def is_excessive(self) -> bool:
        return self.every_column.is_excessive or self.most_columns.is_excessive


def copy_tolerance(isolation: np.ndarray) -> np.ndarray:
    """The per-column distance within which a row matches each record."""
    return COPY_FRACTION * isolation


def count_isolated_copies(
    train: RecordMatches,
    holdout: RecordMatches | None = None,
    chance: RecordMatches | None = None,
) -> IsolatedRecordCopies:
    """Count the copied records among the most isolated ones of each table.

    `train` and `holdout` are matched against the synthetic rows. `chance` is the training
    table matched against the real holdout rows.
    """
    cohort_size = max(1, int(np.ceil(ISOLATED_SHARE * len(train.isolation))))
    return IsolatedRecordCopies(
        cohort_size,
        every_column=_copy_count("matches_every_column", cohort_size, train, holdout, chance),
        most_columns=_copy_count("matches_most_columns", cohort_size, train, holdout, chance),
    )


def _copy_count(
    rule: str,
    cohort_size: int,
    train: RecordMatches,
    holdout: RecordMatches | None,
    chance: RecordMatches | None,
) -> CopyCount:
    train_copies = _copies_among_most_isolated(train, rule, cohort_size)
    if holdout is None or chance is None:
        return CopyCount(train_copies, None, None)
    holdout_copies = _copies_among_most_isolated(holdout, rule, cohort_size)
    return CopyCount(train_copies, holdout_copies, int(getattr(chance, rule).sum()))


def _copies_among_most_isolated(matches: RecordMatches, rule: str, cohort_size: int) -> int:
    most_isolated = np.argsort(-matches.isolation, kind="stable")[:cohort_size]
    return int(getattr(matches, rule)[most_isolated].sum())
