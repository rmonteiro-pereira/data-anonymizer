"""Copies of the training records that stand out from all the others.

The whole-table tests average over every record, so a generator that appends a handful of rare
training records to thousands of fresh rows passes them, while anyone who knows one of those
people can tell they were in the data. A record far from every other record of its table is such
a person. Fresh draws from the population almost never reproduce one exactly, so a synthetic row
equal to an isolated training record is a copy.

A record is isolated when its distance to the nearest other record of its table is among the
largest `ISOLATED_SHARE` of the table and is not zero. The same is counted for the isolated
holdout records, whose isolation is measured the same way inside the holdout, so the two
counts are exchangeable when the generator never saw the training table.

How many copies are tolerated depends on the data:
- When no row of either real table repeats, chance does not reproduce a row, so any excess of
  copies of training records over holdout records is a finding.
- When some rows repeat (categories, rounded or clipped numbers), chance matches happen. The
  excess then counts only when a one-sided sign test on the two counts rejects chance at
  `SIGN_TEST_LEVEL`.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

# Share of the records that count as isolated: those with the largest distance to the nearest
# other record of their own table.
ISOLATED_SHARE = 0.10

# Level of the sign test used when rows repeat.
SIGN_TEST_LEVEL = 0.01


@dataclass(frozen=True, slots=True)
class IsolatedRecordCopies:
    """How many isolated training records, and as many isolated holdout records, are copied."""

    cohort_size: int
    train_copies: int
    holdout_copies: int | None
    rows_repeat: bool

    @property
    def is_excessive(self) -> bool:
        """True when training records are copied more often than chance explains."""
        holdout_copies = self.holdout_copies or 0
        excess = self.train_copies - holdout_copies
        if excess <= 0:
            return False
        if self.holdout_copies is None or not self.rows_repeat:
            return True
        sign_test = stats.binomtest(
            self.train_copies, self.train_copies + holdout_copies, 0.5, alternative="greater"
        )
        return bool(sign_test.pvalue <= SIGN_TEST_LEVEL)


def count_isolated_copies(
    train_isolation: np.ndarray,
    train_to_synthetic: np.ndarray,
    holdout_isolation: np.ndarray | None = None,
    holdout_to_synthetic: np.ndarray | None = None,
) -> IsolatedRecordCopies:
    """Count exact copies among the most isolated records.

    `*_isolation` is each record's distance to the nearest other record of its own table, and
    `*_to_synthetic` its distance to the nearest synthetic row. The holdout arrays are optional.
    """
    cohort_size = max(1, int(np.ceil(ISOLATED_SHARE * len(train_isolation))))
    train_copies = _copies_among_most_isolated(train_isolation, train_to_synthetic, cohort_size)
    rows_repeat = _has_repeated_rows(train_isolation)
    if holdout_isolation is None or holdout_to_synthetic is None:
        return IsolatedRecordCopies(cohort_size, train_copies, None, rows_repeat)
    holdout_copies = _copies_among_most_isolated(
        holdout_isolation, holdout_to_synthetic, cohort_size
    )
    rows_repeat = rows_repeat or _has_repeated_rows(holdout_isolation)
    return IsolatedRecordCopies(cohort_size, train_copies, holdout_copies, rows_repeat)


def _has_repeated_rows(isolation: np.ndarray) -> bool:
    return bool(np.any(isolation == 0))


def _copies_among_most_isolated(
    isolation: np.ndarray, distance_to_synthetic: np.ndarray, cohort_size: int
) -> int:
    most_isolated = np.argsort(-isolation, kind="stable")[:cohort_size]
    # A record with an identical twin in its table has isolation 0 and is not rare.
    is_copied = (isolation[most_isolated] > 0) & (distance_to_synthetic[most_isolated] == 0)
    return int(is_copied.sum())
