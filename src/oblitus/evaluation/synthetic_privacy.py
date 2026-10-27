"""Check whether a synthetic table leaks the real records it was learned from.

Four distance-based tests, all built on each record's distance to its closest record in
another table (see `record_distance.py`):
- Exact copies: the share of synthetic rows that equal a training row.
- DCR share: the share of synthetic rows closer to the training table than to a real holdout
  table of the same size. The generator never saw the holdout, so a share near 0.5 means the
  synthetic data is no closer to the training rows than to any other real rows.
- Membership inference: an attacker holds candidate records, half from the training table and
  half from the holdout, and guesses "member" for those closest to the synthetic data. The
  attack's AUC is 0.5 when it learns nothing, and its advantage is 2 * AUC - 1.

- Isolated records: copies of the training records that stand out from all the others (see
  `isolated_records.py`). The first three average over the table, so a generator that appends a
  few rare records verbatim to many fresh rows passes them; this test looks at those records.

Without a holdout table, only the exact copies and the distance summaries are available.
Each test becomes a finding when it crosses a threshold, and the table passes when there are
no findings. The thresholds are documented in docs/synthesis.md.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from oblitus.errors import SchemaError
from oblitus.evaluation.isolated_records import (
    IsolatedRecordCopies,
    RecordMatches,
    copy_tolerance,
    count_isolated_copies,
)
from oblitus.evaluation.record_distance import (
    MAX_COMPARED_ROWS,
    EncodedColumns,
    NearestDistances,
    encode_tables,
    matched_column_counts,
    nearest_distances,
)

_BOOTSTRAP_DRAWS = 500


@dataclass(frozen=True, slots=True)
class SyntheticPrivacyThresholds:
    """When a measurement becomes a finding.

    `max_exact_copy_share` is the largest share of synthetic rows that may equal a training
    row, beyond the share that equals a holdout row. `max_dcr_share` is the largest share of
    synthetic rows that may be closer to the training table than to the holdout.
    `min_attack_advantage` is the smallest membership-attack advantage that counts: leak-free
    generators show advantages up to about 0.05 by chance, which an interval alone would flag.
    """

    max_exact_copy_share: float = 0.01
    max_dcr_share: float = 0.6
    min_attack_advantage: float = 0.1


@dataclass(frozen=True, slots=True)
class MembershipInference:
    """Outcome of the distance-based membership attack, with bootstrap intervals."""

    auc: float
    auc_ci_low: float
    auc_ci_high: float
    advantage: float
    advantage_ci_low: float
    advantage_ci_high: float
    n_candidates: int


@dataclass(frozen=True, slots=True)
class SyntheticPrivacyReport:
    """Distances from a synthetic table to the real records, and the verdict they give."""

    columns: list[str]
    n_synthetic: int
    n_train: int
    n_holdout: int
    confidence_level: float
    exact_copy_share: float
    median_distance_to_train: float
    median_nndr: float
    holdout_exact_copy_share: float | None = None
    median_distance_to_holdout: float | None = None
    dcr_share: float | None = None
    dcr_share_ci_low: float | None = None
    dcr_share_ci_high: float | None = None
    membership: MembershipInference | None = None
    isolated_records: IsolatedRecordCopies | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no test found a sign of memorization."""
        return not self.findings


def assess_synthetic_privacy(
    synthetic: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    *,
    thresholds: SyntheticPrivacyThresholds | None = None,
    max_rows: int = MAX_COMPARED_ROWS,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> SyntheticPrivacyReport:
    """Measure how close a synthetic table is to the real records it was learned from.

    `train` is the table the generator saw. `holdout` is real data it never saw, from the
    same population; give one to get the DCR share and the membership attack. Only columns
    present in every table are compared. Each table is cut to at most `max_rows` random rows,
    and train and holdout are cut to the same size so their distances are comparable.
    """
    rules = thresholds or SyntheticPrivacyThresholds()
    rng = np.random.default_rng(seed)
    columns = _shared_columns(synthetic, train, holdout)
    synthetic_rows = _cut_to(synthetic[columns], max_rows, rng)
    train_rows, holdout_rows = _matched_samples(train, holdout, columns, max_rows, rng)
    others = [synthetic_rows] if holdout_rows is None else [synthetic_rows, holdout_rows]
    encoded = encode_tables(train_rows, others, columns)
    to_train = nearest_distances(encoded[1], encoded[0])
    report = SyntheticPrivacyReport(
        columns=columns,
        n_synthetic=len(synthetic_rows),
        n_train=len(train_rows),
        n_holdout=0 if holdout_rows is None else len(holdout_rows),
        confidence_level=confidence_level,
        exact_copy_share=float(np.mean(to_train.nearest == 0)),
        median_distance_to_train=float(np.median(to_train.nearest)),
        median_nndr=float(np.median(_nndr(to_train))),
    )
    if holdout_rows is not None:
        train_encoded, synthetic_encoded, holdout_encoded = encoded
        to_holdout = nearest_distances(synthetic_encoded, holdout_encoded)
        report = _with_dcr_share(report, to_train, to_holdout, rng)
        member_distance = nearest_distances(train_encoded, synthetic_encoded).nearest
        outsider_distance = nearest_distances(holdout_encoded, synthetic_encoded).nearest
        attack = _membership_attack(member_distance, outsider_distance, rng, confidence_level)
        report = replace(report, membership=attack)
    report = replace(report, isolated_records=_isolated_record_copies(encoded, holdout_rows))
    return replace(report, findings=_findings(report, rules))


def _isolated_record_copies(
    encoded: list[EncodedColumns], holdout_rows: pd.DataFrame | None
) -> IsolatedRecordCopies | None:
    """Count copies of the isolated training records, and of isolated holdout records."""
    train_encoded, synthetic_encoded = encoded[0], encoded[1]
    train_rows = len(train_encoded.numbers) or len(train_encoded.categories)
    if train_rows < 2:
        return None
    train = _record_matches(train_encoded, synthetic_encoded)
    if holdout_rows is None:
        return count_isolated_copies(train)
    holdout_encoded = encoded[2]
    holdout = _record_matches(holdout_encoded, synthetic_encoded)
    chance = _record_matches(train_encoded, holdout_encoded)
    return count_isolated_copies(train, holdout, chance)


def _record_matches(real: EncodedColumns, rows: EncodedColumns) -> RecordMatches:
    isolation = nearest_distances(real, real).second_nearest
    matched_columns = matched_column_counts(real, rows, copy_tolerance(isolation))
    return RecordMatches(isolation, matched_columns, real.n_columns)


def _shared_columns(
    synthetic: pd.DataFrame, train: pd.DataFrame, holdout: pd.DataFrame | None
) -> list[str]:
    columns = [name for name in train.columns if name in synthetic.columns]
    if holdout is not None:
        columns = [name for name in columns if name in holdout.columns]
    if not columns:
        raise SchemaError("the synthetic and real tables share no column to compare")
    tables = [synthetic, train] + ([] if holdout is None else [holdout])
    if any(len(table) == 0 for table in tables):
        raise SchemaError("every table needs at least one row to measure distances")
    return columns


def _cut_to(table: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(table) <= max_rows:
        return table.reset_index(drop=True)
    chosen = np.sort(rng.choice(len(table), size=max_rows, replace=False))
    return table.iloc[chosen].reset_index(drop=True)


def _matched_samples(
    train: pd.DataFrame,
    holdout: pd.DataFrame | None,
    columns: list[str],
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cut train and holdout to one size, because a larger table is closer to everything."""
    if holdout is None:
        return _cut_to(train[columns], max_rows, rng), None
    size = min(len(train), len(holdout), max_rows)
    return _cut_to(train[columns], size, rng), _cut_to(holdout[columns], size, rng)


def _nndr(distances: NearestDistances) -> np.ndarray:
    """Nearest over second-nearest distance; 1 where both are zero, as no row stands out."""
    ratio = np.ones(len(distances.nearest))
    comparable = distances.second_nearest > 0
    ratio[comparable] = distances.nearest[comparable] / distances.second_nearest[comparable]
    return ratio


def _with_dcr_share(
    report: SyntheticPrivacyReport,
    to_train: NearestDistances,
    to_holdout: NearestDistances,
    rng: np.random.Generator,
) -> SyntheticPrivacyReport:
    """Add the share of synthetic rows closer to train than to holdout, with its interval."""
    scores = np.full(len(to_train.nearest), 0.5)
    scores[to_train.nearest < to_holdout.nearest] = 1.0
    scores[to_train.nearest > to_holdout.nearest] = 0.0
    picks = rng.integers(0, len(scores), size=(_BOOTSTRAP_DRAWS, len(scores)))
    low, high = _percentile_interval(scores[picks].mean(axis=1), report.confidence_level)
    return replace(
        report,
        holdout_exact_copy_share=float(np.mean(to_holdout.nearest == 0)),
        median_distance_to_holdout=float(np.median(to_holdout.nearest)),
        dcr_share=float(scores.mean()),
        dcr_share_ci_low=low,
        dcr_share_ci_high=high,
    )


def _membership_attack(
    member_distance: np.ndarray,
    outsider_distance: np.ndarray,
    rng: np.random.Generator,
    confidence_level: float,
) -> MembershipInference:
    """Guess "member" for candidates close to the synthetic data, and score the guesses."""
    auc = _auc(-member_distance, -outsider_distance)
    draws = np.empty(_BOOTSTRAP_DRAWS)
    for draw in range(_BOOTSTRAP_DRAWS):
        members = rng.choice(member_distance, size=len(member_distance), replace=True)
        outsiders = rng.choice(outsider_distance, size=len(outsider_distance), replace=True)
        draws[draw] = _auc(-members, -outsiders)
    auc_low, auc_high = _percentile_interval(draws, confidence_level)
    return MembershipInference(
        auc=auc,
        auc_ci_low=auc_low,
        auc_ci_high=auc_high,
        advantage=2 * auc - 1,
        advantage_ci_low=2 * auc_low - 1,
        advantage_ci_high=2 * auc_high - 1,
        n_candidates=len(member_distance) + len(outsider_distance),
    )


def _auc(member_scores: np.ndarray, outsider_scores: np.ndarray) -> float:
    """Chance that a member scores above an outsider, counting ties as half."""
    ranks = stats.rankdata(np.concatenate([member_scores, outsider_scores]))
    n_members = len(member_scores)
    n_outsiders = len(outsider_scores)
    rank_sum = ranks[:n_members].sum()
    return float((rank_sum - n_members * (n_members + 1) / 2) / (n_members * n_outsiders))


def _percentile_interval(draws: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail = (1 - confidence_level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return float(low), float(high)


def _findings(report: SyntheticPrivacyReport, rules: SyntheticPrivacyThresholds) -> list[str]:
    findings = _exact_copy_findings(report, rules)
    findings += _dcr_share_findings(report, rules)
    findings += _membership_findings(report, rules)
    findings += _isolated_record_findings(report)
    return findings


def _exact_copy_findings(
    report: SyntheticPrivacyReport, rules: SyntheticPrivacyThresholds
) -> list[str]:
    # Rows that repeat in the population also match the holdout, so only the excess over the
    # holdout's own matches counts as copying.
    natural_share = report.holdout_exact_copy_share or 0.0
    excess = report.exact_copy_share - natural_share
    if excess <= rules.max_exact_copy_share:
        return []
    return [
        f"{report.exact_copy_share:.1%} of synthetic rows are exact copies of training rows, "
        f"{excess:.1%} more than of holdout rows (limit {rules.max_exact_copy_share:.1%})"
    ]


def _dcr_share_findings(
    report: SyntheticPrivacyReport, rules: SyntheticPrivacyThresholds
) -> list[str]:
    if report.dcr_share is None or report.dcr_share_ci_low is None:
        return []
    if report.dcr_share > rules.max_dcr_share and report.dcr_share_ci_low > 0.5:
        return [
            f"{report.dcr_share:.1%} of synthetic rows are closer to the training table than "
            f"to the holdout (limit {rules.max_dcr_share:.1%}; 50% means no memorization)"
        ]
    return []


def _membership_findings(
    report: SyntheticPrivacyReport, rules: SyntheticPrivacyThresholds
) -> list[str]:
    attack = report.membership
    if attack is None or attack.advantage_ci_low <= 0:
        return []
    if attack.advantage < rules.min_attack_advantage:
        return []
    return [
        f"the membership attack has advantage {attack.advantage:.3f} "
        f"(interval {attack.advantage_ci_low:.3f} to {attack.advantage_ci_high:.3f}, "
        "which excludes 0, and is at least "
        f"{rules.min_attack_advantage:.2f})"
    ]


def _isolated_record_findings(report: SyntheticPrivacyReport) -> list[str]:
    isolated = report.isolated_records
    if isolated is None or not isolated.is_excessive:
        return []
    against = ""
    if isolated.holdout_copies is not None:
        against = f", against {isolated.holdout_copies} of the equally isolated holdout records"
    return [
        f"{isolated.train_copies} of the {isolated.cohort_size} most isolated training records "
        f"have a synthetic row matching them on most columns, {isolated.every_column.train} "
        f"on every column{against}"
    ]
