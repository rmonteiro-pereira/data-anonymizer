"""Disclosure risk and information loss for a numeric column released with noise.

These are the measures statistical disclosure control uses to tune perturbation:
- linkage rate: the share of released values whose nearest original value is the
  true one (distance-based record linkage). Pure chance gives 1 / rows.
- interval disclosure: the share of records whose true value lies within a band of
  `band_sd` standard deviations around the released value.
- per-record deviation: how far values moved, in standard deviations.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from oblitus.evaluation.utility import relative_change


@dataclass(frozen=True, slots=True)
class DeflectionMetrics:
    """How far a released column moved from the original, and how exposed it still is."""

    mean_change: float
    std_change: float
    ks_statistic: float
    wasserstein_iqr: float
    median_deviation_sd: float
    p95_deviation_sd: float
    linkage_rate: float
    linkage_baseline: float
    interval_disclosure: float


def deflection_metrics(
    original: np.ndarray, released: np.ndarray, *, band_sd: float = 0.1
) -> DeflectionMetrics:
    """Compare an original numeric column with its released version, row by row."""
    original = np.asarray(original, dtype=float)
    released = np.asarray(released, dtype=float)
    present = ~(np.isnan(original) | np.isnan(released))
    original = original[present]
    released = released[present]
    spread = float(original.std()) or 1.0
    quartiles = np.percentile(original, [25, 75])
    iqr = float(quartiles[1] - quartiles[0]) or spread
    deviation = np.abs(released - original) / spread
    return DeflectionMetrics(
        mean_change=relative_change(original.mean(), released.mean()),
        std_change=relative_change(original.std(), released.std()),
        ks_statistic=_ks_statistic(original, released),
        wasserstein_iqr=float(stats.wasserstein_distance(original, released)) / iqr,
        median_deviation_sd=float(np.median(deviation)),
        p95_deviation_sd=float(np.percentile(deviation, 95)),
        linkage_rate=linkage_rate(original, released),
        linkage_baseline=1 / len(original),
        interval_disclosure=float(np.mean(deviation <= band_sd)),
    )


def linkage_rate(original: np.ndarray, released: np.ndarray) -> float:
    """Share of released values whose nearest original value belongs to the same record.

    When several originals are equally near, the attacker picks one at random, so a
    correct match among them counts as one over their number.
    """
    original = np.asarray(original, dtype=float)
    released = np.asarray(released, dtype=float)
    ordered = np.sort(original)
    positions = np.searchsorted(ordered, released)
    below = ordered[np.clip(positions - 1, 0, len(ordered) - 1)]
    above = ordered[np.clip(positions, 0, len(ordered) - 1)]
    nearest = np.minimum(np.abs(released - below), np.abs(released - above))
    tolerance = 1e-9 * (1 + np.abs(released))
    low = np.searchsorted(ordered, released - nearest - tolerance, side="left")
    high = np.searchsorted(ordered, released + nearest + tolerance, side="right")
    tied = np.maximum(high - low, 1)
    own_is_nearest = np.abs(original - released) <= nearest + tolerance
    return float(np.mean(own_is_nearest / tied))


def joint_linkage_rate(original: np.ndarray, released: np.ndarray) -> float:
    """Share of released records whose nearest original record, over all columns, is theirs.

    Both arguments hold one row per record and one column per released number. The columns
    are standardized by the original's spread first, so no column dominates the distance.
    Ties count as in `linkage_rate`. Pure chance gives 1 / rows.
    """
    original = np.asarray(original, dtype=float)
    released = np.asarray(released, dtype=float)
    spread = original.std(axis=0)
    spread[spread == 0] = 1.0
    centre = original.mean(axis=0)
    scaled_original = (original - centre) / spread
    scaled_released = (released - centre) / spread
    tree = cKDTree(scaled_original)
    nearest = tree.query(scaled_released)[0]
    reach = nearest + 1e-9 * (1 + nearest)
    tied = tree.query_ball_point(scaled_released, r=reach, return_length=True)
    own_distance = np.linalg.norm(scaled_original - scaled_released, axis=1)
    return float(np.mean((own_distance <= reach) / np.maximum(tied, 1)))


def _ks_statistic(original: np.ndarray, released: np.ndarray) -> float:
    # Large samples fall back from the exact p-value with a warning; only the
    # statistic is used here, so the asymptotic method is chosen up front.
    return float(stats.ks_2samp(original, released, method="asymp").statistic)
