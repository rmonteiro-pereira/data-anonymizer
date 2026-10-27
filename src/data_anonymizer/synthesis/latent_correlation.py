"""The correlation of the hidden normal variables behind columns, estimated pair by pair.

A Gaussian copula assumes every column is a monotone function of a normal variable.
Numbers and dates reveal that variable through their ranks. A category or boolean only
says which interval of it the row fell in, so its correlation with another column cannot
be read off the data directly: measuring it on scores placed inside the intervals shrinks
it, and a strong association between two yes/no columns drops to a weak one.

These estimators fit the hidden correlation by maximum likelihood instead:
- two numbers: the Pearson correlation of their normal scores;
- a category and a number (polyserial): the likelihood of each row's interval given the
  number's score;
- two categories (polychoric): the likelihood of the table of counts, with cell
  probabilities taken from the bivariate normal distribution.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

# Correlations are searched inside this range, because 1 and -1 make the likelihood singular.
_LIMIT = 0.995

# Normal values beyond this are treated as infinite.
_TAIL = 8.5

_QUADRATURE_POINTS = 40

# Tables with more cells than this use a cheaper moment estimate instead of the likelihood.
_MAX_TABLE_CELLS = 2500

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_QUADRATURE_POINTS)


@dataclass(frozen=True, slots=True)
class ContinuousScores:
    """Normal scores of a numeric or date column; NaN marks a missing cell."""

    scores: np.ndarray

    @property
    def present(self) -> np.ndarray:
        return ~np.isnan(self.scores)


@dataclass(frozen=True, slots=True)
class OrdinalIntervals:
    """Interval codes of a category or boolean column; -1 marks a missing cell.

    `edges` holds the normal values that split the intervals, from minus infinity to
    infinity, so code k means a value between edges[k] and edges[k + 1].
    """

    codes: np.ndarray
    edges: np.ndarray

    @property
    def present(self) -> np.ndarray:
        return self.codes >= 0


LatentColumn = ContinuousScores | OrdinalIntervals


def latent_correlation(first: LatentColumn, second: LatentColumn) -> float:
    """Estimate the correlation of the normal variables behind two columns."""
    present = first.present & second.present
    if present.sum() < 3:
        return 0.0
    if isinstance(first, ContinuousScores) and isinstance(second, ContinuousScores):
        return _score_correlation(first.scores[present], second.scores[present])
    if isinstance(first, OrdinalIntervals) and isinstance(second, OrdinalIntervals):
        return _polychoric(_restricted(first, present), _restricted(second, present))
    ordinal, continuous = (
        (first, second) if isinstance(first, OrdinalIntervals) else (second, first)
    )
    assert isinstance(ordinal, OrdinalIntervals) and isinstance(continuous, ContinuousScores)
    return _polyserial(ordinal.codes[present], ordinal.edges, continuous.scores[present])


def _restricted(column: OrdinalIntervals, present: np.ndarray) -> OrdinalIntervals:
    return OrdinalIntervals(column.codes[present], column.edges)


def _score_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _maximize(negative_log_likelihood) -> float:
    result = optimize.minimize_scalar(
        negative_log_likelihood,
        bounds=(-_LIMIT, _LIMIT),
        method="bounded",
        options={"xatol": 1e-3},
    )
    return float(result.x)


def _polyserial(codes: np.ndarray, edges: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(codes)) < 2 or scores.std() == 0:
        return 0.0
    lower = edges[codes]
    upper = edges[codes + 1]

    def negative_log_likelihood(correlation: float) -> float:
        spread = np.sqrt(1 - correlation**2)
        probability = special.ndtr((upper - correlation * scores) / spread) - special.ndtr(
            (lower - correlation * scores) / spread
        )
        return -float(np.sum(np.log(np.maximum(probability, 1e-300))))

    return _maximize(negative_log_likelihood)


def _polychoric(first: OrdinalIntervals, second: OrdinalIntervals) -> float:
    first_levels = len(first.edges) - 1
    second_levels = len(second.edges) - 1
    if len(np.unique(first.codes)) < 2 or len(np.unique(second.codes)) < 2:
        return 0.0
    if first_levels * second_levels > _MAX_TABLE_CELLS:
        return _moment_estimate(first, second)
    counts = np.bincount(
        first.codes * second_levels + second.codes, minlength=first_levels * second_levels
    ).reshape(first_levels, second_levels)
    observed = counts > 0
    first_edges = np.clip(first.edges, -_TAIL, _TAIL)
    second_edges = np.clip(second.edges, -_TAIL, _TAIL)

    def negative_log_likelihood(correlation: float) -> float:
        cdf = _bivariate_cdf(first_edges, second_edges, correlation)
        cells = cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1] + cdf[:-1, :-1]
        return -float(np.sum(counts[observed] * np.log(np.maximum(cells[observed], 1e-12))))

    return _maximize(negative_log_likelihood)


def _bivariate_cdf(first_edges: np.ndarray, second_edges: np.ndarray, correlation: float):
    """Return P(X <= first_edges[i], Y <= second_edges[j]) for standard normal X and Y."""
    spread = np.sqrt(1 - correlation**2)
    span = first_edges + _TAIL
    # One Gauss-Legendre rule per upper edge integrates the density of X from -_TAIL to it.
    points = -_TAIL + span[:, None] * (_NODES[None, :] + 1) / 2
    weights = _WEIGHTS[None, :] * span[:, None] / 2 * stats.norm.pdf(points)
    inner = special.ndtr((second_edges[None, :, None] - correlation * points[:, None, :]) / spread)
    return np.einsum("iq,ijq->ij", weights, inner)


def _moment_estimate(first: OrdinalIntervals, second: OrdinalIntervals) -> float:
    """Approximate the hidden correlation from interval means, for very large tables."""
    first_means = _interval_means(first.edges)[first.codes]
    second_means = _interval_means(second.edges)[second.codes]
    if first_means.std() == 0 or second_means.std() == 0:
        return 0.0
    # The correlation of interval means is the hidden correlation times the spread of each.
    shrinkage = first_means.std() * second_means.std()
    return float(np.clip(np.corrcoef(first_means, second_means)[0, 1] / shrinkage, -1, 1))


def _interval_means(edges: np.ndarray) -> np.ndarray:
    """Return the mean of a standard normal variable inside each interval."""
    density = stats.norm.pdf(edges)
    mass = np.maximum(np.diff(special.ndtr(edges)), 1e-300)
    return (density[:-1] - density[1:]) / mass
