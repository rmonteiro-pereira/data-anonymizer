"""A Gaussian copula: learns each column's distribution and how columns move together.

Fitting maps every column to standard normal scores through its own empirical
distribution, then measures the correlation of those scores. Sampling draws
correlated normal scores and maps them back. Numbers come out interpolated between
observed quantiles, so the generator produces new values rather than copying rows.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from data_anonymizer.detection.kinds import as_nanoseconds
from data_anonymizer.taxonomy import ColumnKind

# Uniform values are kept away from 0 and 1, where the normal quantile is infinite.
_EDGE = 1e-9

# Eigenvalues below this are raised so the correlation matrix stays positive definite.
_MIN_EIGENVALUE = 1e-6


@dataclass(frozen=True, slots=True)
class NumericMarginal:
    """The empirical distribution of a numeric or date column."""

    kind: ColumnKind
    sorted_values: np.ndarray

    def to_uniform(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Map values to (0, 1) by their rank."""
        ranks = stats.rankdata(values, method="average")
        return ranks / (len(values) + 1)

    def from_uniform(self, uniform: np.ndarray) -> pd.Series:
        """Map (0, 1) values back to this column's scale by interpolated quantiles."""
        numbers = np.quantile(self.sorted_values, uniform)
        if self.kind is ColumnKind.DATETIME:
            return pd.Series(pd.to_datetime(np.round(numbers).astype("int64")))
        if self.kind is ColumnKind.INTEGER:
            return pd.Series(np.round(numbers).astype("int64"))
        return pd.Series(numbers)


@dataclass(frozen=True, slots=True)
class CategoricalMarginal:
    """Category shares of a category or boolean column, as cumulative intervals of (0, 1)."""

    kind: ColumnKind
    categories: np.ndarray
    cumulative: np.ndarray

    def to_uniform(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Map each category to a random point inside its interval."""
        index_of = {category: index for index, category in enumerate(self.categories)}
        indexes = np.array([index_of[value] for value in values])
        upper = self.cumulative[indexes]
        lower = np.where(indexes > 0, self.cumulative[np.maximum(indexes - 1, 0)], 0.0)
        return lower + rng.random(len(values)) * (upper - lower)

    def from_uniform(self, uniform: np.ndarray) -> pd.Series:
        """Map (0, 1) values to the category whose interval holds them."""
        indexes = np.searchsorted(self.cumulative, uniform, side="right")
        indexes = np.minimum(indexes, len(self.categories) - 1)
        chosen = self.categories[indexes]
        if self.kind is ColumnKind.BOOLEAN:
            return pd.Series(chosen.astype(bool))
        return pd.Series(chosen, dtype=object)


Marginal = NumericMarginal | CategoricalMarginal


@dataclass(frozen=True, slots=True)
class GaussianCopula:
    """A fitted copula over a set of columns."""

    columns: list[str]
    marginals: dict[str, Marginal]
    correlation: np.ndarray

    def sample(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw `rows` new rows with the learned marginals and correlations."""
        means = np.zeros(len(self.columns))
        scores = rng.multivariate_normal(means, self.correlation, size=rows)
        uniform = np.clip(stats.norm.cdf(scores), _EDGE, 1 - _EDGE)
        sampled = {
            column: self.marginals[column].from_uniform(uniform[:, index])
            for index, column in enumerate(self.columns)
        }
        return pd.DataFrame(sampled)


def fit_gaussian_copula(
    frame: pd.DataFrame, kinds: Mapping[str, ColumnKind], rng: np.random.Generator
) -> GaussianCopula:
    """Learn marginals and the correlation of normal scores for the given columns."""
    columns = list(kinds)
    marginals: dict[str, Marginal] = {}
    scores = np.empty((len(frame), len(columns)))
    for index, column in enumerate(columns):
        marginal, column_scores = _fit_column(frame[column], kinds[column], rng)
        marginals[column] = marginal
        scores[:, index] = column_scores
    return GaussianCopula(
        columns=columns,
        marginals=marginals,
        correlation=_correlation(scores),
    )


def _fit_column(
    values: pd.Series, kind: ColumnKind, rng: np.random.Generator
) -> tuple[Marginal, np.ndarray]:
    present_mask = values.notna().to_numpy()
    present = values[present_mask]
    marginal = _marginal(present, kind)
    raw = _raw_values(present, kind)

    # Missing cells get independent noise: they neither add nor remove correlation.
    scores = rng.standard_normal(len(values))
    if len(present):
        uniform = np.clip(marginal.to_uniform(raw, rng), _EDGE, 1 - _EDGE)
        scores[present_mask] = stats.norm.ppf(uniform)
    return marginal, scores


def _raw_values(present: pd.Series, kind: ColumnKind) -> np.ndarray:
    if kind is ColumnKind.DATETIME:
        return as_nanoseconds(present).to_numpy(dtype=float)
    if kind in (ColumnKind.INTEGER, ColumnKind.FLOAT):
        return pd.to_numeric(present).to_numpy(dtype=float)
    if kind is ColumnKind.BOOLEAN:
        return present.astype(bool).to_numpy(dtype=object)
    return present.astype(str).to_numpy(dtype=object)


def _marginal(present: pd.Series, kind: ColumnKind) -> Marginal:
    raw = _raw_values(present, kind)
    if kind in (ColumnKind.CATEGORY, ColumnKind.BOOLEAN):
        counts = pd.Series(raw).value_counts()
        cumulative = np.cumsum(counts.to_numpy()) / counts.sum()
        return CategoricalMarginal(
            kind=kind,
            categories=counts.index.to_numpy(dtype=object),
            cumulative=cumulative,
        )
    if len(raw) == 0:
        raw = np.array([0.0])
    return NumericMarginal(kind=kind, sorted_values=np.sort(raw))


def _correlation(scores: np.ndarray) -> np.ndarray:
    width = scores.shape[1]
    if width == 1 or len(scores) < 2:
        return np.eye(width)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(scores, rowvar=False)
    # A constant column has no defined correlation; treat it as independent.
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    return _nearest_positive_definite(correlation)


def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    raised = np.maximum(eigenvalues, _MIN_EIGENVALUE)
    rebuilt = eigenvectors @ np.diag(raised) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(scale, scale)
