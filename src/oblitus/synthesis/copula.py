"""A Gaussian copula: learns each column's distribution and how columns move together.

Fitting gives every column a hidden normal variable through its own empirical
distribution, then estimates how those variables correlate (see `latent_correlation`).
Sampling draws correlated normal values and maps them back. Numbers come out interpolated
between observed quantiles, so the generator produces new values rather than copying rows.
A category seen only a few times is merged into `OTHER`, and the largest and smallest numbers
are pulled in, so no single record can reappear (see `rare_values`).

A copula can also be fitted per level of some category columns, so that each level keeps its
own distributions and correlations.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from oblitus.detection.kinds import as_nanoseconds
from oblitus.synthesis.latent_correlation import (
    ContinuousScores,
    LatentColumn,
    OrdinalIntervals,
    latent_correlation,
)
from oblitus.synthesis.rare_values import (
    MIN_RECORDS_STATISTICAL,
    merge_rare_categories,
    pull_in_extremes,
)
from oblitus.taxonomy import ColumnKind

# Uniform values are kept away from 0 and 1, where the normal quantile is infinite.
_EDGE = 1e-9

# Eigenvalues below this are raised so the correlation matrix stays positive definite.
_MIN_EIGENVALUE = 1e-6

# A level of a stratifying column gets its own copula from this many reference rows.
# Smaller levels are pooled, because a correlation from a few rows is mostly noise.
MIN_STRATUM_ROWS = 200


@dataclass(frozen=True, slots=True)
class NumericMarginal:
    """The empirical distribution of a numeric or date column."""

    kind: ColumnKind
    sorted_values: np.ndarray

    def to_scores(self, values: np.ndarray) -> np.ndarray:
        """Map values to standard normal scores by their rank."""
        ranks = stats.rankdata(values, method="average")
        return stats.norm.ppf(np.clip(ranks / (len(values) + 1), _EDGE, 1 - _EDGE))

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

    @property
    def edges(self) -> np.ndarray:
        """The normal values that separate the categories' intervals."""
        inner = stats.norm.ppf(np.clip(self.cumulative[:-1], _EDGE, 1 - _EDGE))
        return np.concatenate([[-np.inf], inner, [np.inf]])

    def codes(self, values: np.ndarray) -> np.ndarray:
        """Map each category to the position of its interval."""
        index_of = {category: index for index, category in enumerate(self.categories)}
        return np.array([index_of[value] for value in values], dtype=int)

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


@dataclass(frozen=True, slots=True)
class Stratum:
    """A copula fitted to the reference rows that share the same stratifying values."""

    rows: int
    copula: GaussianCopula | None
    fixed: dict[str, Any]

    def sample(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        table = self.copula.sample(rows, rng) if self.copula else pd.DataFrame(index=range(rows))
        for name, value in self.fixed.items():
            table[name] = pd.Series(
                [value] * rows, dtype=object if isinstance(value, str) else None
            )
        return table


@dataclass(frozen=True, slots=True)
class StratifiedCopula:
    """Copulas fitted separately per level, sampled in proportion to the level's size."""

    strata: list[Stratum]

    def sample(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw `rows` new rows, with each level present at its share of the reference."""
        shares = np.array([stratum.rows for stratum in self.strata], dtype=float)
        counts = rng.multinomial(rows, shares / shares.sum())
        pieces = [
            stratum.sample(int(count), rng)
            for stratum, count in zip(self.strata, counts, strict=True)
            if count > 0 or rows == 0
        ]
        table = pd.concat(pieces, ignore_index=True)
        return table.iloc[rng.permutation(len(table))].reset_index(drop=True)


def fit_gaussian_copula(frame: pd.DataFrame, kinds: Mapping[str, ColumnKind]) -> GaussianCopula:
    """Learn marginals and the hidden correlations for the given columns."""
    columns = list(kinds)
    marginals: dict[str, Marginal] = {}
    latent: list[LatentColumn] = []
    for column in columns:
        marginal, latent_column = _fit_column(frame[column], kinds[column])
        marginals[column] = marginal
        latent.append(latent_column)
    return GaussianCopula(columns=columns, marginals=marginals, correlation=_correlation(latent))


def fit_stratified_copula(
    frame: pd.DataFrame,
    kinds: Mapping[str, ColumnKind],
    stratify_by: Sequence[str],
    *,
    min_rows: int = MIN_STRATUM_ROWS,
) -> GaussianCopula | StratifiedCopula:
    """Fit one copula per level of the stratifying columns that has at least `min_rows` rows.

    The smaller levels, and rows with a missing stratifying value, share one pooled copula.
    The stratifying columns come out with the level's value in its own copula, and are
    learned like any other column in the pooled one.
    """
    if not stratify_by:
        return fit_gaussian_copula(frame, kinds)
    own_kinds = {name: kind for name, kind in kinds.items() if name not in stratify_by}
    strata = []
    pooled_parts = []
    for key, part in frame.groupby(list(stratify_by), dropna=False, sort=True):
        levels: tuple[Any, ...] = key if isinstance(key, tuple) else (key,)
        if len(part) >= min_rows and not any(pd.isna(level) for level in levels):
            fixed = {
                name: _level_value(level, kinds[name])
                for name, level in zip(stratify_by, levels, strict=True)
            }
            copula = fit_gaussian_copula(part, own_kinds) if own_kinds else None
            strata.append(Stratum(rows=len(part), copula=copula, fixed=fixed))
        else:
            pooled_parts.append(part)
    if pooled_parts:
        pooled = pd.concat(pooled_parts)
        strata.append(
            Stratum(rows=len(pooled), copula=fit_gaussian_copula(pooled, kinds), fixed={})
        )
    return StratifiedCopula(strata)


def _level_value(level: Any, kind: ColumnKind) -> Any:
    return bool(level) if kind is ColumnKind.BOOLEAN else str(level)


def _fit_column(values: pd.Series, kind: ColumnKind) -> tuple[Marginal, LatentColumn]:
    present_mask = values.notna().to_numpy()
    present = values[present_mask]
    raw = _raw_values(present, kind)
    if kind is ColumnKind.CATEGORY:
        raw = merge_rare_categories(raw, MIN_RECORDS_STATISTICAL)
    marginal = _marginal(raw, kind)
    if isinstance(marginal, NumericMarginal):
        scores = np.full(len(values), np.nan)
        if len(present):
            scores[present_mask] = marginal.to_scores(raw)
        return marginal, ContinuousScores(scores)
    codes = np.full(len(values), -1, dtype=int)
    codes[present_mask] = marginal.codes(raw)
    return marginal, OrdinalIntervals(codes, marginal.edges)


def _raw_values(present: pd.Series, kind: ColumnKind) -> np.ndarray:
    if kind is ColumnKind.DATETIME:
        return as_nanoseconds(present).to_numpy(dtype=float)
    if kind in (ColumnKind.INTEGER, ColumnKind.FLOAT):
        return pd.to_numeric(present).to_numpy(dtype=float)
    if kind is ColumnKind.BOOLEAN:
        return present.astype(bool).to_numpy(dtype=object)
    return present.astype(str).to_numpy(dtype=object)


def _marginal(raw: np.ndarray, kind: ColumnKind) -> Marginal:
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
    sorted_values = pull_in_extremes(np.sort(raw), MIN_RECORDS_STATISTICAL)
    return NumericMarginal(kind=kind, sorted_values=sorted_values)


def _correlation(columns: list[LatentColumn]) -> np.ndarray:
    width = len(columns)
    matrix = np.eye(width)
    for first in range(width):
        for second in range(first + 1, width):
            value = latent_correlation(columns[first], columns[second])
            matrix[first, second] = value
            matrix[second, first] = value
    return nearest_positive_definite(matrix)


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    raised = np.maximum(eigenvalues, _MIN_EIGENVALUE)
    rebuilt = eigenvectors @ np.diag(raised) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(scale, scale)
