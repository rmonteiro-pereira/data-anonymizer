"""Differentially private counts, sums and means per group, released with OpenDP.

Group-by columns come with a public list of values, and every sum or mean comes with public
lower and upper bounds. Both are declared by the caller and never read from the data: reading
them would leak the very records the noise protects. Every combination of group values is
released, including empty ones, so the set of rows reveals nothing about the data.

A record belongs to exactly one group, so the groups are released side by side (parallel
composition) and each statistic costs its budget share once, however many groups there are.
The privacy unit is one record added or removed.
"""

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import stats

from oblitus.differential_privacy.budget import (
    BudgetReport,
    PrivacyBudget,
    Spend,
    compose_budget,
    pure_epsilon_spend,
)
from oblitus.differential_privacy.domains import CategoricalDomain, numbers_or_none
from oblitus.errors import SchemaError

EXTRA_HINT = 'differentially private aggregates need OpenDP: pip install "oblitus[opendp]"'
DEFAULT_CONFIDENCE = 0.95


class Statistic(BaseModel):
    """One released statistic: a count, or a sum or mean of a column with public bounds."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["count", "sum", "mean"]
    column: str | None = None
    lower: float | None = None
    upper: float | None = None

    @model_validator(mode="after")
    def _needs_bounds_for_sums_and_means(self) -> "Statistic":
        if self.kind == "count":
            return self
        if self.column is None or self.lower is None or self.upper is None:
            raise ValueError(
                f"{self.name!r}: a {self.kind} needs column, lower and upper, declared from "
                "public knowledge and never taken from the data"
            )
        if not self.lower < self.upper:
            raise ValueError(f"{self.name!r}: lower must be below upper")
        return self

    @property
    def query_count(self) -> int:
        """How many noisy queries this statistic makes: a mean needs a sum and a count."""
        return 2 if self.kind == "mean" else 1


@dataclass(frozen=True, slots=True)
class PrivateAggregates:
    """The noisy table, the half-width of each value's confidence interval, and the cost.

    `accuracy` has the same shape as `table`: for every group and statistic, the noisy value
    is within that distance of the exact one with probability `confidence`. The interval of a
    mean is a first-order approximation.
    """

    table: pd.DataFrame
    accuracy: pd.DataFrame
    confidence: float
    budget: BudgetReport


def private_aggregates(
    frame: pd.DataFrame,
    group_domains: Mapping[str, Sequence[object]],
    statistics: Sequence[Statistic],
    budget: PrivacyBudget,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PrivateAggregates:
    """Release `statistics` for every combination of the public `group_domains` values.

    The budget is split over the noisy queries in proportion to how many each statistic
    makes. A pure epsilon budget uses Laplace noise; a budget with delta or rho uses Gaussian
    noise.
    """
    opendp = _load_opendp()
    _check_request(frame, group_domains, statistics, confidence)
    shares = _budget_shares(statistics, budget)
    report = compose_budget(budget, [share.spend for share in shares])
    group_keys = list(group_domains)
    group_codes = _group_codes(frame, group_domains)
    values: dict[str, list[float]] = {statistic.name: [] for statistic in statistics}
    half_widths: dict[str, list[float]] = {statistic.name: [] for statistic in statistics}
    combinations = list(itertools.product(*(group_domains[key] for key in group_keys)))
    for group_index in range(len(combinations)):
        rows = frame[group_codes == group_index]
        for statistic, share in zip(statistics, shares, strict=True):
            value, half_width = _release(opendp, rows, statistic, share, budget, confidence)
            values[statistic.name].append(value)
            half_widths[statistic.name].append(half_width)
    groups = pd.DataFrame(combinations, columns=group_keys)
    table = pd.concat([groups, pd.DataFrame(values)], axis=1)
    accuracy = pd.concat([groups, pd.DataFrame(half_widths)], axis=1)
    return PrivateAggregates(table, accuracy, confidence, report)


@dataclass(frozen=True, slots=True)
class _Share:
    spend: Spend
    epsilon: float | None
    rho: float | None


def _load_opendp():
    try:
        import opendp.prelude as opendp
    except ImportError as error:
        raise ImportError(EXTRA_HINT) from error
    opendp.enable_features("contrib")
    return opendp


def _check_request(
    frame: pd.DataFrame,
    group_domains: Mapping[str, Sequence[object]],
    statistics: Sequence[Statistic],
    confidence: float,
) -> None:
    if not statistics:
        raise SchemaError("ask for at least one statistic")
    names = [statistic.name for statistic in statistics]
    if len(set(names)) != len(names):
        raise SchemaError("statistic names must be unique")
    if not 0 < confidence < 1:
        raise SchemaError("confidence must be between 0 and 1")
    needed = list(group_domains) + [s.column for s in statistics if s.column is not None]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise SchemaError(f"the table has no column {missing}")
    for statistic in statistics:
        if statistic.column is not None and frame[statistic.column].isna().any():
            raise SchemaError(f"column {statistic.column!r} has missing values; fill them first")


def _budget_shares(statistics: Sequence[Statistic], budget: PrivacyBudget) -> list[_Share]:
    total_queries = sum(statistic.query_count for statistic in statistics)
    shares = []
    for statistic in statistics:
        fraction = statistic.query_count / total_queries
        if budget.is_pure:
            epsilon = budget.epsilon * fraction
            shares.append(_Share(pure_epsilon_spend(statistic.name, epsilon), epsilon, None))
        else:
            rho = (budget.rho or 0.0) * fraction
            shares.append(_Share(Spend(statistic.name, rho), None, rho))
    return shares


def _group_codes(frame: pd.DataFrame, group_domains: Mapping[str, Sequence[object]]) -> np.ndarray:
    """Number each row's group, refusing values outside the declared lists.

    Groups are numbered like the rows of `itertools.product` over the declared lists.
    """
    codes = np.zeros(len(frame), dtype=np.int64)
    for key, values in group_domains.items():
        domain = CategoricalDomain(values=list(values))
        codes = codes * domain.size + domain.encode(key, frame[key])
    return codes


def _release(
    opendp, rows: pd.DataFrame, statistic: Statistic, share: _Share, budget, confidence: float
) -> tuple[float, float]:
    """One noisy statistic for one group, and its interval half-width."""
    if statistic.kind == "count":
        count, count_scale = _noisy_query(opendp, share, budget, 1, [1.0] * len(rows), None)
        return count, _half_width(budget, count_scale, confidence)
    numbers = numbers_or_none(rows[statistic.column])
    if numbers is None:
        raise SchemaError(f"column {statistic.column!r} holds values that are not numbers")
    column_values = numbers.tolist()
    bounds = (float(statistic.lower or 0.0), float(statistic.upper or 0.0))
    total, total_scale = _noisy_query(
        opendp, share, budget, statistic.query_count, column_values, bounds
    )
    if statistic.kind == "sum":
        return total, _half_width(budget, total_scale, confidence)
    count, count_scale = _noisy_query(opendp, share, budget, 2, [1.0] * len(rows), None)
    return _mean_with_interval(budget, total, total_scale, count, count_scale, bounds, confidence)


def _noisy_query(opendp, share: _Share, budget, split: int, values, bounds):
    """Run one OpenDP query on one group: a clamped sum when `bounds` are given, else a count."""
    loss = (
        opendp.loss_of(epsilon=share.epsilon) if budget.is_pure else opendp.loss_of(rho=share.rho)
    )
    context = opendp.Context.compositor(
        data=values,
        privacy_unit=opendp.unit_of(contributions=1),
        privacy_loss=loss,
        split_evenly_over=split,
        domain=opendp.vector_domain(opendp.atom_domain(T=float, nan=False)),
    )
    query = context.query()
    query = query.count() if bounds is None else query.clamp(bounds).sum()
    query = query.laplace() if budget.is_pure else query.gaussian()
    return float(query.release()), float(query.param())


def _half_width(budget: PrivacyBudget, scale: float, confidence: float) -> float:
    """Distance within which noise of this scale stays with probability `confidence`."""
    miss = 1 - confidence
    if budget.is_pure:
        return scale * math.log(1 / miss)
    return scale * float(stats.norm.ppf(1 - miss / 2))


def _mean_with_interval(
    budget: PrivacyBudget,
    total: float,
    total_scale: float,
    count: float,
    count_scale: float,
    bounds: tuple[float, float],
    confidence: float,
) -> tuple[float, float]:
    """Noisy total over noisy count, kept inside the public bounds.

    Each of the two noisy values gets half the miss probability, so the combined interval
    holds with at least `confidence`. The interval is first-order: it ignores the curvature
    of the ratio, which matters only when the noisy count is close to zero.
    """
    half_miss_confidence = 1 - (1 - confidence) / 2
    total_half_width = _half_width(budget, total_scale, half_miss_confidence)
    count_half_width = _half_width(budget, count_scale, half_miss_confidence)
    size = max(count, 1.0)
    mean = min(max(total / size, bounds[0]), bounds[1])
    half_width = (total_half_width + abs(mean) * count_half_width) / size
    return mean, min(half_width, bounds[1] - bounds[0])
