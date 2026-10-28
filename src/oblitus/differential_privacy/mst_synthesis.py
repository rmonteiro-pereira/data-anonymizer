"""Differentially private synthetic data with MST (McKenna, Sheldon and Miklau, 2021).

MST spends a zCDP budget in three equal parts:
1. Noisy one-way counts of every column (Gaussian noise).
2. A spanning tree of the columns, chosen edge by edge with the exponential mechanism,
   preferring pairs whose joint counts differ most from independence.
3. Noisy two-way counts of the chosen pairs (Gaussian noise).

The noisy counts are then made consistent and non-negative (`tree_model.py`) and rows are
sampled from the resulting tree. Only steps 1 to 3 touch the data, and each one is recorded
in the budget report. Domains and bins are public, so they are never derived from the data.
"""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oblitus.differential_privacy.budget import (
    BudgetReport,
    PrivacyBudget,
    Spend,
    compose_budget,
    exponential_mechanism_spend,
    gaussian_spend,
)
from oblitus.differential_privacy.domains import (
    CategoricalDomain,
    NumericDomain,
    parse_domain,
    released_columns,
)
from oblitus.differential_privacy.tree_model import NoisyMarginals, TreeModel, fit_tree_model
from oblitus.errors import PrivacyBudgetError, SchemaError

# Joint tables larger than this make the pair counts too slow and too noisy to be useful.
MAX_PAIR_CELLS = 1_000_000
_PHASE_SHARE = 1 / 3


@dataclass(frozen=True, slots=True)
class PrivateSynthesis:
    """A synthetic table and what it cost.

    `budget` lists every noisy release. `measured_marginals` names every group of columns whose
    joint counts were measured; groups not listed are only as related as the model implies.
    `tree_edges` holds the measured column pairs of an MST tree, and is empty for AIM.
    """

    data: pd.DataFrame
    budget: BudgetReport
    tree_edges: tuple[tuple[str, str], ...]
    measured_marginals: tuple[tuple[str, ...], ...] = ()


def synthesize_mst(
    frame: pd.DataFrame,
    domains: Mapping[str, object],
    budget: PrivacyBudget,
    rng: np.random.Generator,
    rows: int | None = None,
) -> PrivateSynthesis:
    """Release a synthetic copy of `frame` under `budget`.

    `domains` maps every released column to its public domain (see `domains.py`). Columns not
    listed are not released. Without `rows`, the synthetic table has the noisy row count.
    """
    if budget.is_pure or budget.rho is None:
        raise PrivacyBudgetError("MST uses Gaussian noise; declare delta (or rho) with epsilon")
    typed_domains = {name: parse_domain(value) for name, value in domains.items()}
    names = released_columns(frame, typed_domains)
    sizes = [typed_domains[name].size for name in names]
    _check_pair_sizes(names, sizes)
    codes = [typed_domains[name].encode(name, frame[name]) for name in names]

    spends: list[Spend] = []
    one_way, one_way_scale = _measure_one_way(codes, sizes, budget.rho, rng, names, spends)
    edges = _select_tree(codes, sizes, one_way, budget.rho, rng, names, spends)
    two_way, two_way_scale = _measure_edges(codes, sizes, edges, budget.rho, rng, names, spends)
    report = compose_budget(budget, spends)

    noisy = NoisyMarginals(one_way, one_way_scale**2, edges, two_way, two_way_scale**2)
    model = fit_tree_model(noisy)
    row_count = rows if rows is not None else int(round(model.estimated_rows))
    data = _decode_rows(model, row_count, names, typed_domains, rng)
    named_edges = tuple((names[first], names[second]) for first, second in edges)
    measured = tuple((name,) for name in names) + named_edges
    return PrivateSynthesis(
        data=data, budget=report, tree_edges=named_edges, measured_marginals=measured
    )


def _check_pair_sizes(names: list[str], sizes: list[int]) -> None:
    for (first, first_size), (second, second_size) in itertools.combinations(
        zip(names, sizes, strict=True), 2
    ):
        if first_size * second_size > MAX_PAIR_CELLS:
            raise SchemaError(
                f"columns {first!r} and {second!r} make a joint table of "
                f"{first_size * second_size:,} cells; use fewer bins or values"
            )


def _phase_rho(total_rho: float, column_count: int) -> float:
    """Each phase's share of the budget; a single column has no tree, so all goes to counts."""
    return total_rho if column_count == 1 else total_rho * _PHASE_SHARE


def _measure_one_way(
    codes: list[np.ndarray],
    sizes: list[int],
    total_rho: float,
    rng: np.random.Generator,
    names: list[str],
    spends: list[Spend],
) -> tuple[list[np.ndarray], float]:
    """Noisy counts of every column. One record adds to one cell of each, so L2 sensitivity is
    the square root of the column count."""
    column_count = len(codes)
    rho = _phase_rho(total_rho, column_count)
    scale = float(np.sqrt(column_count / (2 * rho)))
    counts = []
    for name, column_codes, size in zip(names, codes, sizes, strict=True):
        exact = np.bincount(column_codes, minlength=size).astype(float)
        counts.append(exact + rng.normal(0.0, scale, size))
        spends.append(gaussian_spend(f"one-way counts of {name}", 1.0, scale))
    return counts, scale


def _select_tree(
    codes: list[np.ndarray],
    sizes: list[int],
    noisy_one_way: list[np.ndarray],
    total_rho: float,
    rng: np.random.Generator,
    names: list[str],
    spends: list[Spend],
) -> list[tuple[int, int]]:
    """Pick column_count - 1 edges, each by the exponential mechanism among edges that join
    two separate components."""
    column_count = len(codes)
    if column_count == 1:
        return []
    rho_per_edge = total_rho * _PHASE_SHARE / (column_count - 1)
    epsilon_per_edge = float(np.sqrt(8 * rho_per_edge))
    scores = _dependence_scores(codes, sizes, noisy_one_way)
    component = list(range(column_count))
    chosen: list[tuple[int, int]] = []
    for _ in range(column_count - 1):
        candidates = [pair for pair in scores if component[pair[0]] != component[pair[1]]]
        gumbel = rng.gumbel(size=len(candidates))
        weights = np.array([scores[pair] for pair in candidates]) * epsilon_per_edge / 2
        first, second = candidates[int(np.argmax(weights + gumbel))]
        chosen.append((first, second))
        spends.append(
            exponential_mechanism_spend(
                f"choose tree edge {len(chosen)} ({names[first]}, {names[second]})",
                epsilon_per_edge,
            )
        )
        merged, absorbed = component[first], component[second]
        component = [merged if label == absorbed else label for label in component]
    return chosen


def _dependence_scores(
    codes: list[np.ndarray], sizes: list[int], noisy_one_way: list[np.ndarray]
) -> dict[tuple[int, int], float]:
    """How far each pair's joint counts are from independence, in total absolute count.

    The independence model comes from the noisy one-way counts that were already released, so
    one record changes a score by at most one, which is what the exponential mechanism needs.
    """
    clipped = [np.clip(counts, 0.0, None) for counts in noisy_one_way]
    estimated_rows = max(1.0, float(np.mean([counts.sum() for counts in clipped])))
    shares = [counts / counts.sum() if counts.sum() > 0 else counts for counts in clipped]
    scores = {}
    for first, second in itertools.combinations(range(len(codes)), 2):
        joint = _pair_counts(codes[first], codes[second], sizes[first], sizes[second])
        independent = estimated_rows * np.outer(shares[first], shares[second])
        scores[(first, second)] = float(np.abs(joint - independent).sum())
    return scores


def _pair_counts(
    first_codes: np.ndarray, second_codes: np.ndarray, first_size: int, second_size: int
) -> np.ndarray:
    flat = first_codes * second_size + second_codes
    return np.bincount(flat, minlength=first_size * second_size).reshape(first_size, second_size)


def _measure_edges(
    codes: list[np.ndarray],
    sizes: list[int],
    edges: list[tuple[int, int]],
    total_rho: float,
    rng: np.random.Generator,
    names: list[str],
    spends: list[Spend],
) -> tuple[list[np.ndarray], float]:
    """Noisy joint counts of the chosen pairs, with the noise scale shared by all of them."""
    if not edges:
        return [], 1.0
    rho = total_rho * _PHASE_SHARE
    scale = float(np.sqrt(len(edges) / (2 * rho)))
    tables = []
    for first, second in edges:
        exact = _pair_counts(codes[first], codes[second], sizes[first], sizes[second])
        tables.append(exact + rng.normal(0.0, scale, exact.shape))
        spends.append(
            gaussian_spend(f"two-way counts of ({names[first]}, {names[second]})", 1.0, scale)
        )
    return tables, scale


def _decode_rows(
    model: TreeModel,
    rows: int,
    names: list[str],
    domains: Mapping[str, CategoricalDomain | NumericDomain],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if rows < 0:
        raise SchemaError("rows must not be negative")
    sampled = model.sample(rows, rng)
    columns = {name: domains[name].decode(sampled[index], rng) for index, name in enumerate(names)}
    return pd.DataFrame(columns)
