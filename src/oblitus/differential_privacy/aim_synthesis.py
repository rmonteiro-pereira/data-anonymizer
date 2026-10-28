"""Differentially private synthetic data with AIM (McKenna, Mullins, Sheldon and Miklau, 2022).

AIM spends a zCDP budget in rounds:
1. Noisy one-way counts of every column (Gaussian noise).
2. Each round, the exponential mechanism picks the group of columns, from the workload, whose
   counts the current model gets most wrong, and Gaussian noise measures that group's counts.
   The model is refitted to every measurement so far (`marginal_model.py`).
3. When a measurement barely changes the model, later rounds use less noise.

The workload is the list of column groups the synthetic data should reproduce: every pair of
columns by default. Unlike MST, AIM measures groups of three or more columns when the workload
asks for them, and spends the budget where the model is worst. Once the table passes its domain
checks, only the counts and the round choices touch the data, and each one is recorded in the
budget report. Domains are public, so they are never derived from the data.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oblitus.differential_privacy.budget import (
    PrivacyBudget,
    Spend,
    compose_budget,
    exponential_mechanism_spend,
    gaussian_spend,
)
from oblitus.differential_privacy.domains import parse_domain, released_columns
from oblitus.differential_privacy.marginal_model import (
    Clique,
    MarginalModel,
    NoisyMeasurement,
    fit_marginal_model,
)
from oblitus.differential_privacy.mst_synthesis import PrivateSynthesis
from oblitus.errors import PrivacyBudgetError, SchemaError

# Workload groups with more cells than this are left out: their counts drown in the noise.
MAX_WORKLOAD_CELLS = 10_000
# The share of each round's budget spent on measuring; the rest pays for choosing.
_MEASURE_SHARE = 0.9
_ROUNDS_PER_COLUMN = 16


@dataclass(frozen=True, slots=True)
class _Run:
    """The fixed inputs of one AIM run."""

    codes: list[np.ndarray]
    sizes: list[int]
    names: list[str]
    total_rho: float
    max_model_cells: float
    rng: np.random.Generator


def synthesize_aim(
    frame: pd.DataFrame,
    domains: Mapping[str, object],
    budget: PrivacyBudget,
    rng: np.random.Generator,
    rows: int | None = None,
    workload: Sequence[Sequence[str]] | None = None,
    max_model_cells: int = 10_000_000,
) -> PrivateSynthesis:
    """Release a synthetic copy of `frame` under `budget`, with AIM.

    `domains` maps every released column to its public domain (see `domains.py`). `workload`
    lists the column groups the result should reproduce, every pair by default. The model
    grows only up to `max_model_cells` cells, which bounds memory and run time. Without
    `rows`, the synthetic table has the noisy row count.
    """
    if budget.is_pure or budget.rho is None:
        raise PrivacyBudgetError("AIM uses Gaussian noise; declare delta (or rho) with epsilon")
    typed_domains = {name: parse_domain(value) for name, value in domains.items()}
    names = released_columns(frame, typed_domains)
    sizes = [typed_domains[name].size for name in names]
    codes = [typed_domains[name].encode(name, frame[name]) for name in names]
    run = _Run(codes, sizes, names, budget.rho, float(max_model_cells), rng)
    candidates = _candidates(_workload_cliques(workload, names, sizes), len(names))

    spends: list[Spend] = []
    measurements, model = _measure_rounds(run, candidates, spends)
    report = compose_budget(budget, spends)

    row_count = rows if rows is not None else int(round(model.estimated_rows))
    if row_count < 0:
        raise SchemaError("rows must not be negative")
    sampled = model.sample(row_count, rng)
    data = pd.DataFrame(
        {name: typed_domains[name].decode(sampled[index], rng) for index, name in enumerate(names)}
    )
    measured = tuple(
        dict.fromkeys(tuple(names[column] for column in item.clique) for item in measurements)
    )
    return PrivateSynthesis(data=data, budget=report, tree_edges=(), measured_marginals=measured)


def _workload_cliques(
    workload: Sequence[Sequence[str]] | None, names: list[str], sizes: list[int]
) -> list[Clique]:
    """The workload as column positions, without groups too large to measure usefully."""
    groups: list[Clique]
    if workload is None:
        groups = list(itertools.combinations(range(len(names)), 2))
    else:
        groups = [_positions(group, names) for group in workload]
    return [group for group in groups if _cells(group, sizes) <= MAX_WORKLOAD_CELLS]


def _positions(group: Sequence[str], names: list[str]) -> Clique:
    unknown = [name for name in group if name not in names]
    if unknown or not group:
        raise SchemaError(f"workload groups must name released columns; unknown: {unknown}")
    if len(set(group)) != len(group):
        raise SchemaError(f"a workload group names a column twice: {list(group)}")
    return tuple(sorted(names.index(name) for name in group))


def _candidates(workload: list[Clique], column_count: int) -> dict[Clique, float]:
    """Every subgroup of a workload group, weighted by how many workload columns it covers.

    Without a workload group (a single column, say), the rounds refine the columns' own counts.
    """
    if not workload:
        workload = [(column,) for column in range(column_count)]
    subgroups: set[Clique] = set()
    for group in workload:
        for size in range(1, len(group) + 1):
            subgroups.update(itertools.combinations(group, size))
    return {
        subgroup: float(sum(len(set(subgroup) & set(group)) for group in workload))
        for subgroup in sorted(subgroups, key=len)
    }


def _measure_rounds(
    run: _Run, candidates: dict[Clique, float], spends: list[Spend]
) -> tuple[list[NoisyMeasurement], MarginalModel]:
    """Measure every column, then add one group per round until the budget is spent."""
    column_count = len(run.sizes)
    rounds = _ROUNDS_PER_COLUMN * column_count
    noise_scale = float(np.sqrt(rounds / (2 * _MEASURE_SHARE * run.total_rho)))
    choice_epsilon = float(np.sqrt(8 * (1 - _MEASURE_SHARE) * run.total_rho / rounds))

    measurements = []
    for column in range(column_count):
        measurements.append(_measure(run, (column,), noise_scale, spends))
    model = fit_marginal_model(run.sizes, measurements)
    spent = sum(spend.rho for spend in spends)

    finished = False
    while not finished:
        remaining = run.total_rho - spent
        if remaining < 2 * _round_rho(noise_scale, choice_epsilon):
            # The last round spends whatever is left, so nothing of the budget is wasted.
            noise_scale = float(np.sqrt(1 / (2 * _MEASURE_SHARE * remaining)))
            choice_epsilon = float(np.sqrt(8 * (1 - _MEASURE_SHARE) * remaining))
            finished = True
        spent += _round_rho(noise_scale, choice_epsilon)
        size_limit = run.max_model_cells * spent / run.total_rho
        allowed = _fitting_candidates(candidates, model, run.sizes, size_limit)
        clique = _worst_approximated(run, allowed, model, choice_epsilon, noise_scale, spends)

        before = model.counts(clique)
        measurements.append(_measure(run, clique, noise_scale, spends))
        model = fit_marginal_model(run.sizes, measurements, model)
        change = float(np.abs(model.counts(clique) - before).sum())
        if change <= noise_scale * np.sqrt(2 / np.pi) * before.size:
            # The model already knew this group well: measure later groups more precisely.
            noise_scale /= 2
            choice_epsilon *= 2
    return measurements, model


def _round_rho(noise_scale: float, choice_epsilon: float) -> float:
    return 0.5 / noise_scale**2 + choice_epsilon**2 / 8


def _measure(
    run: _Run, clique: Clique, noise_scale: float, spends: list[Spend]
) -> NoisyMeasurement:
    """Noisy counts of one group. One record adds to one cell, so L2 sensitivity is one."""
    exact = _exact_counts(run, clique)
    noisy = exact + run.rng.normal(0.0, noise_scale, exact.size)
    label = ", ".join(run.names[column] for column in clique)
    spends.append(gaussian_spend(f"counts of ({label})", 1.0, noise_scale))
    return NoisyMeasurement(clique, noisy, noise_scale)


def _exact_counts(run: _Run, clique: Clique) -> np.ndarray:
    shape = tuple(run.sizes[column] for column in clique)
    flat = np.ravel_multi_index(tuple(run.codes[column] for column in clique), shape)
    return np.bincount(flat, minlength=int(np.prod(shape))).astype(float)


def _cells(clique: Clique, sizes: list[int]) -> int:
    return int(np.prod([sizes[column] for column in clique]))


def _fitting_candidates(
    candidates: dict[Clique, float], model: MarginalModel, sizes: list[int], size_limit: float
) -> dict[Clique, float]:
    """Groups the model can take on without growing past the limit for this point in the run.

    Groups inside a clique the model already holds cost no extra size, so they always fit.
    """
    model_cells = sum(_cells(clique, sizes) for clique in model.cliques)
    held = [set(clique) for clique in model.cliques]
    allowed = {}
    for clique, weight in candidates.items():
        is_free = any(set(clique) <= existing for existing in held)
        if is_free or model_cells + _cells(clique, sizes) <= size_limit:
            allowed[clique] = weight
    return allowed


def _worst_approximated(
    run: _Run,
    candidates: dict[Clique, float],
    model: MarginalModel,
    epsilon: float,
    noise_scale: float,
    spends: list[Spend],
) -> Clique:
    """Pick the group the model gets most wrong, with the exponential mechanism.

    The score is the weighted L1 error, less the error the measurement noise alone would
    cause. One record changes a group's counts by one, so a score moves by at most its weight.
    """
    cliques = list(candidates)
    scores = []
    for clique in cliques:
        exact = _exact_counts(run, clique)
        expected_noise = np.sqrt(2 / np.pi) * noise_scale * exact.size
        error = np.abs(exact - model.counts(clique)).sum() - expected_noise
        scores.append(candidates[clique] * error)
    sensitivity = max(candidates.values())
    weights = np.array(scores) * epsilon / (2 * sensitivity)
    picked = cliques[int(np.argmax(weights + run.rng.gumbel(size=len(cliques))))]
    label = ", ".join(run.names[column] for column in picked)
    spends.append(exponential_mechanism_spend(f"choose group ({label})", epsilon))
    return picked
