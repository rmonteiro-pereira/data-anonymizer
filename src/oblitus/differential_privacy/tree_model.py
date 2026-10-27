"""A tree-structured model fitted to noisy marginals, and sampling rows from it.

Everything here is post-processing of already-noisy numbers, so it costs no privacy. The
steps are: combine the noisy estimates of each column's distribution by inverse variance,
turn each noisy pair table into a non-negative table whose margins equal those distributions
(iterative proportional fitting), and read the tables as a tree of conditional
distributions. Because the pair tables agree on every shared column, the tree's rows have
exactly those pair tables as their marginals.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

_FITTING_ROUNDS = 50
# Keeps cells at zero from blocking proportional fitting.
_FLOOR_SHARE = 1e-6


@dataclass(frozen=True, slots=True)
class NoisyMarginals:
    """Noisy counts of each column and of each tree edge, with the noise variances."""

    one_way: list[np.ndarray]
    one_way_variance: float
    edges: list[tuple[int, int]]
    two_way: list[np.ndarray]
    two_way_variance: float


@dataclass(frozen=True, slots=True)
class TreeModel:
    """Root distribution plus, for each other column, its distribution given its parent."""

    root: int
    root_distribution: np.ndarray
    parents: dict[int, int]
    conditionals: dict[int, np.ndarray]
    visiting_order: list[int]
    estimated_rows: float

    def sample(self, rows: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
        """Draw `rows` rows of codes, one array per column."""
        codes = {self.root: rng.choice(len(self.root_distribution), rows, p=self.root_distribution)}
        for node in self.visiting_order:
            if node == self.root:
                continue
            table = self.conditionals[node][codes[self.parents[node]]]
            codes[node] = _draw_from_rows(table, rng)
        return codes


def fit_tree_model(noisy: NoisyMarginals) -> TreeModel:
    """Make the noisy marginals consistent and non-negative, and read them as a tree."""
    column_count = len(noisy.one_way)
    estimates = [_combined_estimate(node, noisy) for node in range(column_count)]
    estimated_rows = max(1.0, float(np.mean([estimate.sum() for estimate in estimates])))
    distributions = [_normalized(estimate) for estimate in estimates]
    joint_tables = {
        edge: _fitted_pair_table(table, distributions[edge[0]], distributions[edge[1]])
        for edge, table in zip(noisy.edges, noisy.two_way, strict=True)
    }
    return _read_as_tree(joint_tables, distributions, estimated_rows)


def _combined_estimate(node: int, noisy: NoisyMarginals) -> np.ndarray:
    """Inverse-variance average of every noisy estimate of one column's counts."""
    weights = [1 / noisy.one_way_variance]
    estimates = [noisy.one_way[node]]
    for (first, second), table in zip(noisy.edges, noisy.two_way, strict=True):
        if node not in (first, second):
            continue
        axis_to_sum = 1 if node == first else 0
        estimates.append(table.sum(axis=axis_to_sum))
        cells_summed = table.shape[axis_to_sum]
        weights.append(1 / (cells_summed * noisy.two_way_variance))
    return np.average(np.array(estimates), axis=0, weights=weights)


def _normalized(estimate: np.ndarray) -> np.ndarray:
    clipped = np.clip(estimate, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return np.full(len(estimate), 1 / len(estimate))
    return clipped / total


def _fitted_pair_table(
    noisy_table: np.ndarray, row_distribution: np.ndarray, column_distribution: np.ndarray
) -> np.ndarray:
    """A non-negative joint distribution whose margins are the two given distributions."""
    table = np.clip(noisy_table, 0.0, None)
    total = table.sum()
    if total <= 0:
        table = np.ones_like(table)
        total = table.sum()
    table = table / total + _FLOOR_SHARE / table.size
    for _ in range(_FITTING_ROUNDS):
        row_sums = table.sum(axis=1)
        table = table * _safe_ratio(row_distribution, row_sums)[:, None]
        column_sums = table.sum(axis=0)
        table = table * _safe_ratio(column_distribution, column_sums)[None, :]
    return table / table.sum()


def _safe_ratio(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.divide(target, current, out=np.zeros_like(target), where=current > 0)


def _read_as_tree(
    joint_tables: dict[tuple[int, int], np.ndarray],
    distributions: list[np.ndarray],
    estimated_rows: float,
) -> TreeModel:
    neighbours: dict[int, list[int]] = {node: [] for node in range(len(distributions))}
    for first, second in joint_tables:
        neighbours[first].append(second)
        neighbours[second].append(first)
    root = 0
    parents: dict[int, int] = {}
    order = [root]
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for neighbour in neighbours[node]:
            if neighbour != root and neighbour not in parents:
                parents[neighbour] = node
                order.append(neighbour)
                queue.append(neighbour)
    conditionals = {
        node: _conditional(node, parent, joint_tables, distributions[node])
        for node, parent in parents.items()
    }
    return TreeModel(root, distributions[root], parents, conditionals, order, estimated_rows)


def _conditional(
    node: int,
    parent: int,
    joint_tables: dict[tuple[int, int], np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    """Rows are the parent's values; each row is the node's distribution given that value."""
    if (parent, node) in joint_tables:
        table = joint_tables[(parent, node)]
    else:
        table = joint_tables[(node, parent)].T
    row_sums = table.sum(axis=1, keepdims=True)
    conditional = np.divide(table, row_sums, out=np.zeros_like(table), where=row_sums > 0)
    empty_rows = row_sums[:, 0] <= 0
    conditional[empty_rows] = fallback
    return conditional


def _draw_from_rows(distributions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """For each row of probabilities, draw one column index."""
    cumulative = np.cumsum(distributions, axis=1)
    draws = rng.random(len(distributions))[:, None] * cumulative[:, -1:]
    picked = (cumulative < draws).sum(axis=1)
    return np.minimum(picked, distributions.shape[1] - 1)
