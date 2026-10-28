"""A graphical model fitted to noisy marginals with Private-PGM, and sampling rows from it.

AIM measures groups of columns that need not form a tree, so the tree model used by MST does
not fit them. Private-PGM (the `mbi` package, from the authors of MST and AIM) finds the
distribution that best agrees with every noisy measurement. Everything here is
post-processing of already-noisy numbers, so it costs no privacy.

Columns are named by their position. Rows are drawn with the caller's generator, not with
`mbi`'s own sampler, which uses numpy's global random state and so ignores the seed.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from oblitus.errors import MissingExtraError

EXTRA_HINT = 'AIM synthesis needs Private-PGM: pip install "oblitus[aim]"'
# Each fit runs this many mirror-descent steps; the AIM paper uses the same number.
_FIT_ITERATIONS = 1000

Clique = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NoisyMeasurement:
    """Noisy counts of one group of columns, flattened in row-major order."""

    clique: Clique
    counts: np.ndarray
    noise_scale: float


class MarginalModel:
    """A distribution over all columns that agrees with the noisy measurements."""

    def __init__(self, fitted: Any, sizes: Sequence[int]) -> None:
        self._fitted = fitted
        self._sizes = list(sizes)

    @property
    def cliques(self) -> list[Clique]:
        """The groups of columns the model holds joint tables for."""
        return [tuple(clique) for clique in self._fitted.cliques]

    @property
    def estimated_rows(self) -> float:
        """The row count the noisy measurements imply."""
        return max(1.0, float(self._fitted.total))

    def counts(self, clique: Clique) -> np.ndarray:
        """The model's counts over a group of columns, flattened in row-major order."""
        return np.asarray(self._fitted.project(clique).datavector(), dtype=float)

    def sample(self, rows: int, rng: np.random.Generator) -> dict[int, np.ndarray]:
        """Draw `rows` rows of codes, one array per column."""
        mbi = load_mbi()
        tree, elimination_order = mbi.junction_tree.make_junction_tree(
            self._fitted.domain, self.cliques
        )
        # Sampling in reverse elimination order is exact when each column is drawn given its
        # already drawn neighbours in the junction tree, which always share one tree clique.
        tree_cliques = [set(clique) for clique in tree.nodes]
        codes: dict[int, np.ndarray] = {}
        for column in reversed(elimination_order):
            parents = _sampled_neighbours(column, tree_cliques, codes)
            table = np.asarray(self._fitted.project(parents + (column,)).datavector(False))
            codes[column] = _draw_column(table, parents, codes, rows, rng)
        return codes


def fit_marginal_model(
    sizes: Sequence[int],
    measurements: Sequence[NoisyMeasurement],
    previous: MarginalModel | None = None,
) -> MarginalModel:
    """Fit Private-PGM to the measurements over columns with the given value counts.

    Starting from the previous round's model makes each refit converge in fewer steps.
    """
    mbi = load_mbi()
    domain = mbi.Domain(tuple(range(len(sizes))), tuple(sizes))
    linear = [
        mbi.LinearMeasurement(measurement.counts, measurement.clique, measurement.noise_scale)
        for measurement in measurements
    ]
    warm_start = None if previous is None else previous._fitted
    fitted = mbi.estimation.MirrorDescent().estimate(
        domain, linear, iters=_FIT_ITERATIONS, warm_start=warm_start
    )
    return MarginalModel(fitted, sizes)


def load_mbi():
    """Import `mbi`, with JAX set to double precision before it loads."""
    try:
        import jax
    except ImportError as error:
        raise MissingExtraError(EXTRA_HINT) from error
    # Single precision makes the fit stall on large tables, and the compilation cache only
    # slows down the many small programs a fit compiles.
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            import mbi  # type: ignore[import-untyped]
        except ImportError as error:
            raise MissingExtraError(EXTRA_HINT) from error
    return mbi


def _sampled_neighbours(
    column: int, cliques: list[set[int]], codes: dict[int, np.ndarray]
) -> tuple[int, ...]:
    """Columns already drawn that share a junction-tree clique with `column`."""
    related = set().union(*(clique for clique in cliques if column in clique))
    return tuple(sorted(related & codes.keys()))


def _draw_column(
    table: np.ndarray,
    parents: tuple[int, ...],
    codes: dict[int, np.ndarray],
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one column from its joint table with its parents, by inverse-CDF sampling.

    The column is the table's last axis; the parents are the others, in order.
    """
    table = np.clip(table, 0.0, None)
    totals = table.sum(axis=-1, keepdims=True)
    uniform = np.full_like(table, 1.0 / table.shape[-1])
    conditional = np.divide(table, totals, out=uniform, where=totals > 0)
    if parents:
        conditional = conditional[tuple(codes[parent] for parent in parents)]
    else:
        conditional = np.broadcast_to(conditional, (rows, table.shape[-1]))
    cumulative = conditional.cumsum(axis=-1)
    draws = rng.random((rows, 1)) * cumulative[:, -1:]
    picked = (draws > cumulative).sum(axis=-1)
    return np.minimum(picked, table.shape[-1] - 1)
