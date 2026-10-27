"""Run each noise technique on one column at increasing strengths and measure the result.

The sweep is what the interactive page shows: for every technique and strength,
the released distribution, a sample of per-record shifts, and the deflection and
risk metrics, so a reader can slide the strength and watch the trade-off move.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

from oblitus.engines import (
    column_names,
    first_row,
    sample_to_pandas,
    to_working_frame,
)
from oblitus.evaluation.disclosure import deflection_metrics
from oblitus.surrogates import SurrogateFactory
from oblitus.techniques import BUILTIN_TECHNIQUES, TechniqueContext
from oblitus.techniques.catalogue import apply_to_series

# For each technique: the parameter swept, its values from weakest to strongest,
# and fixed parameters. Laplace runs with the column's own bounds, which is
# convenient here but leaks the minimum and maximum in a real release.
DEFAULT_SWEEPS: Mapping[str, tuple[str, Sequence[float], Mapping[str, float]]] = {
    "gaussian_noise": ("relative_scale", (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5), {}),
    "multiplicative_noise": ("relative_scale", (0.02, 0.05, 0.1, 0.2, 0.3, 0.5), {}),
    "laplace_noise": ("epsilon", (100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5), {}),
    "rank_swap": ("max_distance_percent", (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0), {}),
    "microaggregate": ("group_size", (2, 3, 5, 10, 20, 50), {}),
}

_HISTOGRAM_BINS = 40
_SCATTER_POINTS = 1500

# Larger columns are explored on a random sample of this many values. The metrics and
# charts are shares and medians, which a sample this size estimates closely, and the
# sweep runs every technique about forty times, so the full table would take minutes.
DEFAULT_MAX_ROWS = 200_000

# The axis covers the middle of the original values, not its extremes, so one huge
# outlier does not squash every other value into a single bar.
_AXIS_QUANTILES = (0.005, 0.995)


@dataclass(frozen=True, slots=True)
class SweepLevel:
    """One technique at one strength."""

    value: float
    histogram: list[int]
    # Share of released values beyond the axis; they are counted in the end bands.
    outside_share: float
    scatter_released: list[float]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TechniqueSweep:
    """One technique across all its strengths."""

    technique: str
    parameter: str
    histogram_edges: list[float]
    original_histogram: list[int]
    original_outside_share: float
    levels: list[SweepLevel]


@dataclass(frozen=True, slots=True)
class NoiseExploration:
    """The full sweep for one column, ready to tabulate or render."""

    column: str
    rows: int
    # Numeric values in the column before sampling; equals `rows` when nothing was sampled.
    total_rows: int
    band_sd: float
    scatter_original: list[float]
    sweeps: list[TechniqueSweep] = field(default_factory=list)

    @property
    def metrics(self) -> pd.DataFrame:
        """One row per technique and strength, one column per metric."""
        records = [
            {"technique": sweep.technique, sweep.parameter: level.value, **level.metrics}
            for sweep in self.sweeps
            for level in sweep.levels
        ]
        return pd.DataFrame(records)

    def to_dict(self) -> dict:
        """Plain data for the page and for JSON export."""
        return asdict(self)


def explore_noise(
    frame: Any,
    column: str,
    *,
    seed: int = 0,
    band_sd: float = 0.1,
    sweeps: Mapping[str, tuple[str, Sequence[float], Mapping[str, float]]] = DEFAULT_SWEEPS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> NoiseExploration:
    """Sweep every technique in `sweeps` over one numeric column of a table.

    The table may be a pandas, Polars or DuckDB frame. Columns with more than `max_rows`
    numeric values are explored on a random sample, which is the only part brought to pandas.
    """
    if column not in column_names(frame):
        raise ValueError(f"column {column!r} is not in the table")
    numeric, total_rows = _numeric_values(frame, column, max_rows, seed)
    if total_rows < 10:
        raise ValueError(f"column {column!r} needs at least 10 numeric values")

    rng = np.random.default_rng(seed)
    original = _sample(numeric, max_rows, rng)
    scatter_rows = np.sort(rng.choice(len(original), min(_SCATTER_POINTS, len(original)), False))
    released_by_level = _run_sweeps(original, sweeps, seed)

    technique_sweeps = []
    for name, (parameter, _values, _fixed) in sweeps.items():
        edges = _technique_edges(original, released_by_level[name])
        levels = [
            SweepLevel(
                value=float(value),
                histogram=_histogram(released, edges),
                outside_share=_outside_share(released, edges),
                scatter_released=_rounded(released.to_numpy()[scatter_rows]),
                metrics=asdict(
                    deflection_metrics(original.to_numpy(), released.to_numpy(), band_sd=band_sd)
                ),
            )
            for value, released in released_by_level[name]
        ]
        technique_sweeps.append(
            TechniqueSweep(
                technique=name,
                parameter=parameter,
                histogram_edges=_rounded(edges),
                original_histogram=_histogram(original, edges),
                original_outside_share=_outside_share(original, edges),
                levels=levels,
            )
        )

    return NoiseExploration(
        column=column,
        rows=len(original),
        total_rows=total_rows,
        band_sd=band_sd,
        scatter_original=_rounded(original.to_numpy()[scatter_rows]),
        sweeps=technique_sweeps,
    )


def _numeric_values(frame: Any, column: str, max_rows: int, seed: int) -> tuple[pd.Series, int]:
    """The column's numeric values, at most about `max_rows` of them, and how many there are."""
    if isinstance(frame, pd.DataFrame):
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        return numeric, len(numeric)
    working, _engine = to_working_frame(frame)
    present = working.select(column).drop_nulls()
    total_rows = int(first_row(present.select(nw.len().alias("rows")))["rows"])
    if max_rows < 10:
        raise ValueError("max_rows must be at least 10")
    sampled = sample_to_pandas(present, [column], max_rows, seed)
    return pd.to_numeric(sampled[column], errors="coerce").dropna(), total_rows


def _sample(values: pd.Series, max_rows: int, rng: np.random.Generator) -> pd.Series:
    if max_rows < 10:
        raise ValueError("max_rows must be at least 10")
    if len(values) <= max_rows:
        return values.reset_index(drop=True)
    chosen = np.sort(rng.choice(len(values), max_rows, replace=False))
    return values.iloc[chosen].reset_index(drop=True)


def _run_sweeps(
    original: pd.Series,
    sweeps: Mapping[str, tuple[str, Sequence[float], Mapping[str, float]]],
    seed: int,
) -> dict[str, list[tuple[float, pd.Series]]]:
    results: dict[str, list[tuple[float, pd.Series]]] = {}
    # Laplace needs bounds; the column's own range is fine for a private exploration,
    # though a real release must fix them without looking at the data.
    bounds = {"lower": float(original.min()), "upper": float(original.max())}
    for name, (parameter, values, fixed) in sweeps.items():
        technique = BUILTIN_TECHNIQUES[name]
        if name == "laplace_noise":
            fixed = {**bounds, **fixed}
        runs = []
        for value in values:
            # The same key per level keeps each level reproducible on its own.
            context = TechniqueContext(
                secret_key=b"noise-explorer",
                surrogates=SurrogateFactory(),
                randomness_key=f"noise-explorer:{seed}".encode(),
            )
            params = {**fixed, parameter: _parameter_value(value)}
            released = apply_to_series(technique, original.astype(float), context, params)
            runs.append((float(value), released.astype(float)))
        results[name] = runs
    return results


def _parameter_value(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


def _technique_edges(original: pd.Series, runs: list[tuple[float, pd.Series]]) -> np.ndarray:
    # One axis per technique, so a strong Laplace setting does not squash the others.
    # It covers the middle of the original and of each released version, but never
    # more than one original span beyond either end; the far tails land in the end bins.
    lower_quantile, upper_quantile = _AXIS_QUANTILES
    original_low = float(original.quantile(lower_quantile))
    original_high = float(original.quantile(upper_quantile))
    span = (original_high - original_low) or 1.0
    low = original_low
    high = original_high
    for _value, released in runs:
        low = min(low, float(released.quantile(lower_quantile)))
        high = max(high, float(released.quantile(upper_quantile)))
    low = max(low, original_low - span)
    high = min(high, original_high + span)
    if high <= low:
        high = low + 1.0
    return np.linspace(low, high, _HISTOGRAM_BINS + 1)


def _histogram(values: pd.Series, edges: np.ndarray) -> list[int]:
    clipped = np.clip(values.to_numpy(dtype=float), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return [int(count) for count in counts]


def _outside_share(values: pd.Series, edges: np.ndarray) -> float:
    numbers = values.to_numpy(dtype=float)
    outside = (numbers < edges[0]) | (numbers > edges[-1])
    return float(outside.mean())


def _rounded(values: np.ndarray) -> list[float]:
    return [float(f"{value:.6g}") for value in values]
