"""Run each noise technique on one column at increasing strengths and measure the result.

The sweep is what the interactive page shows: for every technique and strength,
the released distribution, a sample of per-record shifts, and the deflection and
risk metrics, so a reader can slide the strength and watch the trade-off move.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from data_anonymizer.evaluation.disclosure import deflection_metrics
from data_anonymizer.surrogates import SurrogateFactory
from data_anonymizer.techniques import BUILTIN_TECHNIQUES, TechniqueContext

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


@dataclass(frozen=True, slots=True)
class SweepLevel:
    """One technique at one strength."""

    value: float
    histogram: list[int]
    scatter_released: list[float]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class TechniqueSweep:
    """One technique across all its strengths."""

    technique: str
    parameter: str
    histogram_edges: list[float]
    original_histogram: list[int]
    levels: list[SweepLevel]


@dataclass(frozen=True, slots=True)
class NoiseExploration:
    """The full sweep for one column, ready to tabulate or render."""

    column: str
    rows: int
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
    frame: pd.DataFrame,
    column: str,
    *,
    seed: int = 0,
    band_sd: float = 0.1,
    sweeps: Mapping[str, tuple[str, Sequence[float], Mapping[str, float]]] = DEFAULT_SWEEPS,
) -> NoiseExploration:
    """Sweep every technique in `sweeps` over one numeric column of `frame`."""
    if column not in frame.columns:
        raise ValueError(f"column {column!r} is not in the table")
    original = pd.to_numeric(frame[column], errors="coerce").dropna().reset_index(drop=True)
    if len(original) < 10:
        raise ValueError(f"column {column!r} needs at least 10 numeric values")

    rng = np.random.default_rng(seed)
    scatter_rows = np.sort(rng.choice(len(original), min(_SCATTER_POINTS, len(original)), False))
    released_by_level = _run_sweeps(original, sweeps, seed)

    technique_sweeps = []
    for name, (parameter, _values, _fixed) in sweeps.items():
        edges = _technique_edges(original, released_by_level[name])
        levels = [
            SweepLevel(
                value=float(value),
                histogram=_histogram(released, edges),
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
                levels=levels,
            )
        )

    return NoiseExploration(
        column=column,
        rows=len(original),
        band_sd=band_sd,
        scatter_original=_rounded(original.to_numpy()[scatter_rows]),
        sweeps=technique_sweeps,
    )


def _run_sweeps(
    original: pd.Series,
    sweeps: Mapping[str, tuple[str, Sequence[float], Mapping[str, float]]],
    seed: int,
) -> dict[str, list[tuple[float, pd.Series]]]:
    results: dict[str, list[tuple[float, pd.Series]]] = {}
    for name, (parameter, values, fixed) in sweeps.items():
        technique = BUILTIN_TECHNIQUES[name]
        runs = []
        for value in values:
            # A fresh generator per level keeps each level reproducible on its own.
            context = TechniqueContext(
                rng=np.random.default_rng(seed),
                secret_key=b"noise-explorer",
                surrogates=SurrogateFactory(),
            )
            params = {**fixed, parameter: _parameter_value(value)}
            released = technique.apply(original.astype(float), context, params)
            runs.append((float(value), released.astype(float)))
        results[name] = runs
    return results


def _parameter_value(value: float) -> float | int:
    return int(value) if float(value).is_integer() else float(value)


def _technique_edges(original: pd.Series, runs: list[tuple[float, pd.Series]]) -> np.ndarray:
    # One axis per technique, so a strong Laplace setting does not squash the others.
    # It covers the original and most of each released version, but never more than
    # one original range beyond either end; the far tails land in the end bins.
    low = float(original.min())
    high = float(original.max())
    span = (high - low) or 1.0
    for _value, released in runs:
        low = min(low, float(released.quantile(0.005)))
        high = max(high, float(released.quantile(0.995)))
    low = max(low, float(original.min()) - span)
    high = min(high, float(original.max()) + span)
    return np.linspace(low, high, _HISTOGRAM_BINS + 1)


def _histogram(values: pd.Series, edges: np.ndarray) -> list[int]:
    clipped = np.clip(values.to_numpy(dtype=float), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return [int(count) for count in counts]


def _rounded(values: np.ndarray) -> list[float]:
    return [float(f"{value:.6g}") for value in values]
