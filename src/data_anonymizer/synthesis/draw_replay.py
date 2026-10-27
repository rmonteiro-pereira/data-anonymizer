"""Keep the random numbers behind every generated row, so a build can be replayed.

Every random choice made while a table is built comes from a uniform number between 0 and 1:
- a drawn column's value,
- each call of `bernoulli`, `normal`, `poisson` or `uniform` in a formula,
- the `noise_std` noise of a formula column,
- the redraw of a formula column's conditional override.

A `DrawLedger` hands these numbers out. While a table is built it draws fresh ones from the
generator and remembers them. While a build is replayed it hands the remembered ones back, so a
changed input gives the same row its own random numbers again. That is what lets an intervention
ask what a row would have looked like under another treatment, and what lets a panel evaluate a
formula period by period.

The numbers travel with the rows as hidden columns of the working table, so constraint
filtering keeps them aligned with the rows they belong to.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

# Uniform numbers are kept away from 0 and 1, where the inverse normal function is infinite.
UNIFORM_EDGE = 1e-9

# Hidden columns of the working table that carry the stored numbers start with this text.
DRAW_PREFIX = "__draw__"
_SLOT_SEPARATOR = "::"

# A stored array is found by the column it belongs to and the slot it fills in that column.
DrawKey = tuple[str, str]
StoredDraws = dict[DrawKey, np.ndarray]


def normal_from_uniform(uniform: np.ndarray) -> np.ndarray:
    """Return the standard normal value that sits at each uniform number."""
    return special.ndtri(np.clip(uniform, UNIFORM_EDGE, 1 - UNIFORM_EDGE))


def poisson_from_uniform(uniform: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Return the Poisson count that sits at each uniform number, for a rate per row."""
    if np.any(rate < 0):
        raise ValueError("poisson needs a rate that is not negative")
    positive = rate > 0
    counts = stats.poisson.ppf(uniform, np.where(positive, rate, 1.0))
    return np.where(positive, counts, 0.0).astype(np.int64)


class DrawLedger:
    """The stored uniform numbers of one table, with the generator that makes new ones."""

    def __init__(
        self, rows: int, rng: np.random.Generator | None, arrays: Mapping[DrawKey, np.ndarray]
    ) -> None:
        self._rows = rows
        self._rng = rng
        self._arrays: StoredDraws = dict(arrays)

    @classmethod
    def recording(
        cls, rows: int, rng: np.random.Generator, base_uniforms: Mapping[str, np.ndarray]
    ) -> "DrawLedger":
        """Start a build that draws new numbers; drawn columns bring theirs."""
        return cls(rows, rng, {(name, "base"): values for name, values in base_uniforms.items()})

    @classmethod
    def replaying(cls, arrays: Mapping[DrawKey, np.ndarray]) -> "DrawLedger":
        """Start a replay that only hands back the numbers a build stored."""
        rows = len(next(iter(arrays.values()))) if arrays else 0
        return cls(rows, None, arrays)

    def column(self, name: str) -> "ColumnDraws":
        """Return the numbers of one column, with call sites counted from the first."""
        return ColumnDraws(self, name)

    def stored(self, key: DrawKey) -> np.ndarray | None:
        """Return the stored numbers for a slot, or None when the column has none there."""
        return self._arrays.get(key)

    def uniform(self, key: DrawKey) -> np.ndarray:
        """Return the numbers for a slot, drawing and storing them on first use when recording."""
        if key not in self._arrays:
            if self._rng is None:
                raise KeyError(f"no stored random numbers for {key[0]!r} ({key[1]})")
            self._arrays[key] = self._rng.random(self._rows)
        return self._arrays[key]

    def arrays(self) -> StoredDraws:
        """Return every stored array, by column and slot."""
        return dict(self._arrays)

    def subset(self, positions: np.ndarray) -> "DrawLedger":
        """Return a replay of the same numbers for some rows, such as one period of a panel."""
        chosen = {key: values[positions] for key, values in self._arrays.items()}
        return DrawLedger(len(next(iter(chosen.values()), [])), None, chosen)


class ColumnDraws:
    """The uniform numbers one column uses, each in its own slot."""

    def __init__(self, ledger: DrawLedger, name: str) -> None:
        self._ledger = ledger
        self._name = name
        self._calls_made = 0

    def base(self) -> np.ndarray | None:
        """The numbers a drawn column's values come from, or None for other columns."""
        return self._ledger.stored((self._name, "base"))

    def redraw(self) -> np.ndarray:
        """The numbers that rows of a formula column take when a condition redraws them."""
        return self._ledger.uniform((self._name, "redraw"))

    def noise(self) -> np.ndarray:
        """The numbers behind the formula's `noise_std` noise."""
        return self._ledger.uniform((self._name, "noise"))

    def next_call_site(self) -> np.ndarray:
        """The numbers for the next random function call, in the order the formula reads."""
        key = (self._name, f"call{self._calls_made}")
        self._calls_made += 1
        return self._ledger.uniform(key)


@dataclass(frozen=True, slots=True)
class ReplayableTable:
    """A generated table together with what is needed to build it again under an intervention.

    `table` is the published table, with null cells. `structural` is the same rows before cells
    were blanked, which is what the formulas saw. `draws` are the numbers behind those rows.
    """

    table: pd.DataFrame
    structural: pd.DataFrame
    draws: StoredDraws

    def ledger(self) -> DrawLedger:
        """Return a ledger that replays the numbers of these rows."""
        return DrawLedger.replaying(self.draws)


def attach_draws(frame: pd.DataFrame, ledger: DrawLedger) -> pd.DataFrame:
    """Return the table with the ledger's numbers added as hidden columns."""
    hidden = {
        f"{DRAW_PREFIX}{name}{_SLOT_SEPARATOR}{slot}": values
        for (name, slot), values in ledger.arrays().items()
    }
    return pd.concat([frame, pd.DataFrame(hidden, index=frame.index)], axis=1)


def detach_draws(frame: pd.DataFrame) -> tuple[pd.DataFrame, StoredDraws]:
    """Split a working table into its real columns and the stored numbers."""
    hidden_names = [name for name in frame.columns if str(name).startswith(DRAW_PREFIX)]
    arrays: StoredDraws = {}
    for hidden_name in hidden_names:
        name, slot = str(hidden_name)[len(DRAW_PREFIX) :].rsplit(_SLOT_SEPARATOR, 1)
        arrays[(name, slot)] = frame[hidden_name].to_numpy()
    return frame.drop(columns=hidden_names), arrays
