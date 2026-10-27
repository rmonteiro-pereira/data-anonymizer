"""Missing-value mechanisms: blank cells whose chance depends on the data.

A column's plain `null_fraction` blanks cells completely at random. A `Missingness` setting
makes the chance depend on a value instead:
- `mnar` (missing not at random): on the column's own value, as when high incomes go unreported;
- `mar` (missing at random): on another column, as when older customers skip a field.

The chance follows a logistic curve in the driving value, standardized to mean 0 and standard
deviation 1, and the curve is shifted so that the column's overall null share stays what
`null_fraction` says.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import special

from oblitus.synthesis.logistic_calibration import intercept_for_mean_probability

if TYPE_CHECKING:
    from oblitus.synthesis.schema import ColumnSchema


class MissingMechanism(StrEnum):
    """What a column's missing cells depend on."""

    MAR = "mar"
    MNAR = "mnar"


class Missingness(BaseModel):
    """Make a column's null cells depend on a value.

    `mechanism` is `mnar` (the column's own value) or `mar` (the column `depends_on`).
    `strength` is how much one standard deviation of the driving value moves the log-odds of a
    cell being blank: positive means large values go missing more often, negative less often.
    The column's `null_fraction` is the overall share of blank cells and must be above zero.
    The driving column must be a number, a boolean or a date.
    """

    model_config = ConfigDict(extra="forbid")

    mechanism: MissingMechanism
    depends_on: str | None = None
    strength: float = 1.0

    @model_validator(mode="after")
    def _driver_matches_mechanism(self) -> "Missingness":
        if self.mechanism is MissingMechanism.MAR and self.depends_on is None:
            raise ValueError("a mar missingness needs depends_on, the column the nulls follow")
        if self.mechanism is MissingMechanism.MNAR and self.depends_on is not None:
            raise ValueError("a mnar missingness follows the column itself; leave depends_on out")
        return self


def null_shares_from_value(
    driving_values: pd.Series, setting: Missingness, overall_share: float
) -> np.ndarray:
    """Return each row's chance of a blank cell, averaging `overall_share` over the rows."""
    logits = setting.strength * _standardized(driving_values)
    intercept = intercept_for_mean_probability(logits, overall_share)
    return special.expit(logits + intercept)


def describe_missingness(
    columns: list["ColumnSchema"], splits: dict[str, pd.DataFrame]
) -> dict[str, dict[str, Any]]:
    """Return the answer key's account of why each published column has blank cells.

    `splits` maps a name such as "train" to the published rows it covers; the observed share of
    blank cells is reported for each.

    A column with only a `null_fraction` is missing completely at random. One with a `when`
    override that changes its null share is missing at random given the columns the condition
    reads. One with a `missingness` setting says its own mechanism.
    """
    described: dict[str, dict[str, Any]] = {}
    for column in columns:
        published = [frame for frame in splits.values() if column.name in frame.columns]
        if not published or not _can_be_blank(column):
            continue
        described[column.name] = {
            **_mechanism_of(column),
            "null_fraction": column.null_fraction,
            "observed_null_share": {
                name: float(frame[column.name].isna().mean()) for name, frame in splits.items()
            },
        }
    return described


def _can_be_blank(column: "ColumnSchema") -> bool:
    has_override = any(override.null_fraction is not None for override in column.conditions)
    return column.null_fraction > 0 or has_override


def _mechanism_of(column: "ColumnSchema") -> dict[str, Any]:
    setting = column.missingness
    if setting is not None:
        driver = column.name if setting.depends_on is None else setting.depends_on
        return {
            "mechanism": setting.mechanism.value,
            "depends_on": driver,
            "strength": setting.strength,
        }
    overrides = [override for override in column.conditions if override.null_fraction is not None]
    if overrides:
        return {
            "mechanism": MissingMechanism.MAR.value,
            "conditions": [
                {"when": override.when, "null_fraction": override.null_fraction}
                for override in overrides
            ],
        }
    return {"mechanism": "mcar"}


def _standardized(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(values):
        values = pd.Series(values.astype("int64"))
    numbers = values.astype(float).to_numpy()
    present = np.isfinite(numbers)
    if not present.any():
        return np.zeros(len(numbers))
    center = float(np.mean(numbers[present]))
    filled = np.where(present, numbers, center)
    spread = float(np.std(filled))
    if spread == 0.0:
        return np.zeros(len(filled))
    return (filled - center) / spread
