"""Public column domains: the values and bounds a differentially private release is built on.

A domain must be declared by the data owner from public knowledge (a form's choices, a legal
age range), never read from the data. Reading the minimum, maximum or the list of values from
the data would itself reveal those records. A domain turns a column into small integer codes
and back, and every function here is a fixed, data-independent map per record, so it costs no
privacy.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from oblitus.errors import SchemaError


class CategoricalDomain(BaseModel):
    """A column whose possible values are listed in advance; any other value is an error."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["categorical"] = "categorical"
    values: list[Any] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _values_are_unique(cls, values: list[Any]) -> list[Any]:
        if len(set(values)) != len(values):
            raise ValueError("the domain lists a value more than once")
        return values

    @property
    def size(self) -> int:
        """How many codes the column has."""
        return len(self.values)

    def encode(self, column: str, series: pd.Series) -> np.ndarray:
        """Map each value to its position in `values`."""
        _refuse_missing(column, series)
        codes = pd.Index(self.values).get_indexer(series)
        if (codes < 0).any():
            raise SchemaError(
                f"column {column!r} holds values outside its declared domain; list every value "
                "the column can take (add a catch-all such as 'other' and map to it first)"
            )
        return codes.astype(np.int64)

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> pd.Series:
        """Map codes back to the declared values."""
        return pd.Series(pd.Index(self.values).take(codes))


class NumericDomain(BaseModel):
    """A number with public bounds, cut into equal-width bins.

    Values outside the bounds are clipped to them. Decoding draws a value uniformly inside the
    bin, so the synthetic column is as precise as the bins, not finer.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["numeric"] = "numeric"
    lower: float
    upper: float
    bins: int = Field(default=10, ge=1)
    integer: bool = False

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> "NumericDomain":
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("lower and upper must be finite")
        if not self.lower < self.upper:
            raise ValueError("lower must be below upper")
        return self

    @property
    def size(self) -> int:
        """How many codes the column has."""
        return self.bins

    def encode(self, column: str, series: pd.Series) -> np.ndarray:
        """Clip to the bounds and map each value to its bin."""
        _refuse_missing(column, series)
        values = pd.to_numeric(series).to_numpy(dtype=float)
        clipped = np.clip(values, self.lower, self.upper)
        position = (clipped - self.lower) / self._span()
        return np.minimum((position * self.bins).astype(np.int64), self.bins - 1)

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> pd.Series:
        """Draw a value uniformly inside each code's bin."""
        width = self._span() / self.bins
        values = self.lower + (codes + rng.random(len(codes))) * width
        if self.integer:
            return pd.Series(np.minimum(np.floor(values), self.upper).astype(np.int64))
        return pd.Series(values)

    def _span(self) -> float:
        # Integers occupy unit cells, so the top integer needs a whole cell of its own.
        return self.upper - self.lower + (1 if self.integer else 0)


ColumnDomain = Annotated[CategoricalDomain | NumericDomain, Field(discriminator="kind")]

_DOMAIN_ADAPTER: TypeAdapter[ColumnDomain] = TypeAdapter(ColumnDomain)


def parse_domain(value: Any) -> CategoricalDomain | NumericDomain:
    """Turn a domain object, a dict or a plain list of values into a typed domain."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return CategoricalDomain(values=list(value))
    return _DOMAIN_ADAPTER.validate_python(value)


def released_columns(
    frame: pd.DataFrame, domains: Mapping[str, CategoricalDomain | NumericDomain]
) -> list[str]:
    """The columns a synthesizer releases: every column with a declared domain, in order."""
    if not domains:
        raise SchemaError("declare a public domain for at least one column")
    missing = [name for name in domains if name not in frame.columns]
    if missing:
        raise SchemaError(f"domains name columns that are not in the table: {missing}")
    return list(domains)


def _refuse_missing(column: str, series: pd.Series) -> None:
    if series.isna().any():
        raise SchemaError(
            f"column {column!r} has missing values; fill them with a declared value first, "
            "because dropping or keeping them would depend on the data"
        )
