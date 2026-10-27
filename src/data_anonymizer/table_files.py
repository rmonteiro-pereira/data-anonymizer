"""Read and write tables as CSV or Parquet, chosen by file extension.

Files are read and written with Polars, which is faster and lighter than pandas and
needs no extra packages for Parquet.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from data_anonymizer.errors import TableFormatError

_SUPPORTED = ".csv, .parquet"


def read_table(path: Path) -> pl.DataFrame:
    """Read a CSV or Parquet file into a Polars DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # The whole file decides each column's type, so a late decimal in an integer
        # column does not fail the read. ISO dates are read as dates, so date techniques and
        # checks see them as dates rather than text.
        return pl.read_csv(path, infer_schema_length=None, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise TableFormatError(f"cannot read {path.name}; supported: {_SUPPORTED}")


def write_table(frame: Any, path: Path) -> None:
    """Write a Polars, pandas or DuckDB table to a CSV or Parquet file."""
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise TableFormatError(f"cannot write {path.name}; supported: {_SUPPORTED}")
    table = _as_polars(frame)
    if suffix == ".csv":
        table.write_csv(path)
    else:
        table.write_parquet(path)


def _as_polars(frame: Any) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    return frame.pl()
