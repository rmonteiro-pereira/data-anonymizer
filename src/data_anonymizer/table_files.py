"""Read and write tables as CSV or Parquet, chosen by file extension."""

from pathlib import Path

import pandas as pd

from data_anonymizer.errors import TableFormatError

_SUPPORTED = ".csv, .parquet"


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise TableFormatError(f"cannot read {path.name}; supported: {_SUPPORTED}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a CSV or Parquet file."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise TableFormatError(f"cannot write {path.name}; supported: {_SUPPORTED}")
