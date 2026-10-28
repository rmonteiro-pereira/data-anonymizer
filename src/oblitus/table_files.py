"""Read and write tables as CSV or Parquet, chosen by file extension.

Files are read and written with Polars, which is faster and lighter than pandas and
needs no extra packages for Parquet. A file larger than memory is read in chunks of rows and
written one chunk at a time; `scan_table` gives a lazy view of a file for checks that only
need a few columns.
"""

import io
from collections.abc import Collection, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
import polars as pl
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from oblitus.errors import TableFormatError

_SUPPORTED = ".csv, .parquet"
# Older Polars releases have no unsigned 128-bit type.
_WIDE_INTEGERS = tuple(getattr(pl, name) for name in ("Int128", "UInt128") if hasattr(pl, name))


def read_table(path: Path, text_columns: Collection[str] = ()) -> pl.DataFrame:
    """Read a CSV or Parquet file into a Polars DataFrame.

    `text_columns` are read from a CSV as written, without type inference. Names the file
    does not have are ignored.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path, text_columns)
    if suffix == ".parquet":
        return _wide_integers_as_text(pl.read_parquet(path))
    raise TableFormatError(f"cannot read {path.name}; supported: {_SUPPORTED}")


def _read_csv(path: Path, text_columns: Collection[str]) -> pl.DataFrame:
    try:
        return pl.read_csv(
            path,
            infer_schema_length=None,
            try_parse_dates=True,
            schema_overrides=_csv_text_overrides(path, text_columns),
        )
    except pl.exceptions.PolarsError:
        pass
    # Polars' message quotes the cell it could not parse and its offset in the file, and both
    # are private. Raising after the except block leaves that error out of this one.
    raise TableFormatError(f"cannot read {path.name}: a value does not fit the type of its column")


def _csv_text_overrides(path: Path, text_columns: Collection[str]) -> dict[str, Any]:
    """Columns to read as text: the ones asked for, and every 128-bit integer column.

    The whole file decides each column's type, so a late decimal in an integer column does not
    fail the read, and ISO dates are read as dates. Polars guesses a 128-bit integer for any
    run of digits too long for 64 bits, including ones longer than 128 bits hold and digits
    of other scripts, and then fails on them. pandas has no 128-bit integer either. So such a
    column is read as text, and one long value never decides whether the file can be read.
    """
    inferred = pl.scan_csv(path, infer_schema_length=None, try_parse_dates=True).collect_schema()
    wide = [name for name, dtype in inferred.items() if dtype in _WIDE_INTEGERS]
    return {name: pl.String for name in [*wide, *text_columns]}


def _wide_integers_as_text(table: Any) -> Any:
    """A Polars frame or lazy frame with its 128-bit integer columns cast to text."""
    schema = table.collect_schema() if isinstance(table, pl.LazyFrame) else table.schema
    wide = [name for name, dtype in schema.items() if dtype in _WIDE_INTEGERS]
    if not wide:
        return table
    return table.with_columns(pl.col(wide).cast(pl.String))


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


def scan_table(path: Path, *, schema: pl.Schema | None = None) -> pl.LazyFrame:
    """A lazy view of a CSV or Parquet file, typed as `read_table` would type it.

    With `schema`, a CSV is read with exactly those types, for a file this module wrote.
    """
    suffix = _checked_suffix(path, "read")
    if suffix == ".parquet":
        return _wide_integers_as_text(pl.scan_parquet(path))
    if schema is not None:
        return pl.scan_csv(path, schema=schema)
    return pl.scan_csv(
        path,
        infer_schema_length=None,
        try_parse_dates=True,
        schema_overrides=_csv_text_overrides(path, ()),
    )


def iter_table_chunks(path: Path, chunk_rows: int) -> Iterator[pl.DataFrame]:
    """Read a CSV or Parquet file as frames of at most `chunk_rows` rows, in file order.

    Every chunk has the types a whole-file read gives, so a column is never typed differently
    from one chunk to the next. An empty file gives one empty frame.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1")
    suffix = _checked_suffix(path, "read")
    chunks = (
        _parquet_chunks(path, chunk_rows) if suffix == ".parquet" else _csv_chunks(path, chunk_rows)
    )
    produced = False
    for chunk in chunks:
        produced = True
        yield chunk
    if not produced:
        yield scan_table(path).head(0).collect()


def _parquet_chunks(path: Path, chunk_rows: int) -> Iterator[pl.DataFrame]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_rows):
        chunk = pl.from_arrow(batch)
        assert isinstance(chunk, pl.DataFrame)
        yield _wide_integers_as_text(chunk)


def _csv_chunks(path: Path, chunk_rows: int) -> Iterator[pl.DataFrame]:
    schema = scan_table(path).collect_schema()
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = _next_record(handle)
        records: list[str] = []
        while (record := _next_record(handle)) is not None:
            records.append(record)
            if len(records) == chunk_rows:
                yield _parse_csv_records(header, records, schema)
                records = []
        if records:
            yield _parse_csv_records(header, records, schema)


def _next_record(handle: io.TextIOBase) -> str | None:
    # A record ends at a line break outside quotes, so quote characters are counted: an even
    # count means every quoted field on the lines read so far is closed.
    record = ""
    for line in handle:
        record += line
        if record.count('"') % 2 == 0:
            return record
    return record or None


def _parse_csv_records(header: str | None, records: list[str], schema: pl.Schema) -> pl.DataFrame:
    text = (header or "") + "".join(records)
    chunk = pl.read_csv(io.BytesIO(text.encode("utf-8")), schema=schema)
    if len(chunk) != len(records):
        raise TableFormatError(
            "a quote character inside an unquoted field breaks chunked reading; "
            "quote the field or read the file without --chunk-rows"
        )
    return chunk


class TableChunkWriter:
    """Writes a CSV or Parquet file one chunk at a time, keeping one column type per column.

    The first chunk fixes the file's types. A later chunk is cast to them, and a column that
    cannot be cast raises, because the file would otherwise mix types. A file that fails
    half way is removed.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._suffix = _checked_suffix(path, "write")
        self._schema: pl.Schema | None = None
        self._handle: Any = None
        self._parquet_writer: pq.ParquetWriter | None = None

    @property
    def schema(self) -> pl.Schema:
        """The types of the written file; available after the first chunk."""
        if self._schema is None:
            raise ValueError("nothing has been written yet")
        return self._schema

    def __enter__(self) -> "TableChunkWriter":
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        if self._handle is not None:
            self._handle.close()
        if error is not None:
            self._path.unlink(missing_ok=True)

    def write(self, chunk: pl.DataFrame) -> None:
        """Append a chunk, writing the header or the Parquet schema with the first one."""
        first = self._schema is None
        if first:
            self._schema = _writable_schema(chunk.schema)
        chunk = self._cast_to_schema(chunk)
        if self._suffix == ".csv":
            self._write_csv(chunk, first)
        else:
            self._write_parquet(chunk)

    def _cast_to_schema(self, chunk: pl.DataFrame) -> pl.DataFrame:
        try:
            return chunk.cast(self.schema)
        except pl.exceptions.PolarsError as error:
            raise TableFormatError(
                f"a column changed type between chunks of {self._path.name}: {error}"
            ) from error

    def _write_csv(self, chunk: pl.DataFrame, first: bool) -> None:
        if first:
            self._handle = self._path.open("wb")
        chunk.write_csv(self._handle, include_header=first)

    def _write_parquet(self, chunk: pl.DataFrame) -> None:
        arrow_table = chunk.to_arrow()
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self._path, arrow_table.schema)
        self._parquet_writer.write_table(arrow_table)


def _writable_schema(schema: pl.Schema) -> pl.Schema:
    # A column empty in the first chunk has no type of its own, and text holds any later value.
    return pl.Schema(
        {name: pl.String if dtype == pl.Null else dtype for name, dtype in schema.items()}
    )


def _checked_suffix(path: Path, verb: str) -> str:
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".parquet"):
        raise TableFormatError(f"cannot {verb} {path.name}; supported: {_SUPPORTED}")
    return suffix


def _as_polars(frame: Any) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    return frame.pl()
