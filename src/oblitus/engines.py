"""What differs between the dataframe engines a table can live in.

Techniques are written once, as Narwhals expressions, and run on every engine.
Four things cannot be written that way, and live here instead:
- converting the user's frame in and out, and reading samples or single columns as pandas;
- turning a key column into keyed random numbers, which needs SHA-256;
- building a small lookup table on the same engine, and the same connection, as the data.

pandas and Polars frames are processed in memory with Polars. DuckDB relations and Spark
DataFrames stay lazy: the work becomes SQL that the engine runs, and rows are collected only
when a result is read. A Spark table is any PySpark DataFrame, or a `sqlframe` one, which
has the same API on top of DuckDB and serves as the test stand-in where Java is missing.
"""

import hashlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from enum import StrEnum
from importlib import import_module
from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl

# Each SHA-256 digest yields four uniform numbers of 52 bits, one per 13 hex digits.
UNIFORMS_PER_DIGEST = 4
_HEX_DIGITS_PER_UNIFORM = 13
_UNIFORM_SCALE = float(16**_HEX_DIGITS_PER_UNIFORM)

# What a missing key hashes as, so missing entities still share one random number.
NULL_KEY_TEXT = "<null>"

WorkingFrame = nw.DataFrame[Any] | nw.LazyFrame[Any]

_QUOTE_NAMES = {'"': "double quotes", "`": "backticks"}


class Engine(StrEnum):
    """Where the user's table lives, and so the kind of frame they get back."""

    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"
    SPARK = "spark"

    @property
    def is_lazy(self) -> bool:
        """True when the table is a query the engine runs, not rows held in memory."""
        return self in (Engine.DUCKDB, Engine.SPARK)


def engine_of(native: Any) -> Engine:
    """Name the engine of a user's frame, or raise TypeError for an unsupported one."""
    module = type(native).__module__.split(".")[0]
    if module == "pandas":
        return Engine.PANDAS
    if module == "polars":
        return Engine.POLARS
    if module in ("duckdb", "_duckdb"):
        return Engine.DUCKDB
    if module in ("pyspark", "sqlframe"):
        return Engine.SPARK
    raise TypeError(
        f"tables must be pandas, Polars, DuckDB or Spark frames, not {type(native).__name__}"
    )


def engine_of_frame(frame: WorkingFrame) -> Engine:
    """Name the engine behind a Narwhals frame, as `engine_of` does for a user's frame."""
    if isinstance(frame, nw.DataFrame):
        return Engine.POLARS
    implementation = frame.implementation
    if implementation is nw.Implementation.DUCKDB:
        return Engine.DUCKDB
    if implementation.is_spark_like():
        return Engine.SPARK
    return Engine.POLARS


def to_working_frame(native: Any) -> tuple[WorkingFrame, Engine]:
    """Wrap a user's frame for processing, and say which engine it came from."""
    engine = engine_of(native)
    if engine is Engine.PANDAS:
        return nw.from_native(_polars_from_pandas(native), eager_only=True), engine
    if engine is Engine.POLARS and isinstance(native, pl.LazyFrame):
        return nw.from_native(native.collect(), eager_only=True), engine
    _check_lazy_names(column_names(native), engine)
    return nw.from_native(native), engine


def _check_lazy_names(names: Sequence[str], engine: Engine) -> None:
    # Narwhals writes column references without escaping dots and quotes, so such names fail
    # deep inside a query. Refusing them here gives a clear message.
    quote = '"' if engine is Engine.DUCKDB else "`"
    unsafe = [name for name in names if "." in name or quote in name]
    if unsafe:
        listed = ", ".join(repr(name) for name in unsafe)
        label = "DuckDB" if engine is Engine.DUCKDB else "Spark"
        raise ValueError(
            f"{label} columns cannot contain dots or {_QUOTE_NAMES[quote]} here: {listed}; "
            "rename them, for example with a select ... as alias"
        )


def as_working_frame(frame: Any) -> WorkingFrame:
    """Return a Narwhals frame as it is, and wrap a user's frame for processing."""
    if isinstance(frame, (nw.DataFrame, nw.LazyFrame)):
        return frame
    return to_working_frame(frame)[0]


def collect(frame: WorkingFrame) -> "nw.DataFrame[Any]":
    """Run a lazy frame and return its rows in memory; an in-memory frame is returned as is."""
    if isinstance(frame, nw.LazyFrame):
        return frame.collect()
    return frame


def first_row(frame: WorkingFrame) -> dict[str, Any]:
    """Run a one-row aggregate and return it as a dict of Python values."""
    eager = collect(frame)
    return dict(zip(eager.columns, eager.row(0), strict=True))


def column_names(native: Any) -> list[str]:
    """The column names of a pandas, Polars, DuckDB or Spark frame, without loading rows."""
    return list(nw.from_native(native).collect_schema().names())


def pandas_columns(native: Any) -> Iterator[tuple[str, Any]]:
    """Yield each column of a user's frame as a pandas Series, one at a time.

    Only one column is in pandas memory at once, so a large Polars or DuckDB table can be
    scanned without converting all of it.
    """
    engine = engine_of(native)
    for name in column_names(native):
        if engine is Engine.PANDAS:
            yield name, native[name]
        elif engine is Engine.POLARS:
            yield name, _collected_polars(native).get_column(name).to_pandas()
        elif engine is Engine.SPARK:
            yield name, nw.from_native(native).select(name).collect().to_pandas()[name]
        else:
            yield name, native.project(_quoted(name)).df()[name]


def sample_to_pandas(frame: WorkingFrame, columns: Sequence[str], size: int, seed: int) -> Any:
    """Collect at most `size` random rows of the given columns as a pandas frame."""
    selected = frame.select(*columns)
    if engine_of_frame(selected) is Engine.SPARK:
        return _spark_sample(selected, size, seed)
    if isinstance(selected, nw.LazyFrame):
        relation = selected.to_native()
        reservoir = f"select * from __base using sample reservoir({size} rows) repeatable ({seed})"
        return relation.query("__base", reservoir).df()
    if len(selected) > size:
        return selected.sample(n=size, seed=seed).to_pandas()
    return selected.to_pandas()


def _spark_sample(selected: WorkingFrame, size: int, seed: int) -> Any:
    rows = first_row(selected.select(nw.len().alias("rows")))["rows"]
    if rows <= size:
        return collect(selected).to_pandas()
    # Keep each row with a seeded probability, a little above what is needed, and cut to size.
    fraction = min(1.0, 1.2 * size / rows)
    native = selected.to_native()
    sampled = native.where(_spark_functions(native).rand(seed) < fraction).limit(size)
    return nw.from_native(sampled).collect().to_pandas()


def _collected_polars(native: Any) -> pl.DataFrame:
    return native.collect() if isinstance(native, pl.LazyFrame) else native


def _polars_from_pandas(native: Any) -> pl.DataFrame:
    duplicated = sorted({str(name) for name in native.columns[native.columns.duplicated()]})
    if duplicated:
        raise ValueError(f"the table has duplicate column names: {', '.join(duplicated)}")
    columns = [_polars_column(str(name), native[name]) for name in native.columns]
    return pl.DataFrame(columns)


def _polars_column(name: str, values: Any) -> pl.Series:
    try:
        return pl.from_pandas(values.reset_index(drop=True)).rename(name)
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        # Python objects of mixed types have no single column type, and text holds them all.
        # The user gets the original column back when no action changes it.
        texts = [None if _is_missing(item) else str(item) for item in values.tolist()]
        return pl.Series(name, texts, dtype=pl.String)


def _is_missing(item: Any) -> bool:
    return item is None or (isinstance(item, float) and item != item)


def to_pandas_like(
    frame: WorkingFrame,
    original: Any,
    changed: Collection[str],
    *,
    row_order: Sequence[int] | None = None,
) -> Any:
    """Return an in-memory frame as pandas, shaped like the user's original pandas table.

    Columns no action changed are the user's own columns, with their dtype and values.
    Changed integer and boolean columns with gaps keep nullable pandas types, so large
    integers stay exact. The original index is restored, unless `row_order` says the rows
    were moved: it gives the original position of each row, and the index is then a fresh
    range.
    """
    released = collect(frame).to_native()
    columns = {}
    for name in released.columns:
        if name in changed or name not in original.columns:
            columns[name] = _pandas_column(released.get_column(name), original.get(name))
        else:
            columns[name] = _own_column(original[name], row_order)
    result = pd.DataFrame(columns)
    if row_order is None and len(result) == len(original):
        result.index = original.index
    return result


def _own_column(values: Any, row_order: Sequence[int] | None) -> Any:
    plain = values.reset_index(drop=True)
    if row_order is None:
        return plain
    return plain.iloc[list(row_order)].reset_index(drop=True)


def _pandas_column(values: pl.Series, original: Any) -> Any:
    if values.dtype.is_integer() and values.null_count():
        converted = pd.Series(pd.array(values.to_list(), dtype="Int64"))
    elif values.dtype == pl.Boolean and values.null_count():
        converted = pd.Series(pd.array(values.to_list(), dtype="boolean"))
    else:
        converted = values.to_pandas()
    if original is None:
        return converted
    return _with_original_extension_type(converted, original.dtype)


def _with_original_extension_type(converted: Any, original_dtype: Any) -> Any:
    # A nullable or text column that a technique left in the same kind keeps its pandas type.
    kind = str(original_dtype)
    if kind == "Int64" and pd.api.types.is_integer_dtype(converted.dtype):
        return converted.astype("Int64")
    if kind == "boolean" and pd.api.types.is_bool_dtype(converted.dtype):
        return converted.astype("boolean")
    if kind.startswith("string") and not pd.api.types.is_numeric_dtype(converted.dtype):
        return converted.astype(original_dtype)
    return converted


def to_native(frame: WorkingFrame, engine: Engine) -> Any:
    """Return the frame in the form the user handed it in."""
    native = frame.to_native()
    if engine is Engine.PANDAS:
        return native.to_pandas()
    return native


def add_keyed_uniforms(
    frame: WorkingFrame,
    engine: Engine,
    *,
    key: bytes,
    source: str,
    names: Sequence[str],
) -> WorkingFrame:
    """Add up to four columns of uniform numbers in (0, 1), keyed on `source`.

    Each number is part of SHA-256(key in hex, "|", the source value as text). Equal
    source values get equal numbers on every engine, and nobody without the key can
    predict them. The digest is never released, only used to draw noise, so the
    length-extension weakness of a key-prefixed hash does not apply.
    """
    if len(names) > UNIFORMS_PER_DIGEST:
        raise ValueError(f"at most {UNIFORMS_PER_DIGEST} uniform columns per call")
    prefix = key.hex() + "|"
    if engine is Engine.DUCKDB:
        return _duckdb_uniforms(frame, prefix, source, names)
    if engine is Engine.SPARK:
        return _spark_uniforms(frame, prefix, source, names)
    eager = frame.lazy().collect() if isinstance(frame, nw.LazyFrame) else frame
    texts = eager.get_column(source).cast(nw.String).to_list()
    digests = [_digest(prefix, text) for text in texts]
    columns = {
        name: [_uniform_from_hex(digest, position) for digest in digests]
        for position, name in enumerate(names)
    }
    added = nw.from_dict(columns, backend=nw.Implementation.POLARS)
    return nw.concat([eager, added], how="horizontal")


def _digest(prefix: str, text: str | None) -> str:
    message = prefix + (NULL_KEY_TEXT if text is None else text)
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def _uniform_from_hex(digest: str, position: int) -> float:
    start = position * _HEX_DIGITS_PER_UNIFORM
    whole = int(digest[start : start + _HEX_DIGITS_PER_UNIFORM], 16)
    # Half a step up keeps the number strictly between 0 and 1, so logarithms are safe.
    return (whole + 0.5) / _UNIFORM_SCALE


def _duckdb_uniforms(
    frame: WorkingFrame, prefix: str, source: str, names: Sequence[str]
) -> WorkingFrame:
    relation = frame.to_native()
    # The prefix is hex digits and a bar, so it is safe to place in the SQL text.
    text = f"coalesce(cast({_quoted(source)} as varchar), '{NULL_KEY_TEXT}')"
    digest = f"sha256('{prefix}' || {text})"
    columns = []
    for position, name in enumerate(names):
        start = position * _HEX_DIGITS_PER_UNIFORM + 1
        whole = f"('0x' || substr({digest}, {start}, {_HEX_DIGITS_PER_UNIFORM}))::UBIGINT"
        columns.append(f"(({whole}) + 0.5) / {_UNIFORM_SCALE} as {_quoted(name)}")
    return nw.from_native(relation.project("*, " + ", ".join(columns)))


def _spark_uniforms(
    frame: WorkingFrame, prefix: str, source: str, names: Sequence[str]
) -> WorkingFrame:
    # Spark SQL text works the same on PySpark and on sqlframe, with no function imports.
    text = f"coalesce(cast({_spark_quoted(source)} as string), '{NULL_KEY_TEXT}')"
    digest = f"sha2(concat('{prefix}', {text}), 256)"
    columns = []
    for position, name in enumerate(names):
        whole = _spark_hex_value(digest, position * _HEX_DIGITS_PER_UNIFORM + 1)
        scale = f"cast({_UNIFORM_SCALE} as double)"
        columns.append(f"({whole} + 0.5) / {scale} as {_spark_quoted(name)}")
    native = frame.to_native()
    expr = _spark_functions(native).expr
    return nw.from_native(native.select("*", *[expr(column) for column in columns]))


def _spark_hex_value(digest: str, start: int) -> str:
    # `conv` is not available on every Spark-like backend, so the 13 hex digits are summed
    # as digit * 16^place. Each term is a double below 2^53, so the sum is exact.
    # Spark reads a literal such as 16.0 as a decimal, and decimal division keeps only six
    # digits after the point, so every number is cast to double explicitly.
    terms = []
    for offset in range(_HEX_DIGITS_PER_UNIFORM):
        place = 16 ** (_HEX_DIGITS_PER_UNIFORM - 1 - offset)
        digit = f"(instr('0123456789abcdef', substring({digest}, {start + offset}, 1)) - 1)"
        terms.append(f"cast({digit} as double) * cast({place} as double)")
    return "(" + " + ".join(terms) + ")"


def _spark_functions(native: Any) -> Any:
    # sqlframe has one functions module per backend, next to its DataFrame class.
    module = type(native).__module__
    if module.split(".")[0] == "pyspark":
        return import_module("pyspark.sql.functions")
    return import_module(module.rsplit(".", 1)[0] + ".functions")


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _spark_quoted(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def lookup_frame(
    like: WorkingFrame,
    engine: Engine,
    columns: Mapping[str, Sequence[Any]],
    schema: Mapping[str, Any],
) -> WorkingFrame:
    """Build a small table on the same engine as `like`, so the two can be joined.

    `schema` gives each column's Narwhals type, so join keys match the data's type.
    """
    eager = nw.from_dict(dict(columns), schema=dict(schema), backend=nw.Implementation.POLARS)
    if engine is Engine.SPARK:
        return _spark_lookup(like, eager)
    if engine is not Engine.DUCKDB:
        return eager
    # A DuckDB relation only joins relations of its own connection. Querying through
    # the relation runs on that connection, and finds the local Arrow table by name.
    lookup_arrow = eager.to_arrow()  # noqa: F841 - read by name inside the query
    return nw.from_native(like.to_native().query("__base", "select * from lookup_arrow"))


_SPARK_TYPES = {
    nw.Boolean: "boolean",
    nw.String: "string",
    nw.Date: "date",
    nw.Datetime: "timestamp",
    nw.Float32: "double",
    nw.Float64: "double",
}


def _spark_lookup(like: WorkingFrame, eager: "nw.DataFrame[Any]") -> WorkingFrame:
    # The lookup is created on the user's own Spark session, so it can be joined to the data.
    native = like.to_native()
    session = getattr(native, "sparkSession", None) or native.session
    # Lookup column names are plain helper names, so the type string needs no quoting.
    fields = ", ".join(f"{name} {_spark_type(dtype)}" for name, dtype in eager.schema.items())
    return nw.from_native(session.createDataFrame(eager.rows(), schema=fields))


def _spark_type(dtype: Any) -> str:
    if dtype.is_integer():
        return "bigint"
    for narwhals_type, spark_type in _SPARK_TYPES.items():
        if dtype == narwhals_type:
            return spark_type
    raise TypeError(f"a Spark lookup cannot hold a column of type {dtype}")
