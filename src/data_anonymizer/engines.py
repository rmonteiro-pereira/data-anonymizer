"""What differs between the dataframe engines a table can live in.

Techniques are written once, as Narwhals expressions, and run on every engine.
Three things cannot be written that way, and live here instead:
- converting the user's frame in and out;
- turning a key column into keyed random numbers, which needs SHA-256;
- building a small lookup table on the same engine, and the same connection, as the data.

pandas and Polars frames are processed in memory with Polars. DuckDB relations stay
lazy: the work becomes SQL that DuckDB runs.
"""

import hashlib
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import narwhals as nw
import polars as pl

# Each SHA-256 digest yields four uniform numbers of 52 bits, one per 13 hex digits.
UNIFORMS_PER_DIGEST = 4
_HEX_DIGITS_PER_UNIFORM = 13
_UNIFORM_SCALE = float(16**_HEX_DIGITS_PER_UNIFORM)

# What a missing key hashes as, so missing entities still share one random number.
NULL_KEY_TEXT = "<null>"

WorkingFrame = nw.DataFrame[Any] | nw.LazyFrame[Any]


class Engine(StrEnum):
    """Where the user's table lives, and so the kind of frame they get back."""

    PANDAS = "pandas"
    POLARS = "polars"
    DUCKDB = "duckdb"

    @property
    def is_lazy(self) -> bool:
        """True when the table is a query the engine runs, not rows held in memory."""
        return self is Engine.DUCKDB


def engine_of(native: Any) -> Engine:
    """Name the engine of a user's frame, or raise TypeError for an unsupported one."""
    module = type(native).__module__.split(".")[0]
    if module == "pandas":
        return Engine.PANDAS
    if module == "polars":
        return Engine.POLARS
    if module in ("duckdb", "_duckdb"):
        return Engine.DUCKDB
    raise TypeError(f"tables must be pandas, Polars or DuckDB frames, not {type(native).__name__}")


def to_working_frame(native: Any) -> tuple[WorkingFrame, Engine]:
    """Wrap a user's frame for processing, and say which engine it came from."""
    engine = engine_of(native)
    if engine is Engine.PANDAS:
        return nw.from_native(pl.from_pandas(native), eager_only=True), engine
    if engine is Engine.POLARS and isinstance(native, pl.LazyFrame):
        return nw.from_native(native.collect(), eager_only=True), engine
    return nw.from_native(native), engine


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


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


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
    if engine is not Engine.DUCKDB:
        return eager
    # A DuckDB relation only joins relations of its own connection. Querying through
    # the relation runs on that connection, and finds the local Arrow table by name.
    lookup_arrow = eager.to_arrow()  # noqa: F841 - read by name inside the query
    return nw.from_native(like.to_native().query("__base", "select * from lookup_arrow"))
