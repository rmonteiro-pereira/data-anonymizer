import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from data_anonymizer.engines import Engine
from data_anonymizer.table import Table
from data_anonymizer.techniques import BUILTIN_TECHNIQUES

ENGINE_TECHNIQUES = [
    ("gaussian_noise", "amount", {"relative_scale": 0.2}),
    ("microaggregate", "amount", {"group_size": 4}),
    ("keyed_hash", "email", {}),
    ("normalize", "amount", {}),
    (
        "normalize",
        "amount",
        {"method": "index", "by": "store", "allow_shape_change": True},
    ),
    (
        "normalize",
        "amount",
        {"method": "growth", "by": "store", "order_by": "id", "allow_shape_change": True},
    ),
]


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "id": np.arange(1, 41),
            "store": rng.choice(["a", "b", "c"], 40),
            "email": [f"person{index % 25}@example.com" for index in range(40)],
            "amount": rng.lognormal(5, 1, 40).round(2),
        }
    )


def _native(frame: pd.DataFrame, engine: Engine):
    if engine is Engine.PANDAS:
        return frame
    if engine is Engine.POLARS:
        return pl.from_pandas(frame)
    connection = duckdb.connect()
    return connection.from_df(frame)


def _released(engine: Engine, name: str, column: str, params: dict, context) -> pd.DataFrame:
    table = Table.from_native(_native(_sample(), engine), row_key="id")
    BUILTIN_TECHNIQUES[name].apply(table, column, context, params)
    native = table.to_native()
    if engine is Engine.DUCKDB:
        native = native.df()
    elif engine is Engine.POLARS:
        native = native.to_pandas()
    return native.sort_values("id").reset_index(drop=True)


@pytest.mark.parametrize(("name", "column", "params"), ENGINE_TECHNIQUES)
def test_every_engine_releases_the_same_values(context, name, column, params):
    released = {engine: _released(engine, name, column, params, context) for engine in Engine}

    reference = released[Engine.POLARS][column]
    for engine in (Engine.PANDAS, Engine.DUCKDB):
        other = released[engine][column]
        if pd.api.types.is_numeric_dtype(reference):
            np.testing.assert_allclose(other, reference, rtol=1e-9, equal_nan=True)
        else:
            assert other.tolist() == reference.tolist()


def test_the_table_comes_back_as_the_kind_it_was_given(context):
    for engine, kind in [
        (Engine.PANDAS, pd.DataFrame),
        (Engine.POLARS, pl.DataFrame),
        (Engine.DUCKDB, duckdb.DuckDBPyRelation),
    ]:
        table = Table.from_native(_native(_sample(), engine), row_key="id")
        BUILTIN_TECHNIQUES["keyed_hash"].apply(table, "email", context, {})

        native = table.to_native()

        assert isinstance(native, kind)
        assert list(native.columns) == ["id", "store", "email", "amount"]


def test_noise_depends_on_the_randomness_key(context):
    other = context.__class__(
        context.rng, context.secret_key, context.surrogates, randomness_key=b"another key"
    )

    first = _released(Engine.POLARS, "gaussian_noise", "amount", {}, context)
    second = _released(Engine.POLARS, "gaussian_noise", "amount", {}, other)

    assert not np.allclose(first["amount"], second["amount"])


def test_lazy_tables_need_a_row_key_for_noise(context):
    table = Table.from_native(duckdb.connect().from_df(_sample()))

    with pytest.raises(ValueError, match="row key"):
        BUILTIN_TECHNIQUES["gaussian_noise"].apply(table, "amount", context, {})
