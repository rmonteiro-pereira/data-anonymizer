import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from data_anonymizer.engines import Engine
from data_anonymizer.table import Table
from data_anonymizer.techniques import BUILTIN_TECHNIQUES

# Techniques that map each value on its own: masking, pseudonymization, encryption.
MAPPING_TECHNIQUES = [
    ("keep", "email", {}),
    ("redact", "email", {}),
    ("suppress", "email", {}),
    ("partial_mask", "email", {"keep_last": 3}),
    ("redact_entities", "email", {}),
    ("keyed_hash", "email", {}),
    ("scramble_characters", "email", {}),
    ("surrogate", "email", {"category": "email"}),
    ("surrogate", "email", {"category": "email", "consistent": False}),
    ("argon2_hash", "email", {"memory_cost": 1024, "time_cost": 1}),
    ("bcrypt_hash", "email", {"rounds": 4}),
    ("scramble_categories", "store", {"prefix": "S"}),
    ("scramble_categories", "store", {"min_count": 14, "keyed": True}),
    ("encrypt", "email", {}),
    ("format_preserving_encrypt", "email", {"alphabet": "alphanumeric", "tweak": "t"}),
]

# Techniques built from expressions, windows and keyed noise.
EXPRESSION_TECHNIQUES = [
    ("gaussian_noise", "amount", {"relative_scale": 0.2}),
    ("microaggregate", "amount", {"group_size": 4}),
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

ENGINE_TECHNIQUES = MAPPING_TECHNIQUES + EXPRESSION_TECHNIQUES


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
            assert _gaps_as_none(other) == _gaps_as_none(reference)


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


@pytest.mark.parametrize("engine", list(Engine))
def test_reordered_values_keep_the_column_and_its_gaps(context, engine):
    frame = _sample()
    frame.loc[[3, 17], "amount"] = None
    table = Table.from_native(_native(frame, engine), row_key="id")
    (sort_key,) = table.uniforms(context.stream_key("amount", "test"))

    table.replace("amount", table.reorder_values("amount", sort_key))
    native = table.to_native()
    released = native.df() if engine is Engine.DUCKDB else native
    if engine is Engine.POLARS:
        released = released.to_pandas()
    released = released.sort_values("id").reset_index(drop=True)

    assert released["amount"].isna().tolist() == frame["amount"].isna().tolist()
    assert sorted(released["amount"].dropna()) == sorted(frame["amount"].dropna())
    assert not released["amount"].equals(frame["amount"])


@pytest.mark.parametrize("engine", list(Engine))
@pytest.mark.parametrize(
    ("technique", "params"),
    [
        ("encrypt", {"deterministic": False}),
        ("encrypt", {"deterministic": True}),
        ("format_preserving_encrypt", {"alphabet": "alphanumeric"}),
    ],
)
def test_reversible_techniques_restore_the_original_on_every_engine(
    context, engine, technique, params
):
    frame = _sample()
    frame.loc[[2, 9], "email"] = None
    table = Table.from_native(_native(frame, engine), row_key="id")

    BUILTIN_TECHNIQUES[technique].apply(table, "email", context, params)
    BUILTIN_TECHNIQUES[technique].reverse(table, "email", context, params)

    released = _as_pandas(table.to_native(), engine).sort_values("id").reset_index(drop=True)
    assert _gaps_as_none(released["email"]) == _gaps_as_none(frame["email"])


@pytest.mark.parametrize("engine", list(Engine))
@pytest.mark.parametrize(
    ("technique", "params"),
    [
        ("encrypt", {"deterministic": False}),
        ("argon2_hash", {"memory_cost": 1024, "time_cost": 1, "deterministic": False}),
        ("bcrypt_hash", {"rounds": 4, "deterministic": False}),
    ],
)
def test_randomized_techniques_give_equal_values_different_results(
    context, engine, technique, params
):
    table = Table.from_native(_native(_sample(), engine), row_key="id")

    BUILTIN_TECHNIQUES[technique].apply(table, "email", context, params)

    released = _as_pandas(table.to_native(), engine)
    assert released["email"].nunique() == len(released)


@pytest.mark.parametrize("engine", list(Engine))
def test_random_surrogates_keep_gaps_and_do_not_link_equal_values(context, engine):
    frame = _sample()
    frame.loc[[4, 20], "email"] = None
    table = Table.from_native(_native(frame, engine), row_key="id")
    params = {"category": "email", "consistent": False}

    BUILTIN_TECHNIQUES["surrogate"].apply(table, "email", context, params)

    released = _as_pandas(table.to_native(), engine).sort_values("id")
    assert released["email"].isna().tolist() == frame["email"].isna().tolist()
    present_rows = frame["email"].notna().sum()
    assert released["email"].nunique() > present_rows / 2


def _as_pandas(native, engine: Engine) -> pd.DataFrame:
    if engine is Engine.DUCKDB:
        return native.df()
    if engine is Engine.POLARS:
        return native.to_pandas()
    return native


def _gaps_as_none(column: pd.Series) -> list:
    # Engines hand back missing text as None or NaN; both mean the same here.
    return [None if pd.isna(item) else item for item in column]
