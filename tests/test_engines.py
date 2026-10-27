import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from data_anonymizer.anonymizer import Anonymizer
from data_anonymizer.engines import Engine
from data_anonymizer.errors import PlanError
from data_anonymizer.evaluation import assess_privacy
from data_anonymizer.evaluation.aligned_sample import align_sample
from data_anonymizer.local_suppression import enforce_k_anonymity
from data_anonymizer.noise_explorer import explore_noise
from data_anonymizer.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from data_anonymizer.table import Table
from data_anonymizer.techniques import BUILTIN_TECHNIQUES

# Techniques that map each value on its own: masking, pseudonymization, encryption.
MAPPING_TECHNIQUES = [
    ("keyed_hash", "email", {}),
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


# The public edges: a whole plan applied to a table of each engine.


def _ledger() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = 120
    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "email": [f"person{index % 90}@example.com" for index in range(rows)],
            "store": rng.choice(["a", "b", "c"], rows),
            "segment": rng.choice(["retail", "online"], rows),
            "age": rng.integers(18, 80, rows),
            "amount": rng.lognormal(5, 1, rows).round(2),
        }
    )


def _migrated_plan() -> AnonymizationPlan:
    return AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.PSEUDONYMIZATION,
        actions=[
            ColumnAction(column="email", technique="keyed_hash"),
            ColumnAction(column="age", technique="gaussian_noise", params={"relative_scale": 0.1}),
            ColumnAction(column="amount", technique="normalize", params={"method": "robust"}),
        ],
        direct_identifiers=["email"],
        quasi_identifiers=["store", "segment"],
        sensitive_attributes=["amount"],
        row_key="id",
    )


def _as_pandas(native) -> pd.DataFrame:
    if isinstance(native, duckdb.DuckDBPyRelation):
        native = native.df()
    elif isinstance(native, pl.DataFrame):
        native = native.to_pandas()
    return native.sort_values("id").reset_index(drop=True)


def _anonymizer() -> Anonymizer:
    return Anonymizer(secret_key=b"edges-secret-key-0123456789", seed=3)


@pytest.mark.parametrize("engine", list(Engine))
def test_a_plan_with_a_row_key_returns_the_kind_of_table_it_was_given(engine):
    native = _native(_ledger(), engine)

    result = _anonymizer().apply(native, _migrated_plan())

    assert isinstance(result.data, type(native))


def test_every_engine_reaches_the_same_protection_level_and_problems():
    results = {
        engine: _anonymizer().apply(_native(_ledger(), engine), _migrated_plan())
        for engine in Engine
    }

    reference = results[Engine.PANDAS]
    for result in results.values():
        assert result.protection.level == reference.protection.level
        assert result.protection.problems == reference.protection.problems
        assert result.privacy_before == reference.privacy_before
        assert result.privacy_after == reference.privacy_after


def test_every_engine_releases_the_same_values_for_the_migrated_techniques():
    released = {
        engine: _as_pandas(_anonymizer().apply(_native(_ledger(), engine), _migrated_plan()).data)
        for engine in Engine
    }

    reference = released[Engine.POLARS]
    for engine in (Engine.PANDAS, Engine.DUCKDB):
        for column in ("email", "age", "amount"):
            other = released[engine][column]
            if pd.api.types.is_numeric_dtype(reference[column]):
                np.testing.assert_allclose(other, reference[column], rtol=1e-9)
            else:
                assert other.tolist() == reference[column].tolist()


@pytest.mark.parametrize("engine", [Engine.PANDAS, Engine.POLARS])
def test_a_proposed_plan_runs_and_agrees_between_in_memory_engines(engine, people):
    anonymizer = _anonymizer()
    native = _native(people, engine)

    plan = anonymizer.propose(native)
    result = anonymizer.apply(native, plan)

    pandas_plan = anonymizer.propose(people)
    pandas_result = anonymizer.apply(people, pandas_plan)
    assert plan == pandas_plan
    assert plan.row_key == "customer_id"
    assert isinstance(result.data, type(native))
    assert result.protection.level == pandas_result.protection.level
    assert result.protection.problems == pandas_result.protection.problems


def test_a_duckdb_table_needs_the_plans_row_key():
    plan = _migrated_plan().model_copy(update={"row_key": None})

    with pytest.raises(PlanError, match="row_key"):
        _anonymizer().apply(_native(_ledger(), Engine.DUCKDB), plan)


def test_a_row_key_that_is_not_in_the_table_is_refused():
    plan = _migrated_plan().model_copy(update={"row_key": "missing"})

    with pytest.raises(PlanError, match="row_key"):
        _anonymizer().apply(_ledger(), plan)


@pytest.mark.parametrize("engine", list(Engine))
def test_the_aligned_sample_pairs_each_original_row_with_its_released_row(engine, context):
    table = Table.from_native(_native(_ledger(), engine), row_key="id")
    BUILTIN_TECHNIQUES["normalize"].apply(table, "amount", context, {})

    sample = align_sample(
        table.original_frame(), table.released_frame(), engine, key=b"sample-key", max_rows=50
    )

    assert len(sample.original) == len(sample.released) == 50
    assert sample.original["id"].tolist() == sample.released["id"].tolist()
    assert not np.allclose(sample.original["amount"], sample.released["amount"])
    expected = _ledger().set_index("id").loc[sample.original["id"], "amount"]
    np.testing.assert_allclose(sample.original["amount"], expected)


def test_the_aligned_sample_picks_the_same_rows_on_every_engine():
    picked = {}
    for engine in Engine:
        table = Table.from_native(_native(_ledger(), engine), row_key="id")
        sample = align_sample(
            table.original_frame(), table.released_frame(), engine, key=b"sample-key", max_rows=50
        )
        picked[engine] = sorted(sample.original["id"])

    assert picked[Engine.PANDAS] == picked[Engine.POLARS] == picked[Engine.DUCKDB]


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_scan_finds_the_same_categories_on_every_engine(engine, people):
    expected = [(f.column, f.category) for f in _anonymizer().scan(people)]

    found = _anonymizer().scan(_native(people, engine))

    assert [(f.column, f.category) for f in found] == expected


@pytest.mark.parametrize("engine", list(Engine))
def test_privacy_is_measured_the_same_on_every_engine(engine):
    frame = _ledger()
    frame.loc[[3, 9], "segment"] = None
    expected = assess_privacy(frame, ["store", "segment"], ["age"], k_target=30)

    report = assess_privacy(_native(frame, engine), ["store", "segment"], ["age"], k_target=30)

    assert report == expected
    assert report.rows_below_target > 0


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_suppression_blanks_the_same_cells_on_every_engine(engine):
    frame = _ledger()
    expected = enforce_k_anonymity(frame, ["store", "age"], k=3)

    outcome = enforce_k_anonymity(_native(frame, engine), ["store", "age"], k=3)

    assert outcome.suppressed_cells == expected.suppressed_cells > 0
    released = outcome.data.df() if engine is Engine.DUCKDB else outcome.data.to_pandas()
    released = released.sort_values("id").reset_index(drop=True)
    assert released["age"].isna().tolist() == expected.data["age"].isna().tolist()


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_the_noise_explorer_accepts_any_engine(engine):
    frame = _ledger()

    exploration = explore_noise(_native(frame, engine), "amount", max_rows=60)

    assert exploration.rows == 60
    assert exploration.total_rows == len(frame)
