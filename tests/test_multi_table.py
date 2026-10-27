import json

import duckdb
import pandas as pd
import polars as pl
import pytest

from oblitus import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    MemoryVault,
    PlanError,
    Protection,
)
from oblitus.cli import main
from oblitus.engines import Engine

KEY = "multi-table-test-key-0123456789"
STORES = ["north", "south", "east", "west"]


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"c{number}" for number in range(1, 41)],
            "email": [f"person{number}@example.com" for number in range(1, 41)],
            "card": [f"4000-{number:04d}" for number in range(1, 41)],
            "signup": pd.Timestamp("2022-01-01") + pd.to_timedelta(range(40), unit="D"),
            "store": [STORES[number % 4] for number in range(40)],
        }
    )


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": range(1, 81),
            "customer_id": [f"c{1 + number % 40}" for number in range(80)],
            "email": [f"person{1 + number % 40}@example.com" for number in range(80)],
            "card": [f"4000-{1 + number % 40:04d}" for number in range(80)],
            "ordered": pd.Timestamp("2022-06-01") + pd.to_timedelta(range(80), unit="D"),
            "store": [STORES[number % 4] for number in range(80)],
        }
    )


def _plan(row_key: str, actions: list[ColumnAction], protection=Protection.PSEUDONYMIZATION):
    return AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL, protection=protection, actions=actions, row_key=row_key
    )


def _action(column: str, technique: str, **params) -> ColumnAction:
    return ColumnAction(column=column, technique=technique, params=params)


def _pseudonym_plans() -> dict[str, AnonymizationPlan]:
    return {
        "customers": _plan(
            "customer_id",
            [
                _action("email", "keyed_hash"),
                _action("card", "tokenize"),
                _action("signup", "date_shift", by="customer_id", stream="person", max_days=60),
            ],
        ),
        "orders": _plan(
            "order_id",
            [
                _action("email", "keyed_hash"),
                _action("card", "tokenize"),
                _action("ordered", "date_shift", by="customer_id", stream="person", max_days=60),
            ],
        ),
    }


def _native(frame: pd.DataFrame, engine: Engine):
    if engine is Engine.PANDAS:
        return frame
    if engine is Engine.POLARS:
        return pl.from_pandas(frame)
    return duckdb.connect().from_df(frame)


def _as_pandas(native) -> pd.DataFrame:
    if isinstance(native, pl.DataFrame):
        return native.to_pandas()
    if isinstance(native, duckdb.DuckDBPyRelation):
        return native.df()
    return native


def _apply_many(engine: Engine, plans=None, vault=None):
    frames = {
        "customers": _native(_customers(), engine),
        "orders": _native(_orders(), engine),
    }
    anonymizer = Anonymizer(secret_key=KEY)
    return anonymizer.apply_many(
        frames, plans or _pseudonym_plans(), vault=MemoryVault() if vault is None else vault
    )


ROW_KEYS = {"customers": "customer_id", "orders": "order_id"}
ORIGINALS = {"customers": _customers, "orders": _orders}


def _aligned(results, table: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The original and released rows of a table, in the same order."""
    key = ROW_KEYS[table]
    original = ORIGINALS[table]().sort_values(key).reset_index(drop=True)
    released = _as_pandas(results[table].data).sort_values(key).reset_index(drop=True)
    return original, released


def _day_shifts(results, table: str, column: str) -> pd.Series:
    original, released = _aligned(results, table)
    days = (pd.to_datetime(released[column]) - pd.to_datetime(original[column])).dt.days
    return pd.Series(days.to_numpy(), index=original["customer_id"].to_numpy())


@pytest.mark.parametrize("engine", list(Engine))
def test_the_same_person_gets_the_same_pseudonym_token_and_date_offset_in_every_table(engine):
    results = _apply_many(engine)

    _, customers = _aligned(results, "customers")
    _, orders = _aligned(results, "orders")
    for column in ("email", "card"):
        per_customer = customers.set_index("customer_id")[column]
        assert orders[column].tolist() == orders["customer_id"].map(per_customer).tolist()
        assert customers[column].nunique() == 40

    order_shifts = _day_shifts(results, "orders", "ordered")
    assert order_shifts.groupby(level=0).nunique().eq(1).all()
    signup_shifts = _day_shifts(results, "customers", "signup")
    assert order_shifts.groupby(level=0).first().loc[signup_shifts.index].equals(signup_shifts)


def test_without_a_shared_stream_different_date_columns_move_independently():
    plans = _pseudonym_plans()
    plans["customers"] = _plan(
        "customer_id", [_action("signup", "date_shift", by="customer_id", max_days=60)]
    )
    plans["orders"] = _plan(
        "order_id", [_action("ordered", "date_shift", by="customer_id", max_days=60)]
    )

    results = _apply_many(Engine.PANDAS, plans)

    signup_shifts = _day_shifts(results, "customers", "signup")
    order_shifts = _day_shifts(results, "orders", "ordered").groupby(level=0).first()
    assert not order_shifts.loc[signup_shifts.index].equals(signup_shifts)


def test_the_tokens_of_all_tables_live_in_one_vault():
    vault = MemoryVault()

    results = _apply_many(Engine.PANDAS, vault=vault)
    anonymizer = Anonymizer(secret_key=KEY)
    revealed = anonymizer.reveal(results["orders"].data, _pseudonym_plans()["orders"], vault=vault)

    assert revealed["card"].tolist() == _orders()["card"].tolist()
    assert len(vault) == 40


def test_anonymized_tables_of_one_call_share_codes_and_differ_from_the_next_call():
    actions = [_action("store", "scramble_categories", prefix="S")]
    plans = {
        "customers": _plan("customer_id", actions, Protection.ANONYMIZATION),
        "orders": _plan("order_id", actions, Protection.ANONYMIZATION),
    }
    frames = {"customers": _customers(), "orders": _orders()}
    # Fixed throwaway keys keep the test deterministic: with four stores, two fresh random
    # orders of the codes coincide once in 24 runs.
    throwaway_keys = iter([b"first-call-throwaway-key-000000", b"second-call-throwaway-key-11111"])
    anonymizer = Anonymizer(secret_key=KEY, throwaway_key_source=lambda: next(throwaway_keys))

    first = anonymizer.apply_many(frames, plans)
    second = anonymizer.apply_many(frames, plans)

    codes_first = _codes_by_store(first)
    codes_second = _codes_by_store(second)
    assert codes_first["customers"] == codes_first["orders"]
    assert codes_first["customers"] != codes_second["customers"]


def _codes_by_store(results) -> dict[str, dict[str, set[str]]]:
    """For each table, the codes that each original store turned into."""
    codes = {}
    for table in ("customers", "orders"):
        original, released = _aligned(results, table)
        codes[table] = {
            store: set(released.loc[original["store"] == store, "store"]) for store in STORES
        }
    return codes


def test_protection_is_assessed_for_each_table():
    plans = _pseudonym_plans()

    results = _apply_many(Engine.PANDAS, plans)

    assert set(results) == {"customers", "orders"}
    assert results["customers"].plan is plans["customers"]
    assert results["orders"].plan is plans["orders"]
    for result in results.values():
        assert result.protection.level.value == "pseudonymized"


def test_a_plan_that_keeps_a_link_blocks_the_whole_anonymizing_call_before_any_table_runs():
    plans = _pseudonym_plans()
    plans["orders"] = _plan(
        "order_id", [_action("card", "tokenize")], protection=Protection.ANONYMIZATION
    )

    with pytest.raises(PlanError, match="table 'orders'.*keeps a link"):
        _apply_many(Engine.PANDAS, plans)


def test_tables_and_plans_must_match():
    plans = _pseudonym_plans()
    del plans["orders"]

    with pytest.raises(PlanError, match="tables without a plan: \\['orders'\\]"):
        _apply_many(Engine.PANDAS, plans)


def test_every_problem_is_listed_with_its_table_name():
    plans = _pseudonym_plans()
    plans["customers"] = _plan("customer_id", [_action("missing", "keyed_hash")])
    plans["orders"] = _plan("order_id", [_action("other_missing", "keyed_hash")])

    with pytest.raises(PlanError) as problem:
        _apply_many(Engine.PANDAS, plans)

    assert "table 'customers'" in str(problem.value)
    assert "table 'orders'" in str(problem.value)


def _write_manifest(folder, plans: dict[str, AnonymizationPlan]) -> None:
    tables = {}
    for name, frame in (("customers", _customers()), ("orders", _orders())):
        frame.to_csv(folder / f"{name}.csv", index=False)
        plans[name].save(folder / f"{name}_plan.json")
        tables[name] = {
            "input": f"{name}.csv",
            "output": f"released_{name}.csv",
            "plan": f"{name}_plan.json",
        }
    (folder / "manifest.json").write_text(json.dumps({"tables": tables}))


def test_cli_apply_many_protects_every_table_in_the_manifest_with_one_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", KEY)
    _write_manifest(tmp_path, _pseudonym_plans())
    report = tmp_path / "report.json"
    vault = tmp_path / "vault.sqlite"

    exit_code = main(
        ["apply-many", str(tmp_path / "manifest.json"), "--vault", str(vault)]
        + ["--report", str(report)]
    )

    customers = pd.read_csv(tmp_path / "released_customers.csv")
    orders = pd.read_csv(tmp_path / "released_orders.csv")
    assert exit_code == 0
    assert set(customers["card"]) == set(orders["card"])
    assert len(customers["card"].unique()) == 40
    assert set(json.loads(report.read_text())) == {"customers", "orders"}


def test_cli_apply_many_proposes_a_plan_for_a_table_without_one(tmp_path, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", KEY)
    _customers().to_csv(tmp_path / "customers.csv", index=False)
    manifest = {"tables": {"customers": {"input": "customers.csv", "output": "out.csv"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    exit_code = main(["apply-many", str(tmp_path / "manifest.json")])

    assert exit_code in (0, 3)
    assert (tmp_path / "out.csv").exists()


def test_cli_apply_many_rejects_a_manifest_with_no_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", KEY)
    (tmp_path / "manifest.json").write_text(json.dumps({"tables": {}}))

    exit_code = main(["apply-many", str(tmp_path / "manifest.json")])

    assert exit_code == 2
    assert "error" in capsys.readouterr().err
