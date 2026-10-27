import duckdb
import pandas as pd
import polars as pl

from data_anonymizer import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, Protection


def _plan(protection=Protection.ANONYMIZATION, **fields) -> AnonymizationPlan:
    return AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=protection,
        actions=[ColumnAction(column="secret", technique="suppress")],
        **fields,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"secret": list("abcdefghij") * 5, "n": range(50)},
        index=[f"row-{number}" for number in range(50)],
    )


def test_anonymized_rows_come_back_shuffled_with_a_fresh_index():
    frame = _frame()

    first = Anonymizer().apply(frame, _plan()).data
    second = Anonymizer().apply(frame, _plan()).data

    assert sorted(first["n"]) == list(range(50))
    assert list(first["n"]) != list(frame["n"])
    assert list(first["n"]) != list(second["n"])
    assert list(first.index) == list(range(50))


def test_pseudonymized_rows_keep_their_order_and_index():
    frame = _frame()
    plan = _plan(Protection.PSEUDONYMIZATION)

    released = Anonymizer().apply(frame, plan).data

    assert list(released["n"]) == list(frame["n"])
    assert list(released.index) == list(frame.index)


def test_shuffle_rows_overrides_the_protection_default():
    frame = _frame()

    kept = Anonymizer().apply(frame, _plan(shuffle_rows=False)).data
    moved = Anonymizer().apply(frame, _plan(Protection.PSEUDONYMIZATION, shuffle_rows=True)).data

    assert list(kept["n"]) == list(frame["n"])
    assert list(moved["n"]) != list(frame["n"])


def test_untouched_columns_move_with_the_treated_ones():
    frame = pd.DataFrame({"a": range(40), "b": [str(n) for n in range(40)], "c": range(40)})
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[ColumnAction(column="c", technique="round_numeric", params={"decimals": 0})],
    )

    released = Anonymizer().apply(frame, plan).data

    assert (released["a"].astype(str) == released["b"]).all()
    assert (released["a"] == released["c"]).all()


def test_polars_and_duckdb_tables_are_shuffled_too():
    frame = _frame().reset_index(drop=True)
    plan = _plan(row_key="n")

    polars_rows = Anonymizer().apply(pl.from_pandas(frame), plan).data["n"].to_list()
    relation = duckdb.sql("select * from frame")
    duck_rows = Anonymizer().apply(relation, plan).data.df()["n"].to_list()

    assert polars_rows != list(range(50))
    assert duck_rows != list(range(50))
    assert sorted(duck_rows) == list(range(50))
