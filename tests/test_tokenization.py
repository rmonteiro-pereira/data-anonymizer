import re
import sqlite3
import stat

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
    SqliteVault,
)
from oblitus.cli import main
from oblitus.engines import Engine
from oblitus.surrogates import SurrogateFactory
from oblitus.table import Table
from oblitus.techniques import BUILTIN_TECHNIQUES, TechniqueContext

KEY = "tokenization-test-key-0123456789"


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "card": ["4111-1111", "4222-2222", "4111-1111", None, "4333-3333", "4222-2222"],
            "name": ["Ana", "Bia", "Ana", "Caio", "Dani", "Bia"],
        }
    )


def _plan(params: dict | None = None, protection=Protection.PSEUDONYMIZATION):
    action = ColumnAction(column="card", technique="tokenize", params=params or {})
    return AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL, protection=protection, actions=[action], row_key="id"
    )


def _context(vault) -> TechniqueContext:
    return TechniqueContext(
        secret_key=b"k", surrogates=SurrogateFactory("en_US"), randomness_key=b"r", vault=vault
    )


def test_the_same_value_gets_the_same_token_and_gaps_stay_gaps():
    released = Anonymizer(secret_key=KEY).apply(_customers(), _plan(), vault=MemoryVault()).data

    tokens = released["card"]
    assert tokens[0] == tokens[2]
    assert tokens[1] == tokens[5]
    assert tokens[0] != tokens[1]
    assert pd.isna(tokens[3])
    assert not tokens.dropna().isin(_customers()["card"]).any()


def test_a_token_is_random_letters_and_digits_of_the_requested_length_with_a_prefix():
    plan = _plan({"prefix": "tok_", "length": 12})

    released = Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=MemoryVault()).data

    assert released["card"].dropna().str.fullmatch(r"tok_[A-Za-z0-9]{12}").all()


def test_tokens_do_not_repeat_between_vaults_whatever_the_key_and_seed():
    first = Anonymizer(secret_key=KEY, seed=1).apply(_customers(), _plan(), vault=MemoryVault())
    second = Anonymizer(secret_key=KEY, seed=1).apply(_customers(), _plan(), vault=MemoryVault())

    assert not first.data["card"].dropna().isin(second.data["card"]).any()


def test_format_preserving_tokens_keep_length_and_character_classes():
    plan = _plan({"preserve_format": True})

    released = Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=MemoryVault()).data

    assert released["card"].dropna().str.fullmatch(r"\d{4}-\d{4}").all()
    assert not released["card"].dropna().isin(_customers()["card"]).any()


def test_format_preserving_tokens_keep_letter_case():
    frame = pd.DataFrame({"id": [1], "card": ["Ab-9x"]})
    plan = _plan({"preserve_format": True})

    released = Anonymizer(secret_key=KEY).apply(frame, plan, vault=MemoryVault())

    assert re.fullmatch(r"[A-Z][a-z]-\d[a-z]", released.data["card"][0])


def test_reveal_restores_the_values_from_the_vault():
    anonymizer = Anonymizer(secret_key=KEY)
    plan = _plan()
    vault = MemoryVault()

    released = anonymizer.apply(_customers(), plan, vault=vault).data
    revealed = anonymizer.reveal(released, plan, vault=vault)

    assert revealed["card"].tolist()[:3] == ["4111-1111", "4222-2222", "4111-1111"]
    assert pd.isna(revealed["card"][3])


def test_the_vault_not_the_key_is_what_reveals_a_token():
    plan = _plan()
    released = Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=MemoryVault()).data

    with pytest.raises(PlanError, match="not a token in this vault"):
        Anonymizer(secret_key=KEY).reveal(released, plan, vault=MemoryVault())


def test_a_plan_that_tokenizes_needs_a_vault():
    with pytest.raises(PlanError, match="needs a vault"):
        Anonymizer(secret_key=KEY).apply(_customers(), _plan())


def test_anonymization_refuses_tokenization_because_it_keeps_a_link():
    plan = _plan(protection=Protection.ANONYMIZATION)

    with pytest.raises(PlanError, match="keeps a link"):
        Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=MemoryVault())


def test_the_plan_stays_json_and_names_no_vault():
    plan = _plan({"prefix": "t_"})

    assert AnonymizationPlan.from_json(plan.to_json()) == plan
    assert "vault" not in plan.to_json()


def test_a_namespace_gives_the_same_value_a_different_token():
    frame = pd.DataFrame({"id": [1, 2], "card": ["x1", "x1"], "name": ["x1", "x1"]})
    actions = [
        ColumnAction(column="card", technique="tokenize", params={"namespace": "card"}),
        ColumnAction(column="name", technique="tokenize", params={"namespace": "name"}),
    ]
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.PSEUDONYMIZATION,
        actions=actions,
        row_key="id",
    )

    released = Anonymizer(secret_key=KEY).apply(frame, plan, vault=MemoryVault()).data

    assert released["card"][0] != released["name"][0]


def test_a_vault_with_no_unused_token_left_fails_with_a_clear_error():
    digits = [str(number) for number in range(10)]
    frame = pd.DataFrame({"id": range(1, 11), "card": digits, "name": digits})
    actions = [
        ColumnAction(
            column=column,
            technique="tokenize",
            params={"preserve_format": True, "namespace": column},
        )
        for column in ("card", "name")
    ]
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        protection=Protection.PSEUDONYMIZATION,
        actions=actions,
        row_key="id",
    )

    with pytest.raises(PlanError, match="unused token"):
        Anonymizer(secret_key=KEY).apply(frame, plan, vault=MemoryVault())


def test_a_sqlite_vault_keeps_tokens_between_runs_and_is_private(tmp_path):
    path = tmp_path / "vault.sqlite"
    plan = _plan()
    with SqliteVault(path) as vault:
        first = Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=vault).data
    with SqliteVault(path) as vault:
        second = Anonymizer(secret_key=KEY).apply(_customers(), plan, vault=vault).data
        revealed = Anonymizer(secret_key=KEY).reveal(first, plan, vault=vault)

    assert first["card"].equals(second["card"])
    assert revealed["card"][1] == "4222-2222"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    stored = sqlite3.connect(path).execute("SELECT count(*) FROM tokens").fetchone()[0]
    assert stored == 3


def test_a_sqlite_vault_refuses_a_second_token_for_one_value_or_one_token_twice(tmp_path):
    with SqliteVault(tmp_path / "vault.sqlite") as vault:
        assert vault.add("s", "value", "token")
        assert not vault.add("s", "value", "other")
        assert not vault.add("s", "another", "token")
        assert vault.value_for("token") == "value"
        assert vault.token_for("s", "value") == "token"
        assert vault.value_for("missing") is None


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


@pytest.mark.parametrize("engine", list(Engine))
@pytest.mark.parametrize("params", [{}, {"preserve_format": True, "prefix": "T"}])
def test_tokens_round_trip_on_every_engine(engine, params):
    context = _context(MemoryVault())
    technique = BUILTIN_TECHNIQUES["tokenize"]
    table = Table.from_native(_native(_customers(), engine), row_key="id")

    technique.apply(table, "card", context, params)
    tokens = _as_pandas(table.to_native()).sort_values("id").reset_index(drop=True)["card"]
    technique.reverse(table, "card", context, params)
    restored = _as_pandas(table.to_native()).sort_values("id").reset_index(drop=True)["card"]

    assert tokens[0] == tokens[2]
    assert pd.isna(tokens[3])
    assert tokens.dropna().tolist() != _customers()["card"].dropna().tolist()
    assert restored.tolist()[:3] == ["4111-1111", "4222-2222", "4111-1111"]
    assert pd.isna(restored[3])


def test_cli_apply_and_reveal_use_the_vault_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", KEY)
    source = tmp_path / "customers.csv"
    _customers().to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(_plan().to_json())
    vault_path = tmp_path / "vault.sqlite"
    released = tmp_path / "released.csv"
    revealed = tmp_path / "revealed.csv"

    apply_arguments = ["apply", str(source), "--plan", str(plan_path), "-o", str(released)]
    main(apply_arguments + ["--vault", str(vault_path)])
    reveal_arguments = ["reveal", str(released), "--plan", str(plan_path), "-o", str(revealed)]
    assert main(reveal_arguments + ["--vault", str(vault_path)]) == 0

    assert pd.read_csv(revealed)["card"][1] == "4222-2222"


def test_cli_apply_without_a_vault_reports_the_missing_vault(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", KEY)
    source = tmp_path / "customers.csv"
    _customers().to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(_plan().to_json())
    output = tmp_path / "out.csv"

    exit_code = main(["apply", str(source), "--plan", str(plan_path), "-o", str(output)])

    assert exit_code == 2
    assert "needs a vault" in capsys.readouterr().err
