import json

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from oblitus.anonymizer import Anonymizer
from oblitus.chunked_apply import chunk_blockers
from oblitus.cli import main
from oblitus.engines import Engine
from oblitus.errors import PlanError, TableFormatError
from oblitus.evaluation.aligned_sample import SampleAccumulator, align_sample
from oblitus.plan import AnonymizationPlan, ColumnAction, Fidelity, Protection
from oblitus.table import ROW_KEY_COLUMN
from oblitus.table_files import iter_table_chunks, read_table, write_table

ROWS = 1_050
CHUNK_ROWS = 200


def _ledger() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "id": np.arange(1, ROWS + 1),
            "email": [f"person{index % 300}@example.com" for index in range(ROWS)],
            "note": [
                f'says "hi", twice\nline {index}' if index % 50 == 0 else "ok"
                for index in range(ROWS)
            ],
            "store": rng.choice(["a", "b", "c"], ROWS),
            "age": rng.integers(18, 90, ROWS),
            "amount": rng.lognormal(5, 1, ROWS).round(2),
            "signed_up": pd.Timestamp("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 900, ROWS), "D"),
        }
    )


def _plan(**changes) -> AnonymizationPlan:
    fields = {
        "fidelity": Fidelity.STATISTICAL,
        "protection": Protection.PSEUDONYMIZATION,
        "actions": [
            ColumnAction(column="email", technique="keyed_hash"),
            ColumnAction(column="note", technique="redact_entities"),
            ColumnAction(column="age", technique="round_numeric", params={"base": 5}),
            ColumnAction(
                column="amount", technique="laplace_noise", params={"lower": 0, "upper": 2000}
            ),
            ColumnAction(column="signed_up", technique="date_shift", params={"max_days": 15}),
        ],
        "direct_identifiers": ["email"],
        "quasi_identifiers": ["store", "age"],
        "sensitive_attributes": ["amount"],
        "row_key": "id",
    }
    fields.update(changes)
    return AnonymizationPlan(**fields)


def _anonymizer() -> Anonymizer:
    return Anonymizer(secret_key=b"chunked-secret-key-0123456789", seed=4)


@pytest.fixture(params=[".csv", ".parquet"])
def source(request, tmp_path):
    path = tmp_path / f"ledger{request.param}"
    write_table(pl.from_pandas(_ledger()), path)
    return path


def test_chunks_cover_the_file_in_order_with_whole_file_types(source):
    chunks = list(iter_table_chunks(source, CHUNK_ROWS))

    whole = read_table(source)
    assert [len(chunk) for chunk in chunks] == [200, 200, 200, 200, 200, 50]
    assert pl.concat(chunks).equals(whole)


def test_a_chunked_run_releases_exactly_what_an_in_memory_run_releases(source, tmp_path):
    plan = _plan()
    in_memory = _anonymizer().apply(read_table(source), plan)
    destination = tmp_path / f"released{source.suffix}"

    chunked = _anonymizer().apply_file(source, destination, plan, chunk_rows=CHUNK_ROWS)

    assert read_table(destination).equals(in_memory.data)
    assert chunked.rows == ROWS
    assert chunked.chunks == 6
    assert chunked.protection.level == in_memory.protection.level
    assert chunked.protection.problems == in_memory.protection.problems
    assert chunked.privacy_before == in_memory.privacy_before
    assert chunked.privacy_after == in_memory.privacy_after
    assert chunked.utility == in_memory.utility


def test_the_chunk_size_does_not_change_the_result(source, tmp_path):
    plan = _plan()
    small = tmp_path / f"small{source.suffix}"
    large = tmp_path / f"large{source.suffix}"

    _anonymizer().apply_file(source, small, plan, chunk_rows=77)
    _anonymizer().apply_file(source, large, plan, chunk_rows=5_000)

    assert read_table(small).equals(read_table(large))


def test_a_released_file_below_the_target_is_not_reported_as_anonymized(source, tmp_path):
    plan = _plan(
        protection=Protection.ANONYMIZATION,
        shuffle_rows=False,
        actions=[
            ColumnAction(column="email", technique="suppress"),
            ColumnAction(column="note", technique="suppress"),
            ColumnAction(column="signed_up", technique="suppress"),
        ],
        k_target=2000,
    )

    result = _anonymizer().apply_file(source, tmp_path / f"out{source.suffix}", plan)

    assert result.protection.level.value == "at_risk"
    assert not result.protection.reached
    assert result.privacy_after.k_anonymity < 2000


def test_the_chunked_sample_is_the_sample_an_in_memory_run_picks():
    key = b"sample-key"
    original = _keyed(_ledger())
    released = _keyed(_ledger().assign(amount=lambda frame: frame["amount"] * 2))
    expected = align_sample(original, released, Engine.POLARS, key=key, max_rows=300)

    accumulator = SampleAccumulator(key, max_rows=300)
    for start in range(0, ROWS, 128):
        accumulator.add(original[start : start + 128], released[start : start + 128])
    chunked = accumulator.result()

    assert len(chunked.original) == 300
    pd.testing.assert_frame_equal(chunked.original, expected.original)
    pd.testing.assert_frame_equal(chunked.released, expected.released)


def _keyed(frame: pd.DataFrame) -> nw.DataFrame:
    keyed = pl.from_pandas(frame).with_columns(pl.col("id").alias(ROW_KEY_COLUMN))
    return nw.from_native(keyed, eager_only=True)


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (ColumnAction(column="amount", technique="normalize"), "mean, spread or range"),
        (ColumnAction(column="amount", technique="rank_swap"), "ranks the whole column"),
        (ColumnAction(column="amount", technique="microaggregate"), "ranks the whole column"),
        (ColumnAction(column="amount", technique="shuffle"), "whole column"),
        (ColumnAction(column="amount", technique="gaussian_noise"), "mean and deviation"),
        (ColumnAction(column="amount", technique="bin_numeric"), "quantiles or range"),
        (ColumnAction(column="amount", technique="top_bottom_code"), "quantiles"),
        (ColumnAction(column="store", technique="scramble_categories"), "distinct values"),
        (ColumnAction(column="store", technique="suppress_rare"), "how often"),
        (ColumnAction(column="store", technique="randomized_response"), "distinct values"),
        (
            ColumnAction(column="email", technique="surrogate", params={"category": "email"}),
            "avoid the column's other values",
        ),
    ],
)
def test_techniques_that_need_the_whole_column_are_refused_with_the_reason(
    source, tmp_path, action, reason
):
    plan = _plan(actions=[action])

    with pytest.raises(PlanError, match="cannot run in chunks") as refused:
        _anonymizer().apply_file(source, tmp_path / "out.csv", plan)

    assert action.technique in str(refused.value)
    assert reason in str(refused.value)
    assert not (tmp_path / "out.csv").exists()


def test_parameters_that_remove_the_need_for_statistics_run_in_chunks(source, tmp_path):
    plan = _plan(
        actions=[
            ColumnAction(column="age", technique="bin_numeric", params={"width": 10}),
            ColumnAction(
                column="amount", technique="top_bottom_code", params={"lower": 10, "upper": 900}
            ),
        ]
    )
    in_memory = _anonymizer().apply(read_table(source), plan)

    _anonymizer().apply_file(source, tmp_path / f"out{source.suffix}", plan, chunk_rows=CHUNK_ROWS)

    assert read_table(tmp_path / f"out{source.suffix}").equals(in_memory.data)


def test_every_blocker_is_listed_at_once(source):
    plan = _plan(
        actions=[
            ColumnAction(column="amount", technique="normalize"),
            ColumnAction(column="age", technique="rank_swap"),
        ],
        enforce_k_anonymity=True,
        enforce_l_diversity=True,
        row_key=None,
    )

    reasons = chunk_blockers(plan, _anonymizer().techniques, read_table(source).schema)

    assert len(reasons) == 5
    assert any("row_key" in reason for reason in reasons)
    assert any("enforce_k_anonymity" in reason for reason in reasons)
    assert any("enforce_l_diversity" in reason for reason in reasons)


def test_an_anonymization_plan_must_say_it_keeps_the_input_order(source, tmp_path):
    plan = _plan(protection=Protection.ANONYMIZATION, actions=[])

    with pytest.raises(PlanError, match="shuffle_rows"):
        _anonymizer().apply_file(source, tmp_path / "out.csv", plan)


def test_text_dates_are_refused_because_their_layout_is_guessed_per_chunk(tmp_path):
    frame = _ledger()
    frame["signed_up"] = frame["signed_up"].dt.strftime("%d/%m/%Y")
    path = tmp_path / "text_dates.parquet"
    write_table(pl.from_pandas(frame), path)
    plan = _plan(actions=[ColumnAction(column="signed_up", technique="truncate_date")])

    with pytest.raises(PlanError, match="text dates"):
        _anonymizer().apply_file(path, tmp_path / "out.parquet", plan)


def test_a_row_key_repeated_across_chunks_is_refused(tmp_path):
    frame = _ledger()
    frame.loc[ROWS - 1, "id"] = 1
    path = tmp_path / "repeated.parquet"
    write_table(pl.from_pandas(frame), path)

    with pytest.raises(PlanError, match="repeats"):
        _anonymizer().apply_file(path, tmp_path / "out.parquet", _plan(), chunk_rows=CHUNK_ROWS)


def test_the_output_must_not_overwrite_the_input(source):
    with pytest.raises(PlanError, match="differ"):
        _anonymizer().apply_file(source, source, _plan())


def test_a_failed_run_leaves_no_partial_output(source, tmp_path):
    plan = _plan(actions=[ColumnAction(column="email", technique="tokenize")])

    with pytest.raises(PlanError):
        _anonymizer().apply_file(source, tmp_path / f"out{source.suffix}", plan)

    assert not (tmp_path / f"out{source.suffix}").exists()


def test_an_unsupported_file_extension_is_refused(tmp_path):
    path = tmp_path / "ledger.xlsx"
    path.write_text("x")

    with pytest.raises(TableFormatError):
        _anonymizer().apply_file(path, tmp_path / "out.csv", _plan())


def test_the_cli_applies_a_plan_in_chunks(source, tmp_path, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", "cli-key-for-tests-0123456789")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(_plan().to_json())
    output = tmp_path / f"out{source.suffix}"
    report = tmp_path / "report.json"

    exit_code = main(
        ["--seed", "1", "apply", str(source), "--plan", str(plan_path), "-o", str(output)]
        + ["--chunk-rows", "300", "--report", str(report)]
    )

    assert exit_code == 0
    assert len(read_table(output)) == ROWS
    written = json.loads(report.read_text())
    assert written["protection"]["level"] == "pseudonymized"
    assert written["rows"] == ROWS


def test_the_cli_reports_the_blockers_of_a_plan_that_cannot_chunk(source, tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        _plan(actions=[ColumnAction(column="amount", technique="normalize")]).to_json()
    )

    exit_code = main(
        ["apply", str(source), "--plan", str(plan_path), "-o", str(tmp_path / "out.csv")]
        + ["--chunk-rows", "300"]
    )

    assert exit_code != 0
    assert "cannot run in chunks" in capsys.readouterr().err


def test_wide_and_overlong_integers_read_as_text_in_every_path(tmp_path):
    source = tmp_path / "wide.csv"
    overlong = "9" * 45
    source.write_text(
        "account,amount\n12345678901234567890123,1.5\n" + f"{overlong},2.5\n" + "7,3.5\n"
    )

    whole = read_table(source)
    chunks = pl.concat(list(iter_table_chunks(source, 2)))

    assert whole.schema["account"] == pl.String
    assert whole["account"].to_list() == ["12345678901234567890123", overlong, "7"]
    assert chunks.equals(whole)
    assert len(whole.to_pandas()) == 3
