import json
from pathlib import Path

import pandas as pd
import pytest

from oblitus.cli import main


def test_propose_then_apply_with_a_reviewed_plan(people, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", "cli-key-for-tests-0123456789")
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "out.csv"
    report = tmp_path / "report.json"

    propose = ["propose", str(source), "-o", str(plan_path), "--fidelity", "privacy"]
    assert main(propose + ["--protection", "anonymization"]) == 0
    plan = json.loads(plan_path.read_text())
    plan["actions"] = [action for action in plan["actions"] if action["column"] != "city"]
    plan_path.write_text(json.dumps(plan))

    exit_code = main(
        ["--seed", "1", "apply", str(source), "--plan", str(plan_path), "-o", str(output)]
        + ["--report", str(report)]
    )

    written = json.loads(report.read_text())
    expected_exit = 0 if written["protection"]["reached"] else 3
    assert exit_code == expected_exit
    assert pd.read_csv(output)["email"].isna().all()
    assert "privacy_after" in written


def test_scan_prints_one_line_per_column(people, tmp_path, capsys):
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)

    assert main(["scan", str(source)]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(people.columns) + 1


def test_infer_schema_then_synthesize(people, tmp_path):
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)
    schema = tmp_path / "schema.json"
    output = tmp_path / "synthetic.csv"

    assert main(["infer-schema", str(source), "-o", str(schema)]) == 0
    assert (
        main(
            ["--seed", "2", "synthesize", "--schema", str(schema), "--rows", "50"]
            + ["-o", str(output)]
        )
        == 0
    )

    assert len(pd.read_csv(output)) == 50


def test_errors_exit_with_code_two(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "data.xlsx")]) == 2
    assert "supported" in capsys.readouterr().err


def test_challenge_command_writes_public_and_private_files(tmp_path):
    schema = Path(__file__).parent.parent / "examples" / "churn_challenge.json"
    folder = tmp_path / "churn"

    assert (
        main(
            ["--seed", "3", "challenge", "--schema", str(schema), "--rows", "500"]
            + ["-o", str(folder)]
        )
        == 0
    )

    assert {"train.csv", "test.csv", "QUESTION.md", "answers"} <= {
        path.name for path in folder.iterdir()
    }
    assert "churned" not in pd.read_csv(folder / "test.csv").columns
    assert json.loads((folder / "answers" / "answer_key.json").read_text())["target"] == "churned"


def test_apply_and_reveal_take_a_row_key_that_overrides_the_plan(people, tmp_path, monkeypatch):
    monkeypatch.setenv("OBLITUS_SECRET_KEY", "cli-key-for-tests-0123456789")
    source = tmp_path / "people.parquet"
    people.to_parquet(source, index=False)
    plan_path = tmp_path / "plan.json"
    assert (
        main(["propose", str(source), "-o", str(plan_path), "--protection", "pseudonymization"])
        == 0
    )
    assert json.loads(plan_path.read_text())["row_key"] == "customer_id"
    output = tmp_path / "out.parquet"

    apply_arguments = ["--seed", "1", "apply", str(source), "--plan", str(plan_path)]
    assert main(apply_arguments + ["-o", str(output), "--row-key", "cep"]) == 0
    assert (
        main(["reveal", str(output), "--plan", str(plan_path), "-o", str(tmp_path / "back.csv")])
        == 0
    )
    assert main(apply_arguments + ["-o", str(output), "--row-key", "missing"]) == 2


def test_challenge_command_builds_a_panel_without_a_row_count(tmp_path, capsys):
    schema = Path(__file__).parent.parent / "examples" / "store_sales_panel.json"
    folder = tmp_path / "stores"

    assert main(["--seed", "3", "challenge", "--schema", str(schema), "-o", str(folder)]) == 0

    page = (folder / "QUESTION.md").read_text()
    assert "one `store_id` in one period, dated by `date`" in page
    assert "the last 7 periods, from 2024-06-01" in page
    assert len(pd.read_csv(folder / "test.csv")) == 40 * 7
    assert main(["synthesize", "--schema", str(schema), "-o", str(tmp_path / "s.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "s.csv")) == 40 * 36
    assert main(["synthesize", "-o", str(tmp_path / "x.csv")]) == 2
    assert "rows is required" in capsys.readouterr().err


def test_propose_shows_what_each_column_will_get_and_the_next_command(people, tmp_path, capsys):
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"

    assert main(["propose", str(source), "-o", str(plan_path)]) == 0

    printed = capsys.readouterr().out
    assert "full_name" in printed
    assert "surrogate" in printed
    assert f"oblitus apply {source} --plan {plan_path}" in printed


def test_a_plan_on_standard_output_stays_valid_json(people, tmp_path, capsys):
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)

    assert main(["propose", str(source)]) == 0

    json.loads(capsys.readouterr().out)


def test_a_missing_input_file_is_named_plainly(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope.csv")]) == 2

    assert "nope.csv: no such file" in capsys.readouterr().err


def test_the_help_shows_the_usual_path(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    printed = capsys.readouterr().out
    assert "the usual path:" in printed
    assert "commands by task:" in printed
