import json
from pathlib import Path

import pandas as pd

from data_anonymizer.cli import main


def test_propose_then_apply_with_a_reviewed_plan(people, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANONYMIZER_SECRET_KEY", "cli-key")
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
