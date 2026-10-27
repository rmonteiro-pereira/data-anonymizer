import json

import pandas as pd

from data_anonymizer.cli import main


def test_propose_then_apply_with_a_reviewed_plan(people, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANONYMIZER_SECRET_KEY", "cli-key")
    source = tmp_path / "people.csv"
    people.to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "out.csv"
    report = tmp_path / "report.json"

    assert main(["propose", str(source), "-o", str(plan_path), "--fidelity", "privacy"]) == 0
    plan = json.loads(plan_path.read_text())
    plan["actions"] = [action for action in plan["actions"] if action["column"] != "city"]
    plan_path.write_text(json.dumps(plan))

    exit_code = main(
        ["--seed", "1", "apply", str(source), "--plan", str(plan_path), "-o", str(output)]
        + ["--report", str(report)]
    )

    assert exit_code == 0
    anonymized = pd.read_csv(output)
    assert anonymized["email"].dropna().eq("[REDACTED]").all()
    assert "privacy_after" in json.loads(report.read_text())


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
