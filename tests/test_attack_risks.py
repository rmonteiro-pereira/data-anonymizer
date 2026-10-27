import json
import sys

import numpy as np
import pandas as pd
import pytest

from oblitus.cli import main
from oblitus.errors import MissingExtraError
from oblitus.evaluation import assess_attack_risks

QUASI_IDENTIFIERS = ["age", "zip", "job"]


def _population(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, rows),
            "zip": rng.integers(10000, 10400, rows).astype(str),
            "job": rng.choice(list("abcdefghij"), rows),
            "disease": rng.choice(["flu", "asthma", "diabetes", "none"], rows),
        }
    )


def test_a_missing_extra_names_the_extra_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "anonymeter", None)
    frame = _population(30, 1)

    with pytest.raises(MissingExtraError, match=r"oblitus\[anonymeter\]"):
        assess_attack_risks(frame, frame, quasi_identifiers=QUASI_IDENTIFIERS)


def test_a_copy_of_the_original_is_far_easier_to_attack_than_unrelated_data():
    pytest.importorskip("anonymeter")
    original = _population(400, 1)
    control = _population(400, 2)
    unrelated = _population(400, 3)

    copied = assess_attack_risks(
        original,
        original.copy(),
        quasi_identifiers=QUASI_IDENTIFIERS,
        sensitive_attributes=["disease"],
        control=control,
        n_attacks=150,
        seed=0,
    )
    independent = assess_attack_risks(
        original,
        unrelated,
        quasi_identifiers=QUASI_IDENTIFIERS,
        sensitive_attributes=["disease"],
        control=control,
        n_attacks=150,
        seed=0,
    )

    for attack in ("singling_out", "linkability", "inference:disease"):
        assert copied.risk_of(attack).risk > independent.risk_of(attack).risk
    risk = copied.risk_of("singling_out")
    assert risk.ci_low <= risk.risk <= risk.ci_high


def test_an_attack_that_cannot_run_is_listed_as_skipped():
    pytest.importorskip("anonymeter")
    original = _population(200, 1)

    report = assess_attack_risks(
        original,
        original.copy(),
        quasi_identifiers=["zip"],
        n_attacks=50,
        seed=0,
    )

    assert "linkability" in report.skipped
    assert [risk.attack for risk in report.risks] == ["singling_out"]


def test_the_apply_flag_prints_the_attack_risks_and_writes_them_to_the_report(tmp_path, capsys):
    pytest.importorskip("anonymeter")
    source = tmp_path / "people.csv"
    _population(300, 1).assign(id=range(300)).to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "fidelity": "privacy",
                "protection": "anonymization",
                "actions": [],
                "row_key": "id",
                "quasi_identifiers": QUASI_IDENTIFIERS,
                "sensitive_attributes": ["disease"],
                "enforce_k_anonymity": True,
            }
        )
    )
    report = tmp_path / "report.json"

    arguments = ["apply", str(source), "--plan", str(plan_path), "-o", str(tmp_path / "out.csv")]
    main(["--seed", "1", *arguments, "--report", str(report), "--attack-risks"])

    assert "attack singling_out: risk" in capsys.readouterr().out
    written = json.loads(report.read_text())
    assert {entry["attack"] for entry in written["attack_risks"]["risks"]} >= {"singling_out"}
