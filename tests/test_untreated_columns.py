import numpy as np
import pandas as pd
import pytest

from data_anonymizer import Anonymizer, Fidelity, Protection, ProtectionLevel
from data_anonymizer.cli import main

ROWS = 2000


def _customers() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    start = pd.Timestamp("2023-01-01")
    return pd.DataFrame(
        {
            "signup": start + pd.to_timedelta(rng.uniform(0, 700, ROWS), unit="D"),
            "spend": rng.gamma(4.0, 30.0, ROWS),
            "plan": rng.choice(["free", "basic", "pro"], ROWS),
            "newsletter": rng.choice([True, False], ROWS),
        }
    )


def _propose(frame: pd.DataFrame, fidelity: Fidelity):
    return Anonymizer(seed=1).propose(frame, protection=Protection.ANONYMIZATION, fidelity=fidelity)


def test_statistical_fidelity_keeps_the_day_and_three_significant_figures():
    plan = _propose(_customers(), Fidelity.STATISTICAL)

    assert plan.action_for("signup").technique == "truncate_date"
    assert plan.action_for("signup").params == {"unit": "day"}
    # A typical spend near 100 keeps whole units: three significant figures.
    assert plan.action_for("spend").params == {"decimals": 0}
    assert {"plan", "newsletter"} <= set(plan.quasi_identifiers)


def test_privacy_fidelity_keeps_the_year_and_quintile_bands_under_k_anonymity():
    plan = _propose(_customers(), Fidelity.PRIVACY)

    assert plan.action_for("signup").params == {"unit": "year"}
    assert plan.action_for("spend").technique == "bin_numeric"
    assert {"signup", "spend", "plan", "newsletter"} <= set(plan.quasi_identifiers)


@pytest.mark.parametrize("fidelity", list(Fidelity))
def test_the_default_proposal_reaches_anonymization_on_fingerprint_columns(fidelity):
    frame = _customers()
    anonymizer = Anonymizer(seed=1)

    result = anonymizer.apply(frame, _propose(frame, fidelity))

    assert result.protection.problems == []
    assert result.protection.level is ProtectionLevel.ANONYMIZED


def test_pseudonymization_proposals_leave_fingerprint_columns_alone():
    plan = Anonymizer(seed=1).propose(_customers(), protection=Protection.PSEUDONYMIZATION)

    assert plan.action_for("signup") is None
    assert plan.action_for("spend") is None


def test_the_command_line_reads_iso_dates_in_a_csv_as_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("ANONYMIZER_SECRET_KEY", "cli-key-for-tests-0123456789")
    source = tmp_path / "customers.csv"
    _customers().to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"

    assert main(["propose", str(source), "-o", str(plan_path)]) == 0

    assert '"truncate_date"' in plan_path.read_text()
