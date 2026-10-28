from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oblitus import Anonymizer, Fidelity, Protection, ProtectionLevel
from oblitus.cli import main
from oblitus.synthesis import load_schema

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


def test_statistical_fidelity_keeps_the_year_and_three_significant_figures():
    plan = _propose(_customers(), Fidelity.STATISTICAL)

    # An event date is a quasi-identifier, so it keeps the year, not the day.
    assert plan.action_for("signup").technique == "truncate_date"
    assert plan.action_for("signup").params == {"unit": "year"}
    assert "signup" in plan.quasi_identifiers
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
    monkeypatch.setenv("OBLITUS_SECRET_KEY", "cli-key-for-tests-0123456789")
    source = tmp_path / "customers.csv"
    _customers().to_csv(source, index=False)
    plan_path = tmp_path / "plan.json"

    assert main(["propose", str(source), "-o", str(plan_path)]) == 0

    assert '"truncate_date"' in plan_path.read_text()


def _visits(people: int) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    cities = ["Santos", "Bauru", "Campinas", "Sorocaba", "Marilia", "Franca", "Jundiai", "Osasco"]
    return pd.DataFrame(
        {
            "cidade": rng.choice(cities, people),
            "sexo": rng.choice(["F", "M"], people),
            "visit_date": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(rng.integers(0, 730, people), "D"),
            "diagnostico": rng.choice(list("ABCDEF"), people),
        }
    )


def _singled_out_share(original: pd.DataFrame, released: pd.DataFrame) -> float:
    """People whose city, sex and visit year match exactly one released row, their own."""
    years = pd.to_datetime(released["visit_date"], errors="coerce").dt.to_period("Y")
    hits = 0
    for index in range(len(original)):
        same = (released["cidade"] == original["cidade"][index]) & (
            released["sexo"] == original["sexo"][index]
        )
        same &= years == original["visit_date"][index].to_period("Y")
        matches = np.flatnonzero(same.fillna(False).to_numpy())
        hits += int(len(matches) == 1 and matches[0] == index)
    return hits / len(original)


def test_an_event_date_with_city_and_sex_does_not_single_people_out_of_an_anonymized_release():
    visits = _visits(2000)
    anonymizer = Anonymizer(secret_key=b"untreated-test-key-not-real-0123")
    plan = anonymizer.propose(visits, protection=Protection.ANONYMIZATION)
    plan.shuffle_rows = False

    result = anonymizer.apply(visits, plan)

    assert "visit_date" in plan.quasi_identifiers
    if result.protection.level is ProtectionLevel.ANONYMIZED:
        assert _singled_out_share(visits, result.data) < 0.05


def test_the_quick_start_table_keeps_most_ages_and_dates():
    schema = load_schema(Path(__file__).parent.parent / "examples" / "customers_schema.json")
    anonymizer = Anonymizer(secret_key=b"quick-start-test-key-not-real-01", seed=3)
    customers = anonymizer.synthesize(2000, schema=schema)
    plan = anonymizer.propose(customers, protection=Protection.ANONYMIZATION)

    result = anonymizer.apply(customers, plan)

    assert result.protection.level is ProtectionLevel.ANONYMIZED
    assert result.data["age"].isna().mean() < 0.3
    assert result.data["signup_date"].isna().mean() < 0.3
