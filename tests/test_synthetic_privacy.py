import json

import numpy as np
import pandas as pd
import pytest

from oblitus import Anonymizer, PrivacyBudget, SchemaError
from oblitus.cli import main
from oblitus.evaluation import SyntheticPrivacyThresholds, assess_synthetic_privacy

ROWS = 1500
DOMAINS = {
    "income": {"kind": "numeric", "lower": 0, "upper": 200, "bins": 20, "integer": False},
    "age": {"kind": "numeric", "lower": 18, "upper": 90, "bins": 18, "integer": True},
    "plan": ["free", "pro", "team"],
}


def _population(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, rows)
    income = np.clip(age * 1.2 + rng.normal(0, 25, rows), 0, 200).round(3)
    plan = rng.choice(["free", "pro", "team"], rows, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({"age": age, "income": income, "plan": plan})


@pytest.fixture
def train() -> pd.DataFrame:
    return _population(ROWS, seed=1)


@pytest.fixture
def holdout() -> pd.DataFrame:
    return _population(ROWS, seed=2)


def test_a_synthesizer_that_copies_rows_fails_every_test(train, holdout):
    copies = train.sample(800, random_state=0)

    report = Anonymizer(seed=3).check_synthetic(copies, train, holdout)

    assert not report.passed
    assert report.exact_copy_share == 1.0
    assert report.dcr_share is not None
    assert report.dcr_share > 0.9
    assert report.membership is not None
    assert report.membership.advantage_ci_low > 0
    # Copying every row is also caught by the isolated-record test, so it adds a fourth finding.
    assert len(report.findings) == 4


def test_copies_with_a_little_noise_still_fail_the_membership_and_share_tests(train, holdout):
    rng = np.random.default_rng(4)
    noisy = train.sample(800, random_state=0).reset_index(drop=True)
    noisy["income"] = noisy["income"] + rng.normal(0, 1.0, len(noisy))

    report = Anonymizer(seed=3).check_synthetic(noisy, train, holdout)

    assert report.exact_copy_share == 0.0
    assert not report.passed
    assert report.dcr_share is not None
    assert report.dcr_share > 0.55


def test_fresh_draws_from_the_population_pass(train, holdout):
    fresh = _population(800, seed=99)

    report = Anonymizer(seed=3).check_synthetic(fresh, train, holdout)

    assert report.passed, report.findings
    assert report.dcr_share == pytest.approx(0.5, abs=0.05)
    assert report.membership is not None
    assert report.membership.auc_ci_low <= 0.5 <= report.membership.auc_ci_high


def test_the_copula_synthesizer_passes(train, holdout):
    anonymizer = Anonymizer(seed=5)
    synthetic = anonymizer.synthesize(800, reference=train)

    report = anonymizer.check_synthetic(synthetic, train, holdout)

    assert report.passed, report.findings


def test_mst_at_a_small_epsilon_passes(train, holdout):
    anonymizer = Anonymizer(seed=6)
    budget = PrivacyBudget.from_epsilon_delta(1.0, 1e-6)
    synthetic = anonymizer.synthesize_private(train, DOMAINS, budget, rows=800).data

    report = anonymizer.check_synthetic(synthetic, train, holdout)

    assert report.passed, report.findings


@pytest.mark.parametrize("epsilon", [1.0, 20.0])
def test_aim_passes_at_a_small_and_a_moderate_epsilon(train, holdout, epsilon):
    pytest.importorskip("mbi")
    anonymizer = Anonymizer(seed=7)
    budget = PrivacyBudget.from_epsilon_delta(epsilon, 1e-6)
    synthetic = anonymizer.synthesize_private(train, DOMAINS, budget, rows=800, method="aim").data

    report = anonymizer.check_synthetic(synthetic, train, holdout)

    assert report.passed, report.findings


def test_text_in_a_numeric_column_is_refused_without_naming_it(train, holdout):
    synthetic = train.head(300).assign(income=train["income"].head(300).astype(object))
    synthetic.loc[5, "income"] = "CONFIDENTIAL_DIAGNOSIS"

    with pytest.raises(SchemaError, match="not numbers") as refusal:
        Anonymizer(seed=3).check_synthetic(synthetic, train, holdout)

    assert "CONFIDENTIAL" not in str(refusal.value)
    assert refusal.value.__context__ is None


def test_bytes_in_a_category_column_are_compared_not_decoded(train, holdout):
    synthetic = train.head(300).assign(plan=train["plan"].head(300).astype(object))
    synthetic.loc[5, "plan"] = b"\xff\xfe"

    report = Anonymizer(seed=3).check_synthetic(synthetic, train, holdout)

    assert report.n_synthetic == 300


def test_without_a_holdout_only_copies_and_distances_are_reported(train):
    report = Anonymizer(seed=3).check_synthetic(train.head(100), train)

    assert report.exact_copy_share == 1.0
    assert report.dcr_share is None
    assert report.membership is None
    assert not report.passed


def test_repeating_rows_of_the_population_are_not_counted_as_copies():
    rng = np.random.default_rng(0)
    coarse = pd.DataFrame({"plan": rng.choice(["a", "b", "c"], 500)})
    other = pd.DataFrame({"plan": rng.choice(["a", "b", "c"], 500)})

    report = assess_synthetic_privacy(coarse.head(300), coarse, other, seed=1)

    assert report.exact_copy_share > 0.9
    assert not any("exact copies" in finding for finding in report.findings)


def test_blank_values_match_only_blank_values():
    real = pd.DataFrame(
        {"amount": [1.0, 2.0, np.nan, 4.0] * 10, "kind": ["a", "b", None, "d"] * 10}
    )
    synthetic = pd.DataFrame({"amount": [np.nan, 3.0], "kind": [None, "b"]})

    report = assess_synthetic_privacy(synthetic, real, seed=1)

    assert report.exact_copy_share == 0.5


def test_dates_are_compared_as_numbers():
    days = pd.date_range("2024-01-01", periods=200, freq="D")
    real = pd.DataFrame({"day": days})
    synthetic = pd.DataFrame({"day": days[:20]})

    report = assess_synthetic_privacy(synthetic, real, seed=1)

    assert report.exact_copy_share == 1.0


def test_tables_are_cut_to_the_row_cap(train, holdout):
    report = assess_synthetic_privacy(train, train, holdout, max_rows=200, seed=1)

    assert (report.n_synthetic, report.n_train, report.n_holdout) == (200, 200, 200)


def test_the_thresholds_can_be_loosened(train, holdout):
    copies = train.sample(100, random_state=0)
    lenient = SyntheticPrivacyThresholds(max_exact_copy_share=1.0, max_dcr_share=1.0)

    report = assess_synthetic_privacy(copies, train, holdout, thresholds=lenient, seed=1)

    assert not any("exact copies" in finding for finding in report.findings)
    assert not any("closer to the training" in finding for finding in report.findings)


def test_tables_with_no_shared_column_are_refused():
    with pytest.raises(SchemaError):
        assess_synthetic_privacy(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))


def test_the_command_exits_3_for_copies_and_0_for_fresh_rows(train, holdout, tmp_path, capsys):
    train.to_csv(tmp_path / "train.csv", index=False)
    holdout.to_csv(tmp_path / "holdout.csv", index=False)
    train.head(300).to_csv(tmp_path / "copies.csv", index=False)
    _population(500, seed=77).to_csv(tmp_path / "fresh.csv", index=False)
    real_args = ["--real", str(tmp_path / "train.csv"), "--holdout", str(tmp_path / "holdout.csv")]
    report_path = tmp_path / "report.json"

    copies_exit = main(["check-synthetic", str(tmp_path / "copies.csv"), *real_args])
    assert copies_exit == 3
    assert "verdict: FAIL" in capsys.readouterr().out

    fresh_args = ["check-synthetic", str(tmp_path / "fresh.csv"), *real_args]
    assert main([*fresh_args, "--report", str(report_path)]) == 0
    assert "verdict: pass" in capsys.readouterr().out
    assert json.loads(report_path.read_text())["findings"] == []


def _continuous_population(rows: int, seed: int) -> pd.DataFrame:
    """Four continuous measurements, five of the records in a rare high-lab cohort."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(rng.normal(size=(rows, 4)), columns=["age", "height", "weight", "lab"])
    table.loc[:4, "lab"] += 100
    return table


def _rare_records_appended(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _continuous_population(1000, seed)
    holdout = _continuous_population(1000, seed + 1)
    fresh = _continuous_population(1000, seed + 2)
    return fresh, train, holdout


def test_a_generator_that_appends_the_rare_training_records_fails():
    fresh, train, holdout = _rare_records_appended(seed=141)
    synthetic = pd.concat([fresh, train.iloc[:5]], ignore_index=True)

    report = Anonymizer(seed=141).check_synthetic(synthetic, train, holdout)

    assert report.exact_copy_share < 0.01
    assert not report.passed
    assert any("most isolated training records" in finding for finding in report.findings)
    assert report.isolated_records is not None
    assert (report.isolated_records.train_copies, report.isolated_records.holdout_copies) == (5, 0)


def test_one_verbatim_rare_record_among_fresh_rows_fails():
    fresh, train, holdout = _rare_records_appended(seed=7)
    synthetic = pd.concat([fresh, train.iloc[:1]], ignore_index=True)

    report = Anonymizer(seed=7).check_synthetic(synthetic, train, holdout)

    assert not report.passed


def test_fresh_rows_of_the_same_population_have_no_isolated_copies():
    fresh, train, holdout = _rare_records_appended(seed=141)

    report = Anonymizer(seed=141).check_synthetic(fresh, train, holdout)

    assert report.passed, report.findings
    assert report.isolated_records is not None
    assert report.isolated_records.train_copies == 0


def test_isolated_copies_are_checked_without_a_holdout_too():
    fresh, train, _ = _rare_records_appended(seed=141)
    synthetic = pd.concat([fresh, train.iloc[:5]], ignore_index=True)

    report = Anonymizer(seed=141).check_synthetic(synthetic, train)

    assert any("most isolated training records" in finding for finding in report.findings)


def test_natural_matches_of_a_discrete_table_are_not_isolated_copies():
    def population(seed_offset: int) -> pd.DataFrame:
        rows = np.random.default_rng(seed_offset).integers(0, 40, size=(600, 3))
        return pd.DataFrame(rows, columns=["a", "b", "c"]).astype(str)

    report = assess_synthetic_privacy(population(2), population(1), population(3), seed=1)

    assert not any("isolated" in finding for finding in report.findings)


def test_the_command_reports_the_isolated_records_and_exits_3(tmp_path, capsys):
    fresh, train, holdout = _rare_records_appended(seed=141)
    synthetic = pd.concat([fresh, train.iloc[:5]], ignore_index=True)
    synthetic.to_csv(tmp_path / "synthetic.csv", index=False)
    train.to_csv(tmp_path / "train.csv", index=False)
    holdout.to_csv(tmp_path / "holdout.csv", index=False)

    exit_code = main(
        [
            "check-synthetic",
            str(tmp_path / "synthetic.csv"),
            "--real",
            str(tmp_path / "train.csv"),
            "--holdout",
            str(tmp_path / "holdout.csv"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 3
    assert "isolated training records copied: 5 of 100 (5 on every column)" in output


def _check_with_rare_copies(copied: pd.DataFrame, seed: int, train=None, holdout=None):
    fresh, own_train, own_holdout = _rare_records_appended(seed)
    train = own_train if train is None else train
    holdout = own_holdout if holdout is None else holdout
    synthetic = pd.concat([fresh, copied], ignore_index=True)
    return Anonymizer(seed=seed).check_synthetic(synthetic, train, holdout)


def test_rare_training_records_with_a_tiny_jitter_still_fail():
    _, train, _ = _rare_records_appended(seed=141)
    jitter = np.random.default_rng(1).uniform(-0.0001, 0.0001, (5, 4))
    copied = train.iloc[:5] + jitter

    report = _check_with_rare_copies(copied, seed=141)

    assert report.exact_copy_share == 0
    assert not report.passed
    assert any("most isolated training records" in finding for finding in report.findings)


def test_rare_training_records_copied_on_some_columns_only_fail():
    _, train, _ = _rare_records_appended(seed=141)
    copied = train.iloc[:5].copy()
    copied["lab"] = np.random.default_rng(2).normal(100, 1, len(copied))

    report = _check_with_rare_copies(copied, seed=141)

    assert report.isolated_records is not None
    assert report.isolated_records.every_column.train == 0
    assert report.isolated_records.most_columns.train == 5
    assert not report.passed


def test_repeated_ordinary_rows_do_not_excuse_copies_of_rare_records():
    _, train, holdout = _rare_records_appended(seed=141)
    train.iloc[-1] = train.iloc[-2]
    holdout.iloc[-1] = holdout.iloc[-2]

    report = _check_with_rare_copies(train.iloc[:5], 141, train, holdout)

    assert not report.passed
    assert any("most isolated training records" in finding for finding in report.findings)


def test_a_large_jitter_is_a_known_limit_of_the_isolated_record_test():
    _, train, _ = _rare_records_appended(seed=141)
    copied = train.iloc[:5] + np.random.default_rng(1).uniform(-0.3, 0.3, (5, 4))

    report = _check_with_rare_copies(copied, seed=141)

    assert report.isolated_records is not None
    assert not report.isolated_records.is_excessive
