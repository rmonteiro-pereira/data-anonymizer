import pandas as pd
import pytest

from oblitus.evaluation import assess_privacy, compare_tables
from oblitus.local_suppression import enforce_k_anonymity


def test_k_anonymity_and_l_diversity_of_a_small_table():
    frame = pd.DataFrame(
        {
            "zip": ["100", "100", "100", "200", "200"],
            "age": ["30s", "30s", "30s", "40s", "40s"],
            "disease": ["flu", "flu", "asthma", "flu", "flu"],
        }
    )

    report = assess_privacy(frame, ["zip", "age"], ["disease"], k_target=3)

    assert report.k_anonymity == 2
    assert report.rows_below_target == 2
    assert not report.meets_target
    assert report.l_diversity == {"disease": 1}


def test_local_suppression_blanks_only_rows_in_small_groups():
    frame = pd.DataFrame(
        {
            "zip": ["100"] * 5 + ["200"] * 5 + ["998", "999"],
            "gender": ["F"] * 5 + ["M"] * 7,
        }
    )

    outcome = enforce_k_anonymity(frame, ["zip", "gender"], k=2)

    assert outcome.data["zip"].tolist()[:10] == frame["zip"].tolist()[:10]
    assert outcome.data["zip"].iloc[10:].isna().all()
    assert outcome.data["gender"].equals(frame["gender"])
    assert outcome.suppressed_cells == 2
    assert assess_privacy(outcome.data, ["zip", "gender"], k_target=2).meets_target


def test_a_lone_outlier_is_merged_with_the_smallest_complete_group():
    frame = pd.DataFrame({"zip": ["100"] * 5 + ["200"] * 2 + ["999"]})

    outcome = enforce_k_anonymity(frame, ["zip"], k=2)

    assert assess_privacy(outcome.data, ["zip"], k_target=2).meets_target
    # The two-row group is the cheapest to blank; the five-row group keeps its value.
    assert outcome.data["zip"].isna().tolist() == [False] * 5 + [True] * 3
    assert outcome.suppressed_cells == 3


def test_compare_tables_reports_zero_drift_for_identical_tables():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 5.0, 9.0], "c": list("xyxy")})

    report = compare_tables(frame, frame.copy())

    assert report.for_column("a").ks_statistic == 0
    assert report.for_column("c").total_variation == 0
    assert report.max_correlation_change == 0


def test_the_utility_report_counts_the_cells_a_release_blanked():
    original = pd.DataFrame({"age": [30.0, 41.0, 52.0, 63.0], "city": ["a", "b", "c", "d"]})
    released = original.assign(age=[30.0, None, None, 63.0], city=["a", "b", None, "d"])

    report = compare_tables(original, released)

    assert report.for_column("age").added_blank_share == 0.5
    assert report.for_column("city").added_blank_share == 0.25


def test_a_date_shift_is_measured_in_standard_deviations_not_against_1970():
    original = pd.DataFrame({"day": pd.date_range("2024-01-01", periods=365, freq="D")})
    released = original.assign(day=original["day"] + pd.Timedelta(days=182))

    report = compare_tables(original, released)

    # Half a year on a one-year spread of about 105 days is close to 1.7 deviations.
    assert report.for_column("day").mean_change == pytest.approx(182 / 105.5, rel=0.02)


def test_a_person_in_several_small_groups_is_counted_once():
    frame = pd.DataFrame(
        {
            "person": ["a"] * 5 + ["b", "c"],
            "zip": ["1", "2", "3", "4", "5", "6", "6"],
            "dx": list("xyxyxyx"),
        }
    )

    report = assess_privacy(frame, ["zip"], ["dx"], k_target=5, entity_column="person")

    assert report.rows_below_target == 3
    assert report.share_below_target == 1.0
