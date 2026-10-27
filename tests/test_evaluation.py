import pandas as pd

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
