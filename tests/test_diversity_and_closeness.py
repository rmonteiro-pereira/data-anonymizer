import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from oblitus import (
    AnonymizationPlan,
    Anonymizer,
    ColumnAction,
    Fidelity,
    Protection,
    ProtectionLevel,
)
from oblitus.diversity_enforcement import enforce_l_diversity
from oblitus.engines import Engine
from oblitus.evaluation import assess_privacy, t_closeness


def _clinic() -> pd.DataFrame:
    """Three groups of five: one uniform in diagnosis, two varied."""
    return pd.DataFrame(
        {
            "id": range(15),
            "zip": ["100"] * 5 + ["200"] * 5 + ["300"] * 5,
            "diagnosis": ["flu"] * 5 + ["flu", "asthma", "flu", "cold", "flu"] + list("abcab"),
        }
    )


def test_a_group_with_one_sensitive_value_is_blanked_into_a_diverse_group():
    outcome = enforce_l_diversity(_clinic(), ["zip"], ["diagnosis"], 2)

    zips = outcome.data["zip"]
    # Group 100 alone holds one value, so the next smallest group, 200, joins it.
    assert zips.iloc[:10].isna().all()
    assert zips.iloc[10:].tolist() == ["300"] * 5
    assert outcome.suppressed_cells == 10
    report = assess_privacy(outcome.data, ["zip"], ["diagnosis"], k_target=5)
    assert report.l_diversity == {"diagnosis": 3}


def test_the_blanked_group_takes_the_smallest_groups_until_it_has_enough_values():
    frame = pd.DataFrame(
        {
            "zip": ["1"] * 4 + ["2"] * 3 + ["3"] * 6,
            "diagnosis": ["x"] * 4 + ["x", "x", "y"] + ["x", "y", "z", "x", "y", "z"],
        }
    )

    outcome = enforce_l_diversity(frame, ["zip"], ["diagnosis"], 2)

    # Group 1 is blanked for holding one value and still lacks a second, so group 2 joins it.
    assert outcome.data["zip"].isna().tolist() == [True] * 7 + [False] * 6


def test_a_table_holding_fewer_values_than_l_is_asked_for_only_what_it_has():
    frame = pd.DataFrame({"zip": ["1", "1", "2", "2"], "diagnosis": ["x", "x", "x", "y"]})

    outcome = enforce_l_diversity(frame, ["zip"], ["diagnosis"], 3)

    assert outcome.data["zip"].isna().all()


def test_nothing_changes_when_every_group_is_diverse_enough():
    frame = _clinic().iloc[5:].reset_index(drop=True)

    outcome = enforce_l_diversity(frame, ["zip"], ["diagnosis"], 2)

    assert outcome.suppressed_cells == 0
    assert outcome.data["zip"].equals(frame["zip"])


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_every_engine_blanks_the_same_groups(engine):
    frame = _clinic()
    expected = enforce_l_diversity(frame, ["zip"], ["diagnosis"], 2)
    native = pl.from_pandas(frame) if engine is Engine.POLARS else duckdb.connect().from_df(frame)

    outcome = enforce_l_diversity(native, ["zip"], ["diagnosis"], 2)

    released = outcome.data.to_pandas() if engine is Engine.POLARS else outcome.data.df()
    released = released.sort_values("id").reset_index(drop=True)
    assert released["zip"].isna().tolist() == expected.data["zip"].isna().tolist()
    assert outcome.suppressed_cells == expected.suppressed_cells


def _plan(**overrides) -> AnonymizationPlan:
    settings = {
        "fidelity": Fidelity.PRIVACY,
        "protection": Protection.ANONYMIZATION,
        "actions": [ColumnAction(column="diagnosis", technique="keep")],
        "row_key": "id",
        "quasi_identifiers": ["zip"],
        "sensitive_attributes": ["diagnosis"],
        "k_target": 5,
        "l_target": 2,
        "enforce_k_anonymity": True,
    }
    settings.update(overrides)
    return AnonymizationPlan(**settings)


def test_a_plan_without_enforcement_leaves_a_low_diversity_group_at_risk():
    result = Anonymizer(seed=1).apply(_clinic(), _plan())

    assert result.protection.level is ProtectionLevel.AT_RISK
    assert any("l-diversity" in problem for problem in result.protection.problems)


def test_a_plan_with_enforcement_meets_the_l_diversity_target():
    result = Anonymizer(seed=1).apply(_clinic(), _plan(enforce_l_diversity=True))

    assert result.privacy_after.l_diversity["diagnosis"] >= 2
    assert not any("l-diversity" in problem for problem in result.protection.problems)
    assert result.suppressed_cells == 10


def _brute_force_distance(group: pd.Series, table: pd.Series, *, ordered: bool) -> float:
    values = sorted(table.unique())
    p = np.array([(group == value).mean() for value in values])
    q = np.array([(table == value).mean() for value in values])
    if not ordered:
        return float(np.abs(p - q).sum() / 2)
    return float(np.abs(np.cumsum(p - q)).sum() / (len(values) - 1))


def test_t_closeness_of_categories_is_half_the_total_difference_of_the_shares():
    frame = pd.DataFrame(
        {"zip": ["1"] * 4 + ["2"] * 4, "disease": ["a", "a", "a", "b", "b", "b", "b", "a"]}
    )

    distance = t_closeness(frame, ["zip"], ["disease"])["disease"]

    assert distance == pytest.approx(0.25)


def test_t_closeness_is_zero_when_every_group_mirrors_the_table():
    frame = pd.DataFrame({"zip": ["1"] * 4 + ["2"] * 4, "disease": list("abab") * 2})

    assert t_closeness(frame, ["zip"], ["disease"]) == {"disease": 0.0}


def test_t_closeness_of_a_group_holding_one_value_of_two_is_one_half():
    frame = pd.DataFrame({"zip": ["1"] * 5 + ["2"] * 5, "disease": ["a"] * 5 + list("abbab")})

    assert t_closeness(frame, ["zip"], ["disease"])["disease"] == pytest.approx(0.3)


def test_the_ordered_distance_matches_a_brute_force_calculation():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "zip": rng.choice(list("abcdef"), 400),
            "salary": rng.choice([1000, 1500, 2000, 3000, 5000, 8000, 13000], 400),
        }
    )
    expected = max(
        _brute_force_distance(group["salary"], frame["salary"], ordered=True)
        for _, group in frame.groupby("zip")
    )

    assert t_closeness(frame, ["zip"], ["salary"])["salary"] == pytest.approx(expected)


def test_the_ordered_distance_is_small_for_nearby_values_and_large_for_far_ones():
    near = pd.DataFrame({"zip": ["1"] * 3 + ["2"] * 3, "pay": [1, 1, 2, 2, 3, 3]})
    far = pd.DataFrame({"zip": ["1"] * 3 + ["2"] * 3, "pay": [1, 1, 1, 9, 9, 9]})

    assert t_closeness(near, ["zip"], ["pay"])["pay"] < t_closeness(far, ["zip"], ["pay"])["pay"]


def test_blank_sensitive_cells_are_not_counted():
    frame = pd.DataFrame({"zip": ["1", "1", "2", "2"], "disease": ["a", None, "a", None]})

    assert t_closeness(frame, ["zip"], ["disease"]) == {"disease": 0.0}


def test_the_privacy_report_carries_t_closeness_only_when_asked_for():
    frame = _clinic()

    without = assess_privacy(frame, ["zip"], ["diagnosis"])
    asked = assess_privacy(frame, ["zip"], ["diagnosis"], t_target=0.2)

    assert without.t_closeness == {}
    assert without.meets_t_target
    assert asked.t_closeness["diagnosis"] > 0.2
    assert not asked.meets_t_target


def test_a_t_target_in_the_plan_makes_a_distant_group_a_protection_problem():
    result = Anonymizer(seed=1).apply(_clinic(), _plan(t_target=0.2, l_target=1))

    assert result.privacy_after.t_closeness["diagnosis"] > 0.2
    assert result.protection.level is ProtectionLevel.AT_RISK
    assert any("t-closeness" in problem for problem in result.protection.problems)


def test_a_loose_t_target_adds_no_problem():
    result = Anonymizer(seed=1).apply(_clinic(), _plan(t_target=1.0, l_target=1))

    assert not any("t-closeness" in problem for problem in result.protection.problems)


def _skewed_clinic() -> pd.DataFrame:
    """North holds 99 diabetes and 1 healthy, where the table as a whole holds 11 percent."""
    north = ["D"] * 99 + ["H"]
    south = ["D"] + ["H"] * 99
    east = ["H"] * 695 + ["D"] * 5
    return pd.DataFrame(
        {
            "id": range(900),
            "area": ["north"] * 100 + ["south"] * 100 + ["east"] * 700,
            "diagnosis": north + south + east,
        }
    )


def _skew_plan(**overrides) -> AnonymizationPlan:
    settings = {
        "fidelity": Fidelity.PRIVACY,
        "protection": Protection.ANONYMIZATION,
        "actions": [
            ColumnAction(column="id", technique="suppress"),
            ColumnAction(column="diagnosis", technique="keep"),
        ],
        "direct_identifiers": ["id"],
        "quasi_identifiers": ["area"],
        "sensitive_attributes": ["diagnosis"],
        "enforce_k_anonymity": True,
        "enforce_l_diversity": True,
    }
    settings.update(overrides)
    return AnonymizationPlan(**settings)


def test_a_group_that_is_99_percent_one_value_is_at_risk_despite_l_diversity():
    result = Anonymizer(seed=1).apply(_skewed_clinic(), _skew_plan())

    assert result.protection.level is ProtectionLevel.AT_RISK
    problem = next(text for text in result.protection.problems if "diagnosis" in text)
    assert "area = 'north'" in problem
    assert "'D'" in problem
    assert "t_target" in problem


def test_a_naturally_lopsided_attribute_is_not_flagged_for_what_everyone_knows():
    frame = pd.DataFrame(
        {
            "id": range(400),
            "area": ["a"] * 200 + ["b"] * 200,
            "diagnosis": ["H"] * 190 + ["D"] * 10 + ["H"] * 196 + ["D"] * 4,
        }
    )

    result = Anonymizer(seed=1).apply(frame, _skew_plan())

    assert result.protection.level is ProtectionLevel.ANONYMIZED


def test_the_confidence_and_gain_limits_come_from_the_plan():
    strict = _skew_plan(inference_confidence=0.995)

    result = Anonymizer(seed=1).apply(_skewed_clinic(), strict)

    assert result.protection.level is ProtectionLevel.ANONYMIZED
