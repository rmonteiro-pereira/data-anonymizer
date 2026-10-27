import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from oblitus import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, PlanError, Protection
from oblitus.engines import Engine
from oblitus.joint_microaggregation import mdav_groups, microaggregate_jointly
from oblitus.plan import JointMicroaggregation
from oblitus.protection import ProtectionLevel


def _measurements(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "height": rng.normal(170, 10, rows).round(1),
            "weight": rng.normal(70, 12, rows).round(1),
            "visits": rng.integers(0, 30, rows),
        }
    )


@pytest.mark.parametrize("rows", [7, 10, 17, 30, 31, 100])
def test_every_group_holds_at_least_k_points_and_every_point_is_in_one(rows):
    points = np.random.default_rng(2).normal(size=(rows, 3))

    groups = mdav_groups(points, 3)

    assert min(len(group) for group in groups) >= 3
    assert sorted(np.concatenate(groups).tolist()) == list(range(rows))


def test_every_released_vector_is_shared_by_at_least_k_records():
    frame = _measurements()

    outcome = microaggregate_jointly(frame, ["height", "weight"], 4)

    shared = outcome.data.groupby(["height", "weight"]).size()
    assert shared.min() >= 4
    assert outcome.groups == len(shared)


def test_each_column_keeps_its_mean():
    frame = _measurements()

    outcome = microaggregate_jointly(frame, ["height", "weight"], 4)

    for column in ("height", "weight"):
        assert outcome.data[column].mean() == pytest.approx(frame[column].mean())


def test_integer_columns_stay_whole_numbers():
    frame = _measurements()

    outcome = microaggregate_jointly(frame, ["weight", "visits"], 3)

    assert pd.api.types.is_integer_dtype(outcome.data["visits"])


def test_a_lone_record_with_a_gap_does_not_keep_its_other_values():
    frame = _measurements(30)
    frame.loc[3, "height"] = None

    outcome = microaggregate_jointly(frame, ["height", "weight"], 3)

    assert pd.isna(outcome.data.loc[3, "weight"])
    assert outcome.data.loc[3, "weight"] != frame.loc[3, "weight"]


def test_blanked_records_number_at_least_the_group_size():
    frame = _measurements(30)
    frame.loc[3, "height"] = None

    outcome = microaggregate_jointly(frame, ["height", "weight"], 3)

    blank = outcome.data[["height", "weight"]].isna().all(axis=1)
    assert blank.sum() >= 3
    shared = outcome.data[~blank].groupby(["height", "weight"]).size()
    assert shared.min() >= 3


def test_records_missing_the_same_column_are_aggregated_among_themselves_on_the_others():
    frame = _measurements(60)
    frame.loc[:11, "height"] = None

    outcome = microaggregate_jointly(frame, ["height", "weight"], 4)

    gaps = outcome.data.iloc[:12]
    assert gaps["height"].isna().all()
    assert gaps["weight"].notna().all()
    assert gaps.groupby("weight").size().min() >= 4
    assert gaps["weight"].mean() == pytest.approx(frame.loc[:11, "weight"].mean())


def test_an_outlier_with_a_gap_is_not_released_with_its_own_value():
    rng = np.random.default_rng(812)
    frame = pd.DataFrame(rng.normal(size=(251, 2)), columns=["height", "weight"])
    frame.loc[0, ["height", "weight"]] = [999.123, np.nan]
    plan = _plan(row_key=None, actions=[])

    result = Anonymizer(seed=1).apply(frame, plan)

    assert not (result.data["height"] == 999.123).any()


def test_a_text_column_is_refused():
    frame = _measurements().assign(city="x")

    with pytest.raises(ValueError, match="not numeric"):
        microaggregate_jointly(frame, ["height", "city"], 3)


def _plan(**overrides) -> AnonymizationPlan:
    settings = {
        "fidelity": Fidelity.PRIVACY,
        "protection": Protection.ANONYMIZATION,
        "actions": [ColumnAction(column="visits", technique="keep")],
        "row_key": "id",
        "shuffle_rows": False,
        "joint_microaggregation": [
            JointMicroaggregation(columns=["height", "weight"], group_size=5)
        ],
    }
    settings.update(overrides)
    return AnonymizationPlan(**settings)


def test_the_plan_step_runs_in_apply_and_leaves_other_columns_alone():
    frame = _measurements()

    result = Anonymizer(seed=1).apply(frame, _plan())

    shared = result.data.groupby(["height", "weight"]).size()
    assert shared.min() >= 5
    assert result.data["visits"].equals(frame["visits"])
    assert result.data["id"].equals(frame["id"])


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_every_engine_releases_the_same_group_means(engine):
    frame = _measurements()
    expected = Anonymizer(seed=1).apply(frame, _plan()).data
    native = pl.from_pandas(frame) if engine is Engine.POLARS else duckdb.connect().from_df(frame)

    released = Anonymizer(seed=1).apply(native, _plan()).data

    released = released.to_pandas() if engine is Engine.POLARS else released.df()
    released = released.sort_values("id").reset_index(drop=True)
    for column in ("height", "weight"):
        np.testing.assert_allclose(released[column], expected[column])


def test_a_plan_listing_a_missing_column_for_microaggregation_is_refused():
    plan = _plan(joint_microaggregation=[JointMicroaggregation(columns=["height", "nowhere"])])

    with pytest.raises(PlanError, match="nowhere"):
        Anonymizer(seed=1).apply(_measurements(), plan)


def test_one_column_is_not_enough_for_a_joint_step():
    with pytest.raises(ValueError):
        JointMicroaggregation(columns=["height"])


def test_the_protection_check_counts_microaggregated_columns_even_with_many_distinct_values():
    from oblitus.engines import as_working_frame
    from oblitus.protection import _isolated_records

    rng = np.random.default_rng(5)
    released = pd.DataFrame(rng.normal(size=(200, 2)), columns=["height", "weight"])

    problems = _isolated_records(_plan(actions=[]), as_working_frame(released))

    assert problems
    assert "height" in problems[0]


def test_a_group_of_missing_weight_records_that_all_share_a_diagnosis_is_flagged():
    rng = np.random.default_rng(812)
    frame = pd.DataFrame(rng.normal(size=(500, 2)), columns=["height", "weight"])
    frame["diagnosis"] = ["D"] * 5 + ["H"] * 495
    frame.loc[:4, "height"] = np.linspace(999.1, 999.5, 5)
    frame.loc[:4, "weight"] = np.nan
    plan = _plan(actions=[], row_key=None, sensitive_attributes=["diagnosis"], l_target=1)

    result = Anonymizer(seed=1).apply(frame, plan)

    assert result.protection.level is ProtectionLevel.AT_RISK
    assert any("diagnosis" in problem for problem in result.protection.problems)
