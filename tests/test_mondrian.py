import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from oblitus import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, PlanError, Protection
from oblitus.engines import Engine
from oblitus.evaluation import assess_privacy
from oblitus.generalization_hierarchies import GeneralizationHierarchy
from oblitus.mondrian import enforce_k_anonymity_by_mondrian
from oblitus.plan import KAnonymityMethod, RangeStyle

K = 5
QUASI_IDENTIFIERS = ["age", "birth", "state", "gender"]


def _people() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = 240
    return pd.DataFrame(
        {
            "id": np.arange(rows),
            "age": rng.integers(18, 90, rows),
            "birth": pd.Timestamp("1950-01-01")
            + pd.to_timedelta(rng.integers(0, 15000, rows), "D"),
            "state": rng.choice(["SP", "RJ", "MG", "BA", "RS"], rows),
            "gender": rng.choice(["F", "M", None], rows),
            "diagnosis": rng.choice(["flu", "asthma", "diabetes"], rows),
        }
    )


def _native(frame: pd.DataFrame, engine: Engine):
    if engine is Engine.PANDAS:
        return frame
    if engine is Engine.POLARS:
        return pl.from_pandas(frame)
    return duckdb.connect().from_df(frame)


def _as_pandas(data) -> pd.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, pd.DataFrame):
        return data
    return data.df()


def _plan(**overrides) -> AnonymizationPlan:
    settings = {
        "fidelity": Fidelity.PRIVACY,
        "protection": Protection.ANONYMIZATION,
        "actions": [ColumnAction(column="diagnosis", technique="keep")],
        "row_key": "id",
        "quasi_identifiers": QUASI_IDENTIFIERS,
        "sensitive_attributes": ["diagnosis"],
        "k_target": K,
        "enforce_k_anonymity": True,
        "k_anonymity_method": KAnonymityMethod.MONDRIAN,
    }
    settings.update(overrides)
    return AnonymizationPlan(**settings)


def test_every_group_holds_at_least_k_rows_and_no_cell_is_blanked():
    frame = _people()

    result = Anonymizer(seed=1).apply(frame, _plan())

    assert result.privacy_after.k_anonymity >= K
    assert result.suppressed_cells == 0
    released = result.data
    assert released["age"].notna().all()
    assert released["state"].notna().all()


def test_numbers_are_replaced_by_the_cuts_that_bound_their_group():
    frame = pd.DataFrame({"age": [20, 22, 30, 31, 40, 41, 50, 52]})

    outcome = enforce_k_anonymity_by_mondrian(frame, ["age"], 2)

    assert outcome.data["age"].tolist() == [
        "under 30",
        "under 30",
        "30 to under 40",
        "30 to under 40",
        "40 to under 50",
        "40 to under 50",
        "50 or more",
        "50 or more",
    ]


def test_an_outlier_does_not_move_the_description_of_its_group():
    ages = [20] * 5 + [40] * 5 + [90]
    for style in (RangeStyle.RANGE, RangeStyle.MIDPOINT):
        with_outlier = pd.DataFrame({"age": ages})
        without_outlier = pd.DataFrame({"age": ages[:-1] + [41]})

        released = enforce_k_anonymity_by_mondrian(with_outlier, ["age"], K, style=style)
        neighbour = enforce_k_anonymity_by_mondrian(without_outlier, ["age"], K, style=style)

        assert released.data["age"].tolist() == neighbour.data["age"].tolist()
        assert "90" not in set(released.data["age"].astype(str))


def test_the_outermost_groups_are_open_ended():
    frame = pd.DataFrame({"age": [20] * 5 + [40] * 5 + [90]})

    released = enforce_k_anonymity_by_mondrian(frame, ["age"], K).data["age"]

    assert set(released) == {"under 40", "40 or more"}


def test_the_midpoint_style_publishes_the_group_median():
    frame = pd.DataFrame({"age": [20] * 5 + [40] * 5 + [90]})

    released = enforce_k_anonymity_by_mondrian(frame, ["age"], K, style=RangeStyle.MIDPOINT)

    assert set(released.data["age"]) == {20.0, 40.0}


def test_a_date_outlier_does_not_show_in_the_description():
    days = pd.to_datetime(["2000-01-01"] * 5 + ["2001-01-01"] * 5 + ["2090-01-01"])
    frame = pd.DataFrame({"birth": days})

    released = enforce_k_anonymity_by_mondrian(frame, ["birth"], K).data["birth"]

    assert set(released) == {"before 2001-01-01", "2001-01-01 or later"}


def test_the_midpoint_style_keeps_numbers_and_dates_as_numbers_and_dates():
    frame = _people()

    outcome = enforce_k_anonymity_by_mondrian(frame, ["age", "birth"], K, style=RangeStyle.MIDPOINT)

    released = outcome.data
    assert pd.api.types.is_float_dtype(released["age"])
    assert pd.api.types.is_datetime64_any_dtype(released["birth"])
    assert released.groupby(["age", "birth"]).size().min() >= K


def test_dates_are_written_as_the_cuts_that_bound_their_group():
    frame = _people()

    outcome = enforce_k_anonymity_by_mondrian(frame, ["birth"], K)

    texts = set(outcome.data["birth"])
    assert any(text.startswith("before 19") for text in texts)
    assert any(text.endswith(" or later") for text in texts)
    assert any(" to before " in text for text in texts)


def test_categories_become_the_set_of_their_group():
    frame = pd.DataFrame({"state": ["SP"] * 3 + ["RJ"] * 3 + ["BA"] * 2})

    outcome = enforce_k_anonymity_by_mondrian(frame, ["state"], 4)

    assert set(outcome.data["state"]) == {"{BA, RJ, SP}"}


def test_categories_become_their_closest_common_ancestor_with_a_hierarchy():
    states = ["SP"] * 3 + ["RJ"] * 3 + ["BA"] * 3 + ["PE"] * 3
    frame = pd.DataFrame({"state": states})
    regions = GeneralizationHierarchy(
        levels=[{"SP": "Southeast", "RJ": "Southeast", "BA": "Northeast", "PE": "Northeast"}]
    )

    outcome = enforce_k_anonymity_by_mondrian(frame, ["state"], 6, hierarchies={"state": regions})

    assert outcome.data["state"].tolist() == ["Southeast"] * 6 + ["Northeast"] * 6


def test_a_hierarchy_in_the_plan_describes_the_categories_of_a_group():
    frame = _people()
    regions = {"levels": [{"SP": "SE", "RJ": "SE", "MG": "SE", "BA": "NE", "RS": "S"}]}
    plan = _plan(quasi_identifiers=["state"], hierarchies={"state": regions}, k_target=40)

    result = Anonymizer(seed=1).apply(frame, plan)

    assert set(result.data["state"]) <= {"SE", "NE", "S", "*", "SP", "RJ", "MG", "BA", "RS"}
    assert result.privacy_after.k_anonymity >= 40


def test_local_suppression_stays_the_default_method():
    plan = AnonymizationPlan(fidelity=Fidelity.PRIVACY, actions=[])

    assert plan.k_anonymity_method is KAnonymityMethod.LOCAL_SUPPRESSION


def test_the_plan_round_trips_with_mondrian_settings():
    plan = _plan(range_style=RangeStyle.MIDPOINT, hierarchies={"state": {"preset": "cep"}})

    assert AnonymizationPlan.from_json(plan.to_json()) == plan


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DUCKDB])
def test_every_engine_releases_the_same_generalization(engine):
    frame = _people()
    expected = Anonymizer(seed=1).apply(frame, _plan(shuffle_rows=False)).data

    released = Anonymizer(seed=1).apply(_native(frame, engine), _plan(shuffle_rows=False)).data

    result = _as_pandas(released).sort_values("id").reset_index(drop=True)
    columns = QUASI_IDENTIFIERS
    for column in columns:
        assert result[column].astype(str).tolist() == expected[column].astype(str).tolist()
    assert assess_privacy(released, columns, k_target=K).meets_target


def test_the_gaps_of_a_group_take_its_description():
    frame = pd.DataFrame({"age": [20, 21, None, 40, 41, 42], "city": list("aabbbb")})

    outcome = enforce_k_anonymity_by_mondrian(frame, ["age"], 3)

    assert outcome.data["age"].isna().sum() == 0
    assert outcome.data["age"].value_counts().min() >= 3


def test_a_table_with_fewer_than_k_rows_becomes_one_group():
    frame = pd.DataFrame({"age": [20, 30], "state": ["SP", "RJ"]})

    outcome = enforce_k_anonymity_by_mondrian(frame, ["age", "state"], 5)

    assert outcome.groups == 1
    assert outcome.data["age"].nunique() == 1


def test_a_plan_naming_a_hierarchy_for_a_missing_column_is_refused():
    frame = _people()
    plan = _plan(hierarchies={"nowhere": {"preset": "cep"}})

    with pytest.raises(PlanError, match="nowhere"):
        Anonymizer(seed=1).apply(frame, plan)
