import pandas as pd
import pytest

from oblitus import AnonymizationPlan, Anonymizer, ColumnAction, Fidelity, PlanError
from oblitus.evaluation.dominance import dominant_segments
from oblitus.proposal import propose_plan


def _scramble(frame, **params):
    plan = AnonymizationPlan(
        fidelity=Fidelity.STATISTICAL,
        actions=[ColumnAction(column="segment", technique="scramble_categories", params=params)],
        shuffle_rows=False,
    )
    return Anonymizer(seed=1).apply(frame, plan).data["segment"]


def test_scramble_categories_merges_segments_with_too_few_entities():
    # Jewellery has many rows but only two stores, so it must not keep a code of its own.
    # Two stores are too few for OTHER as well, so the smallest segment joins it.
    frame = pd.DataFrame(
        {
            "store": [f"s{index % 6}" for index in range(60)]
            + [f"b{index % 7}" for index in range(70)]
            + ["j1", "j2"] * 36,
            "segment": ["Fashion"] * 60 + ["Books"] * 70 + ["Jewellery"] * 72,
        }
    )

    by_rows = _scramble(frame, prefix="SEG", min_count=3)
    by_entities = _scramble(frame, prefix="SEG", min_count=3, min_entities=3, by="store")

    assert by_rows.nunique() == 3
    assert set(by_entities) == {"SEG-01", "SEG-OTHER"}
    assert (by_entities[60:130] == "SEG-01").all()
    assert (by_entities[:60] == "SEG-OTHER").all()
    assert (by_entities[130:] == "SEG-OTHER").all()


def test_other_grows_until_it_holds_the_minimum_number_of_entities():
    segments = {"Big": 8, "Mid": 6, "Small": 2, "Tiny": 1}
    stores = [f"{name}{index}" for name, count in segments.items() for index in range(count)]
    frame = pd.DataFrame(
        {
            "store": stores * 4,
            "segment": [name for name, count in segments.items() for _ in range(count)] * 4,
        }
    )

    released = _scramble(frame, prefix="SEG", min_count=1, min_entities=5, by="store")

    in_other = released == "SEG-OTHER"
    assert frame[in_other].store.nunique() >= 5
    assert set(frame[in_other].segment) == {"Tiny", "Small", "Mid"}


def test_the_column_is_blanked_when_no_pool_can_reach_the_minimum():
    frame = pd.DataFrame({"store": ["a", "b"] * 10, "segment": ["x", "y"] * 10})

    released = _scramble(frame, prefix="SEG", min_entities=3, by="store")

    assert released.isna().all()


def test_min_entities_needs_the_entity_column():
    frame = pd.DataFrame({"segment": ["a", "b"]})

    with pytest.raises(PlanError, match="min_entities needs `by`"):
        _scramble(frame, min_entities=3)


def test_proposed_business_segments_count_entities():
    frame = pd.DataFrame(
        {
            "store_name": [f"Store {index % 8}" for index in range(80)],
            "segmento": ["Moda", "Casa"] * 40,
            "faturamento": range(80),
        }
    )
    anonymizer = Anonymizer()

    plan = propose_plan(anonymizer.scan(frame))

    params = plan.action_for("segmento").params
    assert params["min_entities"] == 3
    assert params["by"] == "store_name"


def test_a_segment_with_two_contributors_is_flagged_while_values_keep_their_units():
    values = pd.Series([100.0, 120.0, 90.0, 95.0, 105.0, 110.0, 400.0, 300.0])
    groups = pd.Series(["big"] * 6 + ["pair"] * 2)
    contributors = pd.Series(list("abcdefgh"))

    found = dominant_segments(values, groups, contributors, group_column="segment")

    assert [(segment.group, segment.rule) for segment in found] == [
        ("pair", "only 2 contributor(s)")
    ]
